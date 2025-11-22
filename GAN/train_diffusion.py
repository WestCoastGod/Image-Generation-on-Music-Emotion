"""
Emotion-Conditioned Diffusion Model for Landscape Generation
Stable alternative to GAN - no discriminator balance issues

Architecture: DDPM with VA conditioning
- No mode collapse
- Natural diversity through noise
- Stable training (simple MSE loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to avoid Tkinter threading errors
import matplotlib.pyplot as plt
import lpips



from Utils.dataloader import EmotionDataset


class SinusoidalPositionEmbeddings(nn.Module):
    """Timestep embeddings for diffusion"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SelfAttention(nn.Module):
    """Multi-head self-attention for global coherence (horizons, sky-land boundaries)"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0, f"channels {channels} not divisible by num_heads {num_heads}"
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x = self.norm(x)
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # [B, 3*C, H, W]
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, heads, HW, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = (C // self.num_heads) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = attn @ v  # [B, heads, HW, head_dim]
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.proj(out)
        
        return out + residual


class EmotionConditionedUNet(nn.Module):
    """UNet with VA emotion conditioning - simpler than GAN!"""
    def __init__(self, in_channels=3, out_channels=3, time_dim=256, emotion_dim=2):
        super().__init__()
        
        # Time and emotion embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        
        # Emotion embedding (v, a) for adaptive normalization
        self.emotion_embed = nn.Sequential(
            nn.Linear(emotion_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Encoder (downsample) - 5 stages with deeper channels and self-attention
        self.enc1 = self.conv_block(in_channels, 64, time_dim)     # 128×128 → 64×64
        self.enc2 = self.conv_block(64, 128, time_dim)             # 64×64 → 32×32
        self.enc3 = self.conv_block(128, 256, time_dim)            # 32×32 → 16×16
        self.attn3 = SelfAttention(256, num_heads=8)               # Self-attention at 16×16
        self.enc4 = self.conv_block(256, 512, time_dim)            # 16×16 → 8×8
        self.enc5 = self.conv_block(512, 1024, time_dim)           # 8×8 → 4×4 (deeper)
        
        # Bottleneck at 4×4 with max capacity
        self.bottleneck = self.conv_block(1024, 1024, time_dim)
        
        # Decoder (upsample) with skip connections
        self.dec5 = self.conv_block(1024 + 1024, 512, time_dim)   # bottleneck(1024) + enc5(1024)
        self.dec4 = self.conv_block(512 + 512, 256, time_dim)     # dec5(512) + enc4(512)
        self.attn4 = SelfAttention(256, num_heads=8)              # Self-attention at 16×16
        self.dec3 = self.conv_block(256 + 256, 128, time_dim)     # dec4(256) + enc3(256)
        self.dec2 = self.conv_block(128 + 128, 64, time_dim)      # dec3(128) + enc2(128)
        self.dec1 = self.conv_block(64 + 64, 64, time_dim)        # dec2(64) + enc1(64)
        
        # Final output
        self.final = nn.Conv2d(64, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def conv_block(self, in_ch, out_ch, time_dim):
        """Convolution block with residual connection - 2 ResNet blocks"""
        return nn.ModuleDict({
            # First ResNet block
            'conv1': nn.Conv2d(in_ch, out_ch, 3, padding=1),
            'norm1': nn.GroupNorm(8, out_ch),
            'conv2': nn.Conv2d(out_ch, out_ch, 3, padding=1),
            'norm2': nn.GroupNorm(8, out_ch),
            # Second ResNet block
            'conv3': nn.Conv2d(out_ch, out_ch, 3, padding=1),
            'norm3': nn.GroupNorm(8, out_ch),
            'conv4': nn.Conv2d(out_ch, out_ch, 3, padding=1),
            'norm4': nn.GroupNorm(8, out_ch),
            # Projection for residual if channels change
            'residual_proj': nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity(),
            # Time conditioning
            'time_emb': nn.Linear(time_dim, out_ch),
            # VA adaptive normalization
            'va_scale': nn.Linear(time_dim, out_ch),
            'va_shift': nn.Linear(time_dim, out_ch),
        })
    
    def apply_conv_block(self, x, block, time_emb, emotion_emb):
        """Apply convolution block with residual connections"""
        # Residual projection
        residual = block['residual_proj'](x)
        
        # First ResNet block
        h = block['conv1'](x)
        h = block['norm1'](h)
        h = F.silu(h)
        
        # Add time conditioning
        time_cond = block['time_emb'](time_emb)[:, :, None, None]
        h = h + time_cond
        
        h = block['conv2'](h)
        h = block['norm2'](h)
        
        # Residual connection
        h = F.silu(h + residual)
        
        # Second ResNet block with VA modulation
        residual2 = h
        h = block['conv3'](h)
        h = block['norm3'](h)
        
        # Apply VA as adaptive normalization with stronger influence
        va_scale = block['va_scale'](emotion_emb)[:, :, None, None]
        va_shift = block['va_shift'](emotion_emb)[:, :, None, None]
        h = h * (1 + va_scale * 0.3) + va_shift * 0.3
        
        h = F.silu(h)
        h = block['conv4'](h)
        h = block['norm4'](h)
        
        # Second residual connection
        h = F.silu(h + residual2)
        
        return h
    
    def forward(self, x, timestep, valence, arousal):
        """
        Args:
            x: Noisy image [B, 3, 128, 128]
            timestep: Diffusion timestep [B]
            valence: Valence values [B]
            arousal: Arousal values [B]
        Returns:
            Predicted noise [B, 3, 128, 128]
        """
        # Embeddings
        t_emb = self.time_mlp(timestep)
        
        # Emotion embedding for adaptive normalization
        emotion = torch.stack([valence, arousal], dim=1)
        e_emb = self.emotion_embed(emotion)  # [B, time_dim]
        
        # Encoder with skip connections (5 stages)
        e1 = self.apply_conv_block(x, self.enc1, t_emb, e_emb)                    # 128×128
        e2 = self.apply_conv_block(self.pool(e1), self.enc2, t_emb, e_emb)        # 64×64
        e3 = self.apply_conv_block(self.pool(e2), self.enc3, t_emb, e_emb)        # 32×32
        e3 = self.attn3(e3)                                                        # Self-attention at 16×16
        e4 = self.apply_conv_block(self.pool(e3), self.enc4, t_emb, e_emb)        # 16×16
        e5 = self.apply_conv_block(self.pool(e4), self.enc5, t_emb, e_emb)        # 8×8
        
        # Bottleneck at 4×4
        b = self.apply_conv_block(self.pool(e5), self.bottleneck, t_emb, e_emb)
        
        # Decoder with skip connections (5 stages)
        d5 = self.apply_conv_block(torch.cat([self.upsample(b), e5], dim=1), self.dec5, t_emb, e_emb)     # 8×8
        d4 = self.apply_conv_block(torch.cat([self.upsample(d5), e4], dim=1), self.dec4, t_emb, e_emb)    # 16×16
        d4 = self.attn4(d4)                                                                                # Self-attention at 16×16
        d3 = self.apply_conv_block(torch.cat([self.upsample(d4), e3], dim=1), self.dec3, t_emb, e_emb)    # 32×32
        d2 = self.apply_conv_block(torch.cat([self.upsample(d3), e2], dim=1), self.dec2, t_emb, e_emb)    # 64×64
        d1 = self.apply_conv_block(torch.cat([self.upsample(d2), e1], dim=1), self.dec1, t_emb, e_emb)    # 128×128
        
        return self.final(d1)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule - better for high-frequency details"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.001, 0.02)  # More reasonable range


def get_noise_schedule(timesteps=1000):
    """Prepare noise schedule for diffusion"""
    betas = cosine_beta_schedule(timesteps)
    
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    return {
        'betas': betas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
    }


def q_sample(x_start, t, noise, schedule):
    """Forward diffusion: add noise to image"""
    sqrt_alphas_cumprod_t = schedule['sqrt_alphas_cumprod'][t][:, None, None, None]
    sqrt_one_minus_alphas_cumprod_t = schedule['sqrt_one_minus_alphas_cumprod'][t][:, None, None, None]
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


@torch.no_grad()
def p_sample(model, x, t, t_index, v, a, schedule, device, guidance_scale=3.0):
    """Reverse diffusion: denoise one step with CFG"""
    betas_t = schedule['betas'][t][:, None, None, None]
    sqrt_one_minus_alphas_cumprod_t = schedule['sqrt_one_minus_alphas_cumprod'][t][:, None, None, None]
    sqrt_recip_alphas_t = torch.sqrt(1.0 / (1. - betas_t))
    
    # Classifier-free guidance: predict with and without conditioning
    if guidance_scale != 1.0:
        # Conditional prediction
        noise_cond = model(x, t, v, a)
        # Unconditional prediction (zero VA)
        noise_uncond = model(x, t, torch.zeros_like(v), torch.zeros_like(a))
        # Apply guidance
        predicted_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
    else:
        predicted_noise = model(x, t, v, a)
    
    # Denoise
    model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
    
    if t_index == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(betas_t) * noise


@torch.no_grad()
def sample_images(model, n_samples, v_values, a_values, schedule, timesteps, device, guidance_scale=3.0):
    """Generate images by reversing diffusion with CFG"""
    model.eval()
    
    # Start from random noise
    x = torch.randn(n_samples, 3, 128, 128, device=device)
    
    v = torch.tensor(v_values, device=device)
    a = torch.tensor(a_values, device=device)
    
    # Reverse diffusion with guidance
    for i in reversed(range(timesteps)):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)
        x = p_sample(model, x, t, i, v, a, schedule, device, guidance_scale)
    
    return x


def train_diffusion(
    data_root=r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Data\Landscape",
    va_file=r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Data\EmotionLabel\all_photos_valence_arousal.csv",
    epochs=500,
    batch_size=16,
    lr=0.0001,
    timesteps=1000,
    device="cuda"
):
    """Train emotion-conditioned diffusion model"""
    
    print("="*60)
    print("EMOTION-CONDITIONED DIFFUSION MODEL")
    print("="*60)
    print("Advantages over GAN:")
    print("  ✓ No discriminator - no balance issues")
    print("  ✓ No mode collapse - stable training")
    print("  ✓ Natural diversity - noise guarantees variation")
    print("  ✓ Simple MSE loss - easy to optimize")
    print(f"\nTraining: {epochs} epochs, timesteps={timesteps}")
    print("="*60 + "\n")
    
    # Model
    model = EmotionConditionedUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Perceptual loss (LPIPS) for sharper textures
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    for param in lpips_fn.parameters():
        param.requires_grad = False
    
    # Noise schedule
    schedule = get_noise_schedule(timesteps)
    for key in schedule:
        schedule[key] = schedule[key].to(device)
    
    # Dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = EmotionDataset(data_root, va_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=8, drop_last=True, pin_memory=True, 
                           prefetch_factor=4, persistent_workers=True)
    
    print(f"Dataset: {len(dataset)} images")
    print(f"Batch size: {batch_size}")
    print(f"IMPROVED MODEL:")
    print(f"  ✓ Self-attention @ 16×16 (global mood consistency)")
    print(f"  ✓ 1024 bottleneck channels (2× capacity)")
    print(f"  ✓ LPIPS perceptual loss (sharper details)")
    print(f"  ✓ VA scaling 0.3 (stronger emotion control)\n")
    
    os.makedirs("GAN/Samples_Diffusion", exist_ok=True)
    os.makedirs("GAN/Weights", exist_ok=True)
    
    # Loss tracking
    loss_history = []
    
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batch_idx, (images, v, a) in enumerate(pbar):
            images = images.to(device)
            v = v.to(device)
            a = a.to(device)
            
            # Classifier-free guidance: randomly drop VA 10% of time
            cfg_mask = torch.rand(v.shape[0], device=device) < 0.1  # 10% dropout
            v = torch.where(cfg_mask, torch.zeros_like(v), v)
            a = torch.where(cfg_mask, torch.zeros_like(a), a)
            
            # Random timestep for each image
            t = torch.randint(0, timesteps, (images.shape[0],), device=device).long()
            
            # Add noise
            noise = torch.randn_like(images)
            noisy_images = q_sample(images, t, noise, schedule)
            
            # Predict noise
            predicted_noise = model(noisy_images, t, v, a)
            
            # MSE loss for denoising
            mse_loss = F.mse_loss(predicted_noise, noise)
            
            # Perceptual loss: compute on detached tensors to save memory
            with torch.no_grad():
                sqrt_alphas_t = schedule['sqrt_alphas_cumprod'][t][:, None, None, None]
                sqrt_one_minus_t = schedule['sqrt_one_minus_alphas_cumprod'][t][:, None, None, None]
                denoised_pred = (noisy_images - sqrt_one_minus_t * predicted_noise.detach()) / sqrt_alphas_t
            perceptual_loss = lpips_fn(images, denoised_pred.clamp(-1, 1)).mean()
            
            # Combined loss: MSE (denoising) + 0.1 * LPIPS (perceptual guidance)
            loss = mse_loss + 0.1 * perceptual_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        # Generate samples every 50 epochs
        if epoch % 50 == 0:
            print(f"Generating samples at epoch {epoch}...")
            
            # 3×3 VA grid
            va_grid = [(0.2, 0.2), (0.2, 0.5), (0.2, 0.8),
                       (0.5, 0.2), (0.5, 0.5), (0.5, 0.8),
                       (0.8, 0.2), (0.8, 0.5), (0.8, 0.8)]
            
            v_vals = [v for v, a in va_grid]
            a_vals = [a for v, a in va_grid]
            
            samples = sample_images(model, 9, v_vals, a_vals, schedule, timesteps, device, guidance_scale=5.0)
            
            # Save grid
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            fig.suptitle(f"Epoch {epoch} - Diffusion Model", fontsize=16)
            
            for idx, (v_val, a_val) in enumerate(va_grid):
                img = samples[idx].cpu().permute(1, 2, 0)
                img = (img * 0.5 + 0.5).clamp(0, 1).numpy()
                
                row, col = idx // 3, idx % 3
                axes[row, col].imshow(img)
                axes[row, col].set_title(f"V={v_val}, A={a_val}")
                axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"GAN/Samples_Diffusion/epoch_{epoch:04d}.png")
            plt.close()
            print(f"✓ Saved samples")
        
        # Save checkpoint and plot loss
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"GAN/Weights/diffusion_epoch{epoch}.pth")
            print(f"✓ Saved checkpoint")
            
            # Plot loss curve
            plt.figure(figsize=(10, 5))
            plt.plot(loss_history)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Diffusion Training Loss')
            plt.grid(True)
            plt.savefig(f"GAN/Samples_Diffusion/loss_curve_epoch{epoch}.png")
            plt.close()
            print(f"✓ Saved loss plot")
    
    # Final save
    torch.save(model.state_dict(), "GAN/Weights/diffusion_final.pth")
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    train_diffusion(
        epochs=3500,
        batch_size=16,
        lr=0.0001,
        timesteps=1000,
        device=device
    )
