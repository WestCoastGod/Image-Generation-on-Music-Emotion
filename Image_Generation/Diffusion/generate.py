import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    """Multi-head self-attention for global coherence"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x = self.norm(x)
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scale = (C // self.num_heads) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = attn @ v
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.proj(out)
        
        return out + residual


class EmotionConditionedUNet(nn.Module):
    """UNet with VA emotion conditioning for diffusion model"""
    def __init__(self, in_channels=3, out_channels=3, time_dim=256, emotion_dim=2):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        
        self.emotion_embed = nn.Sequential(
            nn.Linear(emotion_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        self.enc1 = self.conv_block(in_channels, 64, time_dim)
        self.enc2 = self.conv_block(64, 128, time_dim)
        self.enc3 = self.conv_block(128, 256, time_dim)
        self.attn3 = SelfAttention(256, num_heads=8)
        self.enc4 = self.conv_block(256, 512, time_dim)
        self.enc5 = self.conv_block(512, 1024, time_dim)
        
        self.bottleneck = self.conv_block(1024, 1024, time_dim)
        
        self.dec5 = self.conv_block(1024 + 1024, 512, time_dim)
        self.dec4 = self.conv_block(512 + 512, 256, time_dim)
        self.attn4 = SelfAttention(256, num_heads=8)
        self.dec3 = self.conv_block(256 + 256, 128, time_dim)
        self.dec2 = self.conv_block(128 + 128, 64, time_dim)
        self.dec1 = self.conv_block(64 + 64, 64, time_dim)
        
        self.final = nn.Conv2d(64, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def conv_block(self, in_ch, out_ch, time_dim):
        return nn.ModuleDict({
            'conv1': nn.Conv2d(in_ch, out_ch, 3, padding=1),
            'norm1': nn.GroupNorm(8, out_ch),
            'conv2': nn.Conv2d(out_ch, out_ch, 3, padding=1),
            'norm2': nn.GroupNorm(8, out_ch),
            'conv3': nn.Conv2d(out_ch, out_ch, 3, padding=1),
            'norm3': nn.GroupNorm(8, out_ch),
            'conv4': nn.Conv2d(out_ch, out_ch, 3, padding=1),
            'norm4': nn.GroupNorm(8, out_ch),
            'residual_proj': nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity(),
            'time_emb': nn.Linear(time_dim, out_ch),
            'va_scale': nn.Linear(time_dim, out_ch),
            'va_shift': nn.Linear(time_dim, out_ch),
        })
    
    def apply_conv_block(self, x, block, time_emb, emotion_emb):
        residual = block['residual_proj'](x)
        
        h = block['conv1'](x)
        h = block['norm1'](h)
        h = F.silu(h)
        
        time_cond = block['time_emb'](time_emb)[:, :, None, None]
        h = h + time_cond
        
        h = block['conv2'](h)
        h = block['norm2'](h)
        h = F.silu(h + residual)
        
        residual2 = h
        h = block['conv3'](h)
        h = block['norm3'](h)
        
        va_scale = block['va_scale'](emotion_emb)[:, :, None, None]
        va_shift = block['va_shift'](emotion_emb)[:, :, None, None]
        h = h * (1 + va_scale * 0.3) + va_shift * 0.3
        
        h = F.silu(h)
        h = block['conv4'](h)
        h = block['norm4'](h)
        h = F.silu(h + residual2)
        
        return h
    
    def forward(self, x, timestep, valence, arousal):
        t_emb = self.time_mlp(timestep)
        emotion = torch.stack([valence, arousal], dim=1)
        e_emb = self.emotion_embed(emotion)
        
        e1 = self.apply_conv_block(x, self.enc1, t_emb, e_emb)
        e2 = self.apply_conv_block(self.pool(e1), self.enc2, t_emb, e_emb)
        e3 = self.apply_conv_block(self.pool(e2), self.enc3, t_emb, e_emb)
        e3 = self.attn3(e3)
        e4 = self.apply_conv_block(self.pool(e3), self.enc4, t_emb, e_emb)
        e5 = self.apply_conv_block(self.pool(e4), self.enc5, t_emb, e_emb)
        
        b = self.apply_conv_block(self.pool(e5), self.bottleneck, t_emb, e_emb)
        
        d5 = self.apply_conv_block(torch.cat([self.upsample(b), e5], dim=1), self.dec5, t_emb, e_emb)
        d4 = self.apply_conv_block(torch.cat([self.upsample(d5), e4], dim=1), self.dec4, t_emb, e_emb)
        d4 = self.attn4(d4)
        d3 = self.apply_conv_block(torch.cat([self.upsample(d4), e3], dim=1), self.dec3, t_emb, e_emb)
        d2 = self.apply_conv_block(torch.cat([self.upsample(d3), e2], dim=1), self.dec2, t_emb, e_emb)
        d1 = self.apply_conv_block(torch.cat([self.upsample(d2), e1], dim=1), self.dec1, t_emb, e_emb)
        
        return self.final(d1)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.001, 0.02)


def get_noise_schedule(timesteps=1000):
    betas = cosine_beta_schedule(timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    return {
        'betas': betas,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
    }


@torch.no_grad()
def p_sample(model, x, t, t_index, v, a, schedule, device, guidance_scale=3.0):
    """Reverse diffusion: denoise one step with classifier-free guidance"""
    betas_t = schedule['betas'][t][:, None, None, None]
    sqrt_one_minus_alphas_cumprod_t = schedule['sqrt_one_minus_alphas_cumprod'][t][:, None, None, None]
    sqrt_recip_alphas_t = torch.sqrt(1.0 / (1. - betas_t))
    
    if guidance_scale != 1.0:
        noise_cond = model(x, t, v, a)
        noise_uncond = model(x, t, torch.zeros_like(v), torch.zeros_like(a))
        predicted_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
    else:
        predicted_noise = model(x, t, v, a)
    
    model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
    
    if t_index == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(betas_t) * noise


def generate_emotion_image(v, a, model_path, save_path=r"C:\Users\cxoox\Desktop\image.png", 
                          guidance_scale=5.0, timesteps=1000, seed=None):
    """Generate landscape image using trained diffusion model.
    
    Args:
        v: Valence value (1-9 scale, where 1=sad, 9=happy) - matches dataset scale
        a: Arousal value (1-9 scale, where 1=calm, 9=energetic) - matches dataset scale
        model_path: Path to trained diffusion model checkpoint (.pth file)
                   Use: GAN/Weights/diffusion_final.pth or GAN/Weights/diffusion_epoch*.pth
        save_path: Where to save the generated image
        guidance_scale: Classifier-free guidance strength (higher = stronger VA effect, 3-7 typical)
        timesteps: Number of diffusion steps (1000 for training, can use fewer for faster generation)
        seed: Random seed for reproducibility (optional)
    """
    # Normalize VA from [1,9] to [0,1] (same as training)
    v_norm = v / 9.0
    a_norm = a / 9.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    # Load model
    print(f"Loading diffusion model from {model_path}...")
    model = EmotionConditionedUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Prepare noise schedule
    schedule = get_noise_schedule(timesteps)
    for key in schedule:
        schedule[key] = schedule[key].to(device)
    
    # Start from random noise
    x = torch.randn(1, 3, 128, 128, device=device)
    v_tensor = torch.tensor([v_norm], device=device)
    a_tensor = torch.tensor([a_norm], device=device)
    
    print(f"Generating image with V={v:.1f}/9 (norm={v_norm:.2f}), A={a:.1f}/9 (norm={a_norm:.2f}), guidance={guidance_scale}...")
    
    # Reverse diffusion process
    for i in tqdm(reversed(range(timesteps)), desc="Denoising", total=timesteps):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        x = p_sample(model, x, t, i, v_tensor, a_tensor, schedule, device, guidance_scale)
    
    # Convert to image
    img = x.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)
    
    plt.imsave(save_path, img)
    print(f"✓ Generated image saved to {save_path}")
    print(f"  Valence: {v:.1f}/9 (1=sad, 9=happy)")
    print(f"  Arousal: {a:.1f}/9 (1=calm, 9=energetic)")


# Sample usage
if __name__ == "__main__":
    # Example: Generate images with different emotions
    
    # Path to your trained diffusion model
    # Use diffusion_final.pth (after full training) or diffusion_epoch*.pth (specific checkpoint)
    model_path = r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Weights\diffusion_epoch1650.pth"
    # Or use specific epoch: model_path = r"...\GAN\Weights\diffusion_epoch2000.pth"
    
    # Generate 9 images covering the VA space (3x3 grid)
    va_configs = [
        # Low valence (sad)
        (2.0, 2.0, "sad_calm.png"),           # Sad + Calm
        (2.0, 5.0, "sad_moderate.png"),       # Sad + Moderate arousal
        (2.0, 8.0, "sad_energetic.png"),      # Sad + Energetic
        
        # Medium valence (neutral)
        (5.0, 2.0, "neutral_calm.png"),       # Neutral + Calm
        (5.0, 5.0, "neutral_moderate.png"),   # Neutral + Moderate
        (5.0, 8.0, "neutral_energetic.png"),  # Neutral + Energetic
        
        # High valence (happy)
        (8.0, 2.0, "happy_calm.png"),         # Happy + Calm
        (8.0, 5.0, "happy_moderate.png"),     # Happy + Moderate
        (8.0, 8.0, "happy_energetic.png"),    # Happy + Energetic
    ]
    
    for v, a, filename in va_configs:
        save_path = rf"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Generated_Images\{filename}"
        print(f"\n{'='*50}")
        generate_emotion_image(
            v=v, a=a,
            model_path=model_path,
            save_path=save_path,
            guidance_scale=5.0,
            seed=None  # Different random noise for each
        )
    
    print("\n✅ Generation complete!")
    print("\nNOTE: Use trained diffusion model checkpoints:")
    print("  - GAN/Weights/diffusion_final.pth (final trained model)")
    print("  - GAN/Weights/diffusion_epoch*.pth (intermediate checkpoints)")
