"""
Paper-Accurate Progressive GAN Training
WGAN-GP + AFM Loss Implementation

Architecture:
- Generator: ERU + PixelNorm (paper) + PixelShuffle (SOTA)
- Discriminator: ERU + Spectral Norm (paper + SOTA)
- Loss: WGAN-GP (Eq. 6&7) + AFM λ=100 (Eq. 4&5) + LPIPS (SOTA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import numpy as np
from Models.generator import Generator
from Models.discriminator import Discriminator
from Utils.dataloader import EmotionDataset
from Utils.losses import LPIPSPerceptualLoss
from Utils.ema import EMA


def wgan_loss_dis(real_logits, fake_logits):
    """WGAN loss for discriminator (paper Eq. 6)"""
    return -torch.mean(real_logits) + torch.mean(fake_logits)


def wgan_loss_gen(fake_logits):
    """WGAN loss for generator (paper Eq. 7)"""
    return -torch.mean(fake_logits)


def wgan_gradient_penalty(discriminator, real_imgs, fake_imgs, v, a, stage, device):
    """WGAN-GP: Gradient penalty for Lipschitz constraint"""
    batch_size = real_imgs.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    interpolated = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
    
    disc_stage = 5 - stage
    d_interpolated, _, _ = discriminator(interpolated, v, a, stage=disc_stage, alpha=1.0, return_eru_features=False)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty


def afm_loss(real_eru_attentions, fake_eru_attentions):
    """Affective Feature Matching Loss (paper Eq. 4 & 5)
    
    Compares ERU attention maps (v, a) between real and fake images.
    λ = 100 in paper (very high weight to enforce emotion matching)
    """
    real_v_atts, real_a_atts = real_eru_attentions
    fake_v_atts, fake_a_atts = fake_eru_attentions
    
    if len(real_v_atts) == 0 or len(fake_v_atts) == 0:
        return torch.tensor(0.0, device=fake_v_atts[0].device if len(fake_v_atts) > 0 else 'cpu')
    
    # Lv: L1 distance of valence attention maps (Eq. 4)
    loss_v = 0.0
    for real_v, fake_v in zip(real_v_atts, fake_v_atts):
        loss_v += F.l1_loss(fake_v, real_v.detach())
    loss_v = loss_v / len(real_v_atts)
    
    # La: L1 distance of arousal attention maps (Eq. 5)
    loss_a = 0.0
    for real_a, fake_a in zip(real_a_atts, fake_a_atts):
        loss_a += F.l1_loss(fake_a, real_a.detach())
    loss_a = loss_a / len(real_a_atts)
    
    return loss_v + loss_a


def get_stage_from_resolution(resolution):
    """Convert resolution to stage number (4×4=0, 8×8=1, ..., 128×128=5)"""
    return int(np.log2(resolution)) - 2


def train_progressive_gan(
    data_root="GAN/Data/Landscape",
    va_file="GAN/Data/EmotionLabel/all_photos_valence_arousal.csv",
    batch_size=16,
    latent_dim=128,
    lr_g=0.0005,  # G needs much more power
    lr_d=0.000000005,  # 100,000× slower than G (extreme)
    n_critic=1,  # CRITICAL: Only 1 D update per G (balance)
    afm_lambda=100.0,  # Back to paper value - force VA learning
    lpips_weight=0.0001,  # Minimal
    gp_lambda=2.0,  # Very weak GP
    device="cuda"
):
    """
    Progressive GAN training with paper-accurate WGAN + AFM loss
    
    Paper: "Emotional Landscape Image Generation Using GANs" (2020)
    - WGAN-GP loss (Eq. 6 & 7)
    - AFM loss λ=100 (Eq. 4 & 5) - emotion-specific feature matching
    - ERUs in both generator and discriminator
    - Progressive training 4×4 → 128×128
    
    SOTA additions:
    - LPIPS perceptual loss for quality
    - PixelShuffle upsampling (zero artifacts)
    - Extreme TTUR (100× slower D)
    """
    
    # Progressive training schedule
    resolutions = [4, 8, 16, 32, 64, 128]
    stabilization_epochs_4x4 = 50
    stabilization_epochs = 100
    fade_in_epochs = 10
    
    print("="*60)
    print("PROGRESSIVE GAN - PAPER + SOTA")
    print("="*60)
    print("Architecture:")
    print("  Generator: ERU + PixelNorm + PixelShuffle")
    print("  Discriminator: ERU + Spectral Norm")
    print("  ERU count: k-1 ERUs at resolution 2^k × 2^k")
    print("\nLoss Components:")
    print(f"  WGAN-GP: λ={gp_lambda} (Lipschitz constraint)")
    print(f"  AFM Loss: λ={afm_lambda} (emotion feature matching)")
    print(f"  LPIPS: weight={lpips_weight} (weak blur prevention)")
    print(f"\nTraining: n_critic={n_critic}, TTUR={int(lr_g/lr_d)}×")
    print(f"High-res epochs: 64×64 and 128×128 train for 150 epochs (vs 100 for lower res)")
    print(f"CRITICAL: D learning rate extremely slow (0.0000001) to prevent gradient explosion")
    print("="*60)
    
    # Initialize models
    generator = Generator(z_dim=latent_dim, emotion_dim=2).to(device)
    discriminator = Discriminator().to(device)
    
    # EMA for generator
    ema_generator = EMA(generator, decay=0.999)
    
    # Optimizers (TTUR: D much slower)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.0, 0.99))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.0, 0.99))
    
    # LR Schedulers (gentle decay)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=100, gamma=0.95)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=100, gamma=0.90)
    
    # LPIPS perceptual loss
    lpips_loss = LPIPSPerceptualLoss().to(device)
    
    # Dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = EmotionDataset(data_root, va_file, transform=transform)
    
    # Print VA distribution
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    all_v = torch.tensor([dataset[i][1].item() for i in range(len(dataset))])
    all_a = torch.tensor([dataset[i][2].item() for i in range(len(dataset))])
    print(f"Total samples: {len(dataset)}")
    print(f"Valence: min={all_v.min():.3f}, max={all_v.max():.3f}, mean={all_v.mean():.3f}")
    print(f"Arousal: min={all_a.min():.3f}, max={all_a.max():.3f}, mean={all_a.mean():.3f}")
    print("="*60 + "\n")
    
    os.makedirs("GAN/Samples_WGAN_AFM", exist_ok=True)
    os.makedirs("GAN/Weights", exist_ok=True)
    
    global_epoch = 0
    loss_history = {
        'epochs': [],
        'g_loss': [],
        'd_loss': [],
        'afm_loss': [],
        'resolution': []
    }
    
    # Train each resolution stage
    for stage_idx, resolution in enumerate(resolutions):
        stage = get_stage_from_resolution(resolution)
        print(f"\n{'='*60}")
        print(f"STAGE {stage_idx}: Training at {resolution}×{resolution}")
        print(f"{'='*60}")
        
        # Adaptive batch size for 16GB VRAM
        if resolution == 4:
            stage_batch_size = 32
        elif resolution == 8:
            stage_batch_size = 24
        elif resolution == 16:
            stage_batch_size = 20
        elif resolution == 32:
            stage_batch_size = 16
        elif resolution == 64:
            stage_batch_size = 14
        else:
            stage_batch_size = 12
        
        stage_batch_size = min(stage_batch_size, len(dataset))
        dataloader = DataLoader(dataset, batch_size=stage_batch_size, shuffle=True, 
                               num_workers=4, drop_last=True, pin_memory=True, persistent_workers=True)
        print(f"Batch size: {stage_batch_size}")
        
        # Determine epochs for this stage (more time for high-resolution stages)
        if resolution == 4:
            stage_stab_epochs = stabilization_epochs_4x4
        elif resolution >= 64:
            stage_stab_epochs = 150  # 50% more training for difficult 64×64 and 128×128 stages
        else:
            stage_stab_epochs = stabilization_epochs
        
        # Phase 1: Fade-in new layer
        print(f"\n--- Phase 1: Fade-in {resolution}×{resolution} ({fade_in_epochs} epochs) ---")
        for fade_epoch in range(fade_in_epochs):
            alpha = fade_epoch / fade_in_epochs
            global_epoch += 1
            train_epoch(
                generator, discriminator, ema_generator,
                optimizer_G, optimizer_D, lpips_loss,
                dataloader, latent_dim, device,
                global_epoch, resolution, stage, alpha,
                n_critic, afm_lambda, lpips_weight, gp_lambda,
                is_fade_in=True,
                loss_history=loss_history,
                scheduler_G=scheduler_G,
                scheduler_D=scheduler_D
            )
        
        # Phase 2: Stabilization
        print(f"\n--- Phase 2: Stabilize {resolution}×{resolution} ({stage_stab_epochs} epochs) ---")
        for stab_epoch in range(stage_stab_epochs):
            global_epoch += 1
            train_epoch(
                generator, discriminator, ema_generator,
                optimizer_G, optimizer_D, lpips_loss,
                dataloader, latent_dim, device,
                global_epoch, resolution, stage, 1.0,
                n_critic, afm_lambda, lpips_weight, gp_lambda,
                is_fade_in=False,
                loss_history=loss_history,
                scheduler_G=scheduler_G,
                scheduler_D=scheduler_D
            )
            
            # Save checkpoint at end of stage
            if stab_epoch == stage_stab_epochs - 1:
                torch.save(generator.state_dict(), f"GAN/Weights/wgan_afm_gen_stage{stage_idx}.pth")
                torch.save(discriminator.state_dict(), f"GAN/Weights/wgan_afm_dis_stage{stage_idx}.pth")
                print(f"✓ Saved checkpoint for stage {stage_idx}")
                
                plot_loss_curves(loss_history, stage_idx, resolution)
            
            # Save latest (overwriting)
            torch.save(generator.state_dict(), "GAN/Weights/wgan_afm_gen_latest.pth")
            torch.save(discriminator.state_dict(), "GAN/Weights/wgan_afm_dis_latest.pth")
    
    plot_loss_curves(loss_history, 'final', 'all')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


def train_epoch(
    generator, discriminator, ema_generator,
    optimizer_G, optimizer_D, lpips_loss,
    dataloader, latent_dim, device,
    epoch, resolution, stage, alpha,
    n_critic, afm_lambda, lpips_weight, gp_lambda,
    is_fade_in,
    loss_history=None,
    scheduler_G=None,
    scheduler_D=None
):
    """Train one epoch with WGAN-GP + AFM loss"""
    
    generator.train()
    discriminator.train()
    
    epoch_g_loss = 0.0
    epoch_d_loss = 0.0
    epoch_afm_loss = 0.0
    num_batches = 0
    
    phase_name = f"Fade-in α={alpha:.2f}" if is_fade_in else "Stable"
    disc_stage = 5 - stage
    
    for batch_idx, (real_imgs, v, a) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        v = v.to(device)
        a = a.to(device)
        batch = real_imgs.size(0)
        
        # Downsample to current resolution
        if resolution < 128:
            real_imgs = F.interpolate(real_imgs, size=(resolution, resolution), 
                                     mode='bilinear', align_corners=False)
        
        # ==================== Train Discriminator (n_critic times) ====================
        for _ in range(n_critic):
            optimizer_D.zero_grad()
            
            # Generate fake images
            z = torch.randn(batch, latent_dim, device=device)
            fake_imgs = generator(z, v, a, stage=stage, alpha=alpha, return_eru_features=False)
            
            # Discriminator forward (collect ERU attentions for AFM)
            real_logits, real_eru_atts, _ = discriminator(real_imgs, v, a, stage=disc_stage, 
                                                          alpha=alpha, return_eru_features=True)
            fake_logits, _, _ = discriminator(fake_imgs.detach(), v, a, stage=disc_stage, 
                                             alpha=alpha, return_eru_features=False)
            
            # WGAN loss (paper Eq. 6)
            d_wgan_loss = wgan_loss_dis(real_logits, fake_logits)
            
            # WGAN Gradient Penalty
            gp = wgan_gradient_penalty(discriminator, real_imgs, fake_imgs.detach(), 
                                      v, a, stage, device)
            
            # Total D loss
            d_loss = d_wgan_loss + gp_lambda * gp
            
            if not torch.isnan(d_loss):
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                optimizer_D.step()
        
        # ==================== Train Generator ====================
        optimizer_G.zero_grad()
        
        # Generate fake images
        z = torch.randn(batch, latent_dim, device=device)
        fake_imgs = generator(z, v, a, stage=stage, alpha=alpha, return_eru_features=False)
        
        # Discriminator forward (collect ERU attentions for AFM)
        fake_logits, fake_eru_atts, _ = discriminator(fake_imgs, v, a, stage=disc_stage, 
                                                      alpha=alpha, return_eru_features=True)
        
        # Get real ERU attentions (need fresh forward pass)
        with torch.no_grad():
            _, real_eru_atts_for_afm, _ = discriminator(real_imgs, v, a, stage=disc_stage,
                                                        alpha=alpha, return_eru_features=True)
        
        # WGAN loss for generator (paper Eq. 7)
        g_wgan_loss = wgan_loss_gen(fake_logits)
        
        # AFM loss (paper Eq. 4 & 5) - λ=100
        afm_loss_value = afm_loss(real_eru_atts_for_afm, fake_eru_atts)
        
        # LPIPS perceptual loss (SOTA addition)
        if resolution >= 32:
            lpips_value = lpips_loss(fake_imgs, real_imgs)
        else:
            lpips_value = torch.tensor(0.0, device=device)
        
        # Total G loss (paper Eq. 6 with AFM, plus LPIPS for SOTA quality)
        g_loss = g_wgan_loss + afm_lambda * afm_loss_value + lpips_weight * lpips_value
        
        if not torch.isnan(g_loss):
            g_loss.backward()
            g_grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()
            
            # Update EMA
            ema_generator.update(generator)
            
            # Monitor gradients
            if batch_idx % 50 == 0:
                d_grad_norm = sum(p.grad.norm().item() if p.grad is not None else 0 
                                 for p in discriminator.parameters())
                print(f"  Grad norms - G: {g_grad_norm:.3f}, D: {d_grad_norm:.3f}")
        
        # Accumulate losses
        if not torch.isnan(g_loss):
            epoch_g_loss += g_loss.item()
        if not torch.isnan(d_loss):
            epoch_d_loss += d_loss.item()
        epoch_afm_loss += afm_loss_value.item()
        num_batches += 1
        
        # Print first batch
        if batch_idx == 0:
            print(f"Epoch [{epoch}] {resolution}×{resolution} {phase_name} - "
                  f"G={g_loss.item():.4f}, D={d_loss.item():.4f}, AFM={afm_loss_value.item():.4f}")
    
    # Save samples every 10 epochs
    if epoch % 10 == 0:
        save_sample_grid(ema_generator, latent_dim, device, epoch, resolution)
    
    # LR scheduling
    if scheduler_G is not None and epoch >= 200:
        scheduler_G.step()
    if scheduler_D is not None and epoch >= 200:
        scheduler_D.step()
    
    # Summary every 10 epochs
    if epoch % 10 == 0 and num_batches > 0:
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{resolution}×{resolution} Summary - {phase_name}")
        print(f"G_loss: {epoch_g_loss/num_batches:.4f}, D_loss: {epoch_d_loss/num_batches:.4f}")
        print(f"AFM_loss: {epoch_afm_loss/num_batches:.4f}")
        print(f"{'='*60}\n")
    
    # Record history
    if loss_history is not None and num_batches > 0:
        loss_history['epochs'].append(epoch)
        loss_history['g_loss'].append(epoch_g_loss / num_batches)
        loss_history['d_loss'].append(epoch_d_loss / num_batches)
        loss_history['afm_loss'].append(epoch_afm_loss / num_batches)
        loss_history['resolution'].append(resolution)


def save_sample_grid(ema_generator, latent_dim, device, epoch, resolution):
    """Save 3×3 grid using EMA weights"""
    import matplotlib.pyplot as plt
    from Models.generator import Generator
    
    # Create temp model with EMA weights
    temp_gen = Generator(z_dim=latent_dim, emotion_dim=2).to(device)
    for name, param in temp_gen.named_parameters():
        if name in ema_generator.shadow:
            param.data = ema_generator.shadow[name].clone()
    temp_gen.eval()
    
    with torch.no_grad():
        save_dir = "GAN/Samples_WGAN_AFM"
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle(f"Epoch {epoch} - {resolution}×{resolution} WGAN+AFM", fontsize=16)
        
        # 3×3 VA grid
        va_grid = [(0.2, 0.2), (0.2, 0.5), (0.2, 0.8),
                   (0.5, 0.2), (0.5, 0.5), (0.5, 0.8),
                   (0.8, 0.2), (0.8, 0.5), (0.8, 0.8)]
        
        z_samples = torch.randn(9, latent_dim, device=device)
        
        for idx, (v_val, a_val) in enumerate(va_grid):
            z = z_samples[idx:idx+1]
            v_t = torch.tensor([v_val], device=device)
            a_t = torch.tensor([a_val], device=device)
            
            fake_img = temp_gen(z, v_t, a_t, stage=get_stage_from_resolution(resolution), 
                               alpha=1.0, return_eru_features=False)
            
            if resolution < 128:
                fake_img = F.interpolate(fake_img, size=(128, 128), mode='nearest')
            
            fake_img = fake_img.cpu().squeeze(0).permute(1, 2, 0)
            fake_img = (fake_img * 0.5 + 0.5).clamp(0, 1).numpy()
            
            row, col = idx // 3, idx % 3
            axes[row, col].imshow(fake_img)
            axes[row, col].set_title(f"V={v_val}, A={a_val}")
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/epoch_{epoch:04d}_res{resolution}.png")
        plt.close()
        print(f"Samples saved to {save_dir}/epoch_{epoch:04d}_res{resolution}.png (EMA)")


def plot_loss_curves(loss_history, stage_idx, resolution):
    """Plot and save loss curves"""
    import matplotlib.pyplot as plt
    
    if len(loss_history['epochs']) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training Progress - Stage {stage_idx} ({resolution}×{resolution})', fontsize=16)
    
    epochs = loss_history['epochs']
    
    # G Loss
    axes[0, 0].plot(epochs, loss_history['g_loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Generator Loss')
    axes[0, 0].set_title('Generator Loss (WGAN)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # D Loss
    axes[0, 1].plot(epochs, loss_history['d_loss'], 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Discriminator Loss')
    axes[0, 1].set_title('Discriminator Loss (WGAN-GP)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # AFM Loss
    axes[1, 0].plot(epochs, loss_history['afm_loss'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AFM Loss')
    axes[1, 0].set_title('Affective Feature Matching Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # All combined
    axes[1, 1].plot(epochs, loss_history['g_loss'], 'b-', linewidth=2, label='G Loss', alpha=0.7)
    axes[1, 1].plot(epochs, loss_history['d_loss'], 'r-', linewidth=2, label='D Loss', alpha=0.7)
    axes[1, 1].plot(epochs, loss_history['afm_loss'], 'g-', linewidth=2, label='AFM Loss', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('All Losses Combined')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f"GAN/Samples_WGAN_AFM/loss_curves_stage{stage_idx}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved loss curves to loss_curves_stage{stage_idx}.png")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_progressive_gan(
        data_root=r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Data\Landscape",
        va_file=r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Data\EmotionLabel\all_photos_valence_arousal.csv",
        batch_size=16,
        latent_dim=128,
        lr_g=0.0005,
        lr_d=0.000000005,
        n_critic=1,
        afm_lambda=100.0,
        lpips_weight=0.0001,
        gp_lambda=2.0,
        device=device
    )
