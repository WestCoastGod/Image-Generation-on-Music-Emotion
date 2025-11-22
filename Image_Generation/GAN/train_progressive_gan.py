"""
Progressive GAN Training with VAE Warm Start
============================================
Phase 1 (0-30 epochs): Warm start with reconstruction loss only
Phase 2 (31-60 epochs): Gradually introduce discriminator
Phase 3 (61+ epochs): Full adversarial training with VA conditioning
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.vae import VAEDecoder
from Models.discriminator import Discriminator
from Utils.dataloader import EmotionDataset
from Utils.losses import LPIPSPerceptualLoss
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np


def hinge_loss_dis(real_logits, fake_logits):
    """Hinge loss for discriminator (more stable than WGAN)."""
    real_loss = torch.mean(F.relu(1.0 - real_logits))
    fake_loss = torch.mean(F.relu(1.0 + fake_logits))
    return real_loss + fake_loss


def hinge_loss_gen(fake_logits):
    """Hinge loss for generator."""
    return -torch.mean(fake_logits)


def r1_gradient_penalty(real_images, real_logits, device):
    """R1 gradient penalty - simpler and more stable than WGAN-GP."""
    gradients = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=real_images,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    penalty = (gradients.norm(2, dim=1) ** 2).mean()
    return penalty


def va_diversity_loss(fake_imgs, va_values, batch_size):
    """
    Encourage visually distinct images for different VA values.
    Compute pairwise L2 distance in image space and VA space.
    If VA distance is large but image distance is small, penalize.
    """
    if batch_size < 2:
        return torch.tensor(0.0, device=fake_imgs.device)
    
    # Flatten images: [B, C, H, W] -> [B, C*H*W]
    flat_imgs = fake_imgs.view(batch_size, -1)
    
    # Pairwise image L2 distances
    img_dists = torch.cdist(flat_imgs, flat_imgs, p=2)  # [B, B]
    
    # Pairwise VA L2 distances
    va_dists = torch.cdist(va_values, va_values, p=2)  # [B, B]
    
    # Normalize distances to [0, 1]
    img_dists = img_dists / (img_dists.max() + 1e-8)
    va_dists = va_dists / (va_dists.max() + 1e-8)
    
    # Diversity loss: encourage img_dist to be proportional to va_dist
    # Use L1 instead of MSE for stronger penalty
    diversity_loss = F.l1_loss(img_dists, va_dists)
    
    return diversity_loss


def affective_feature_matching_loss(gen_eru_features, real_eru_features):
    """AFM Loss from paper: Compare emotion-affected features (attention maps)
    
    Args:
        gen_eru_features: Dict of {eru_name: (v_attn, a_attn)} from generator
        real_eru_features: Dict of {eru_name: (v_attn, a_attn)} from discriminator on real images
    
    Returns:
        afm_loss: Average L1 distance between attention maps across all ERUs
    """
    # Only compare the FINAL ERU (eru5) which exists at 128x128 resolution in both
    # Generator builds bottom-up: eru1@8x8, eru2@16x16, eru3@32x32, eru4@64x64, eru5@128x128
    # Discriminator processes top-down at stage=5: starts at 128x128 (becomes eru1 due to our indexing)
    # So generator's eru5 == discriminator's eru1, both at 128x128
    
    # Map generator's final ERU (eru5) to discriminator's first ERU (eru1)
    if 'eru5' in gen_eru_features and 'eru1' in real_eru_features:
        gen_v_attn, gen_a_attn = gen_eru_features['eru5']
        real_v_attn, real_a_attn = real_eru_features['eru1']
        
        # Verify shapes match (both should be at 128x128 resolution)
        if gen_v_attn.shape != real_v_attn.shape:
            print(f"[AFM WARNING] Shape mismatch: gen_eru5={gen_v_attn.shape} vs disc_eru1={real_v_attn.shape}")
            return torch.tensor(0.0, device=gen_v_attn.device)
        
        # L1 loss on valence attention maps (Eq. 4 in paper)
        loss_v = F.l1_loss(gen_v_attn, real_v_attn)
        
        # L1 loss on arousal attention maps (Eq. 5 in paper)
        loss_a = F.l1_loss(gen_a_attn, real_a_attn)
        
        return loss_v + loss_a
    else:
        # Fallback: return 0 if ERUs not found
        return torch.tensor(0.0, device=list(gen_eru_features.values())[0][0].device)


def train_progressive_gan(
    img_dir,
    label_csv,
    vae_checkpoint=None,
    epochs=100,
    batch_size=16,
    latent_dim=128,
    lr_gen=0.0002,
    lr_dis=0.0001,  # Slower discriminator (TTUR)
    device=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("Progressive GAN Training with VAE Warm Start")
    print("=" * 60)
    
    # Initialize generator from VAE decoder
    generator = VAEDecoder(latent_dim=latent_dim).to(device)
    
    # Load VAE weights if available
    if vae_checkpoint and os.path.exists(vae_checkpoint):
        print(f"Loading VAE decoder weights from {vae_checkpoint}")
        state_dict = torch.load(vae_checkpoint, map_location=device)
        # Extract only decoder weights
        decoder_state = {k.replace('decoder.', ''): v for k, v in state_dict.items() if 'decoder' in k}
        generator.load_state_dict(decoder_state, strict=False)
        print("VAE decoder loaded successfully!")
    else:
        print("No VAE checkpoint found, training from scratch")
        # Initialize from scratch with better init
        for m in generator.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    # Create discriminator (ERU scales now fixed per-stage, no initialization needed)
    discriminator = Discriminator().to(device)
    
    # Perceptual loss (LPIPS - strongly penalizes blur)
    perceptual_loss = LPIPSPerceptualLoss().to(device)
    
    # EMA generator for stable evaluation
    generator_ema = deepcopy(generator).eval()
    for param in generator_ema.parameters():
        param.requires_grad = False
    
    # Optimizers with different learning rates (TTUR)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_gen, betas=(0.0, 0.99))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_dis, betas=(0.0, 0.99))
    
    # Schedulers
    scheduler_G = optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.99)
    scheduler_D = optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.99)
    
    # Data loading
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Upgraded to 128x128
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    dataset = EmotionDataset(img_dir, label_csv, transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    
    print(f"Loaded {len(dataset)} images")
    print(f"Batch size: {batch_size}, Latent dim: {latent_dim}")
    print(f"LR_gen: {lr_gen}, LR_dis: {lr_dis} (TTUR)")
    
    # Training phases
    PHASE1_END = 30   # Reconstruction only
    PHASE2_END = 60   # Progressive adversarial
    
    os.makedirs("GAN/Samples_ProgressiveGAN", exist_ok=True)
    os.makedirs("GAN/Weights", exist_ok=True)
    
    best_loss = float('inf')
    
    # Training curves tracking
    history = {
        'g_loss': [], 'd_loss': [], 'recon_loss': [], 'va_loss': [],
        'perceptual_loss': [], 'diversity_loss': [], 'afm_loss': []
    }
    
    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        
        # Determine training phase
        if epoch <= PHASE1_END:
            phase = "PHASE 1: Reconstruction Only"
            adv_weight = 0.0
            recon_weight = 1.0
            train_discriminator = False
        elif epoch <= PHASE2_END:
            phase = "PHASE 2: Progressive Adversarial"
            # Gradually increase adversarial, decrease reconstruction
            progress = (epoch - PHASE1_END) / (PHASE2_END - PHASE1_END)
            adv_weight = 0.5 * progress  # Ramp up slowly
            recon_weight = 1.0 - 0.8 * progress  # Keep some reconstruction
            train_discriminator = True
        else:
            phase = "PHASE 3: Full Adversarial"
            adv_weight = 0.5  # Reduced from 1.0 to prevent mode collapse
            recon_weight = 0.3  # Increased to preserve VAE feature diversity
            train_discriminator = True
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_va_loss = 0.0
        epoch_perceptual_loss = 0.0
        epoch_diversity_loss = 0.0
        epoch_afm_loss = 0.0
        num_batches = 0
        
        for i, (real_imgs, v, a) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            v, a = v.to(device), a.to(device)
            batch = real_imgs.size(0)
            
            # Generate fake images (ERU features disabled since AFM loss incompatible)
            z = torch.randn(batch, latent_dim, device=device)
            fake_imgs = generator(z, v, a, return_eru_features=False)
            
            # ==================== Train Discriminator ====================
            if train_discriminator:
                optimizer_D.zero_grad()
                
                # Real images
                real_imgs.requires_grad_(True)
                real_logits, _, real_va_pred = discriminator(real_imgs, v, a, stage=5)  # Stage 5 for 128x128
                
                # Fake images (detach to not train generator)
                fake_logits, _, _ = discriminator(fake_imgs.detach(), v, a, stage=5)
                
                # Hinge loss
                d_loss = hinge_loss_dis(real_logits, fake_logits)
                
                # VA auxiliary loss on real images
                va_target = torch.stack([v, a], dim=1)
                va_loss_d = F.mse_loss(real_va_pred, va_target)
                
                # R1 gradient penalty (apply every 16 iterations)
                if i % 16 == 0:
                    r1_penalty = r1_gradient_penalty(real_imgs, real_logits, device)
                    d_loss = d_loss + 5.0 * r1_penalty  # Reduced for 128x128
                
                d_total_loss = d_loss + 0.1 * va_loss_d
                d_total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                optimizer_D.step()
                
                epoch_d_loss += d_loss.item()
            else:
                d_loss = torch.tensor(0.0)
                va_loss_d = torch.tensor(0.0)
            
            # ==================== Train Generator ====================
            optimizer_G.zero_grad()
            
            loss_components = []
            
            # Reconstruction loss (Phase 1 and early Phase 2)
            if recon_weight > 0:
                recon_loss = F.mse_loss(fake_imgs, real_imgs)
                loss_components.append(recon_weight * recon_loss)
                epoch_recon_loss += recon_loss.item()
            else:
                recon_loss = torch.tensor(0.0)
            
            # Perceptual loss - SOTA aggressive schedule for small dataset (1.3K images)
            # Pix2Pix uses L1 weight=100, we use LPIPS=0.5 for similar effect
            if epoch <= 60:
                lpips_weight = 0.0  # Phase 1-2: Pure adversarial
            elif epoch <= 100:
                # Rapid ramp to force realistic structure
                progress = (epoch - 60) / 40
                lpips_weight = 0.5 * progress  # Ramp 0→0.5 (50× stronger than original)
            elif epoch <= 300:
                lpips_weight = 0.5  # Strong perceptual guidance (SOTA level)
            else:
                lpips_weight = 0.3  # Reduce for fine-tuning
            
            percept_loss = perceptual_loss(fake_imgs, real_imgs)
            loss_components.append(lpips_weight * percept_loss)
            epoch_perceptual_loss += percept_loss.item()
            
            # Adversarial loss (Phase 2 and 3)
            if adv_weight > 0:
                # Get discriminator output (ERU features disabled since AFM loss incompatible)
                fake_logits, _, fake_va_pred = discriminator(fake_imgs, v, a, stage=5, return_eru_features=False)
                # No need for disc_real_eru since AFM loss is disabled
                
                # Hinge loss for generator
                g_adv_loss = hinge_loss_gen(fake_logits)
                
                # VA auxiliary loss - generator should produce images matching VA
                va_target = torch.stack([v, a], dim=1)
                g_va_loss = F.mse_loss(fake_va_pred, va_target)
                
                # VA diversity loss - encourage different VA → different images
                diversity_loss = va_diversity_loss(fake_imgs, va_target, batch)
                
                # AFM Loss disabled: Architecturally incompatible (gen@128x128 vs disc@64x64)
                # Compensate with stronger VA auxiliary loss instead
                # afm_loss = affective_feature_matching_loss(gen_eru_features, disc_real_eru)
                afm_loss = torch.tensor(0.0, device=device)
                
                # Conservative weights to prevent gradient explosion and preserve VAE diversity
                va_weight = 10.0  # Moderate VA conditioning
                diversity_weight = 2.0  # Minimal diversity enforcement (reconstruction handles diversity)
                
                loss_components.append(adv_weight * g_adv_loss)
                loss_components.append(va_weight * g_va_loss)
                loss_components.append(diversity_weight * diversity_loss)
                # loss_components.append(100.0 * afm_loss)  # AFM loss disabled
                
                epoch_va_loss += g_va_loss.item()
                epoch_diversity_loss += diversity_loss.item()
                if isinstance(afm_loss, torch.Tensor):
                    epoch_afm_loss += afm_loss.item()
                else:
                    epoch_afm_loss += 0.0
            else:
                g_adv_loss = torch.tensor(0.0, device=device)
                g_va_loss = torch.tensor(0.0, device=device)
                diversity_loss = torch.tensor(0.0, device=device)
                afm_loss = torch.tensor(0.0, device=device)
            
            g_total_loss = sum(loss_components)
            g_total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()
            
            # Update EMA generator
            with torch.no_grad():
                ema_decay = 0.999
                for ema_param, param in zip(generator_ema.parameters(), generator.parameters()):
                    ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
            
            epoch_g_loss += g_total_loss.item()
            num_batches += 1
            
            if i % 20 == 0:
                print(f"Epoch [{epoch}/{epochs}] {phase}")
                print(f"  Batch {i}: G_loss={g_total_loss.item():.4f}, D_loss={d_loss.item():.4f}, "
                      f"Recon={recon_loss.item():.4f}, VA={g_va_loss.item():.4f}")
                print(f"  Weights: Adv={adv_weight:.2f}, Recon={recon_weight:.2f}")
        
        # Epoch summary
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches if train_discriminator else 0.0
        avg_recon = epoch_recon_loss / num_batches if recon_weight > 0 else 0.0
        avg_va = epoch_va_loss / num_batches
        avg_percept = epoch_perceptual_loss / num_batches
        avg_diversity = epoch_diversity_loss / num_batches if adv_weight > 0 else 0.0
        avg_afm = epoch_afm_loss / num_batches if adv_weight > 0 else 0.0
        
        # Record history
        history['g_loss'].append(avg_g_loss)
        history['d_loss'].append(avg_d_loss)
        history['recon_loss'].append(avg_recon)
        history['va_loss'].append(avg_va)
        history['perceptual_loss'].append(avg_percept)
        history['diversity_loss'].append(avg_diversity)
        history['afm_loss'].append(avg_afm)
        
        if epoch % 20 == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{epochs} Summary - {phase}")
            print(f"G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}")
            print(f"Recon: {avg_recon:.4f}, VA: {avg_va:.4f}")
            print(f"Percept: {avg_percept:.4f}, Diversity: {avg_diversity:.4f}")
            print(f"AFM: {avg_afm:.4f}")
            print(f"{'='*60}\n")
        
        # Save samples every 10 epochs
        if epoch % 10 == 0:
            generator_ema.eval()
            with torch.no_grad():
                # Generate samples with different VA values
                save_dir = "GAN/Samples_ProgressiveGAN"
                fig, axes = plt.subplots(3, 3, figsize=(12, 12))
                fig.suptitle(f"Epoch {epoch} - Progressive GAN", fontsize=16)
                
                va_grid = [(0.2, 0.2), (0.2, 0.5), (0.2, 0.8),
                          (0.5, 0.2), (0.5, 0.5), (0.5, 0.8),
                          (0.8, 0.2), (0.8, 0.5), (0.8, 0.8)]
                
                for idx, (v_val, a_val) in enumerate(va_grid):
                    z = torch.randn(1, latent_dim, device=device)
                    v_t = torch.tensor([v_val], device=device)
                    a_t = torch.tensor([a_val], device=device)
                    
                    fake_img = generator_ema(z, v_t, a_t)
                    fake_img = fake_img.cpu().squeeze(0).permute(1, 2, 0)
                    fake_img = (fake_img * 0.5 + 0.5).clamp(0, 1).numpy()
                    
                    row, col = idx // 3, idx % 3
                    axes[row, col].imshow(fake_img)
                    axes[row, col].set_title(f"V={v_val}, A={a_val}")
                    axes[row, col].axis('off')
                
                plt.tight_layout()
                plt.savefig(f"{save_dir}/epoch_{epoch:04d}.png")
                plt.close()
                print(f"Samples saved to {save_dir}/epoch_{epoch:04d}.png")
            
            generator.train()
        
        # Save best model
        if avg_g_loss < best_loss:
            best_loss = avg_g_loss
            torch.save(generator.state_dict(), "GAN/Weights/progressive_gen_best.pth")
            torch.save(generator_ema.state_dict(), "GAN/Weights/progressive_gen_ema_best.pth")
            torch.save(discriminator.state_dict(), "GAN/Weights/progressive_dis_best.pth")
        
        # Learning rate decay
        if epoch > PHASE2_END:
            scheduler_G.step()
            scheduler_D.step()
    
    # Save final models
    torch.save(generator.state_dict(), "GAN/Weights/progressive_gen_final.pth")
    torch.save(generator_ema.state_dict(), "GAN/Weights/progressive_gen_ema_final.pth")
    torch.save(discriminator.state_dict(), "GAN/Weights/progressive_dis_final.pth")
    
    # Plot training curves
    print("\nGenerating training curves...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Progressive GAN Training Curves', fontsize=16)
    
    epochs_range = range(1, len(history['g_loss']) + 1)
    
    axes[0, 0].plot(epochs_range, history['g_loss'], 'b-', label='Generator Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Generator Loss')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    axes[0, 1].plot(epochs_range, history['d_loss'], 'r-', label='Discriminator Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Discriminator Loss')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    axes[0, 2].plot(epochs_range, history['recon_loss'], 'g-', label='Reconstruction Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_title('Reconstruction Loss')
    axes[0, 2].grid(True)
    axes[0, 2].legend()
    
    axes[1, 0].plot(epochs_range, history['va_loss'], 'm-', label='VA Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('VA Auxiliary Loss')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    axes[1, 1].plot(epochs_range, history['perceptual_loss'], 'c-', label='Perceptual Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Perceptual Loss')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    axes[1, 2].plot(epochs_range, history['diversity_loss'], 'y-', label='Diversity Loss')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].set_title('VA Diversity Loss')
    axes[1, 2].grid(True)
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('GAN/Samples_ProgressiveGAN/training_curves.png', dpi=150)
    plt.close()
    print("Training curves saved to GAN/Samples_ProgressiveGAN/training_curves.png")
    
    print("\n" + "="*60)
    print("Progressive GAN Training Complete!")
    print("="*60)


if __name__ == "__main__":
    # Using Landscape dataset (1,311 images) for faster testing iterations
    # Switch to All_photos (10,766) for final training
    img_dir = r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Data\Landscape - Copy"
    label_csv = r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Data\EmotionLabel\all_photos_valence_arousal.csv"
    vae_checkpoint = r"GAN\Weights\vae_final.pth"
    
    print("Starting Progressive GAN Training")
    print("Using VAE decoder as warm start for generator")
    
    train_progressive_gan(
        img_dir=img_dir,
        label_csv=label_csv,
        vae_checkpoint=vae_checkpoint,
        epochs=1000,
        batch_size=12,  # Reduced for smaller Landscape dataset (1,311 images)
        latent_dim=128,
        lr_gen=0.0005,  # Increased to StyleGAN2 level for better quality
        lr_dis=0.0001,  # Balanced ratio (5:1) for small dataset
    )
