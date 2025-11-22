"""
True Progressive GAN Training with ERU - Following Paper Architecture
Gradually grows from 4×4 → 8×8 → 16×16 → 32×32 → 64×64 → 128×128
Each stage trains to stability before adding next resolution layer

KEY ARCHITECTURE (Paper-Accurate + SOTA):
- Generator: ERU with PixelNorm (paper spec) + PixelShuffle (SOTA)
  * Conservative ERU scaling: m = x + 0.2*(v_att*x + a_att*x)
- Discriminator: ERUs in inverted order (paper spec) + Spectral Norm (SOTA)
- Loss: WGAN-GP (paper) + AFM Loss λ=100 (paper) + LPIPS (SOTA)
- Training: Extreme TTUR (D 100× slower) + Minibatch discrimination
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import os
from tqdm import tqdm
import numpy as np
from Models.generator import Generator
from Models.discriminator import Discriminator
from Utils.dataloader import EmotionDataset
from Utils.losses import LPIPSPerceptualLoss
from Utils.ema import EMA
# from Utils.diff_augment import DiffAugment # Removed as per user request


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
    d_interpolated, _, _ = discriminator(interpolated, v, a, stage=disc_stage, alpha=1.0)
    
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
    vae_weights="GAN/Weights/vae_best.pth",
    batch_size=16,   # Increased for 16GB VRAM
    latent_dim=128,  # Matches VAE
    lr_g=0.0001,  # Generator LR (paper baseline)
    lr_d=0.000001,  # EXTREME TTUR: D 100× slower (prevents dominance)
    n_critic=5,  # Train D 5 times per G update (WGAN standard)
    afm_lambda=100.0,  # Paper's λ for AFM loss
    device="cuda"
):
    """
    True progressive training following paper:
    - WGAN-GP loss (paper Eq. 6 & 7)
    - AFM loss λ=100 (paper Eq. 4 & 5)
    - ERUs in both G and D
    - PixelNorm in G, Spectral Norm in D
    """
    
    # Progressive training schedule - EXTENDED for better quality
    resolutions = [4, 8, 16, 32, 64, 128]  # Progressive stages
    # Increased epochs significantly for vivid, clear results
    # More training time = better feature learning and convergence
    stabilization_epochs_4x4 = 50  # More time at base resolution
    stabilization_epochs = 100  # 2x longer per stage for sharp details
    fade_in_epochs = 10  
    
    print("="*60)
    print("PROGRESSIVE GAN - PAPER ARCHITECTURE + SOTA")
    print("Loss: WGAN-GP + AFM (λ=100) + LPIPS")
    print("Generator: ERU + PixelNorm")
    print("Discriminator: ERU + Spectral Norm")
    print(f"Training: EXTREME TTUR (D {int(lr_g/lr_d)}× slower)")
    print("="*60)
    print(f"Training schedule (LONGER for clarity):")
    print(f"  Stage 0: 4×4 - {fade_in_epochs} fade-in + {stabilization_epochs_4x4} stable epochs")
    for i, res in enumerate(resolutions[1:], 1):
        print(f"  Stage {i}: {res}×{res} - {fade_in_epochs} fade-in + {stabilization_epochs} stable epochs")
    total_epochs = (fade_in_epochs + stabilization_epochs_4x4) + 5 * (fade_in_epochs + stabilization_epochs)
    print(f"Total: {total_epochs} epochs (~8-10 hours)")
    print("="*60)
    
    # Initialize models (full architecture, but will use progressively)
    generator = Generator(z_dim=latent_dim, emotion_dim=2).to(device)
    discriminator = Discriminator().to(device)
    
    print("\nArchitecture:")
    print("  Generator: ERU + PixelNorm (paper) + PixelShuffle (SOTA)")
    print("  Discriminator: ERU + Spectral Norm (paper + SOTA)")
    print("  ERU count: k-1 ERUs at resolution 2^k × 2^k")
    print("\nLoss Components:")
    print(f"  WGAN-GP: Standard Wasserstein loss")
    print(f"  AFM Loss: λ={afm_lambda} (emotion-specific feature matching)")
    print(f"  LPIPS: 0.01 weight (perceptual quality boost)")
    
    # EMA for generator
    ema_generator = EMA(generator, decay=0.999)
    
    # Optimizers with TTUR (Two Time-Scale Update Rule)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.0, 0.99))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.0, 0.99))
    
    # LR Schedulers - decay discriminator LR to prevent dominance
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=100, gamma=0.95)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=100, gamma=0.90)
    
    # Perceptual loss
    lpips_loss = LPIPSPerceptualLoss().to(device)
    # afm_loss_fn = AFMLoss().to(device) # Removed as we switched to AdaIN
    
    # Dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Always load at 128×128
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = EmotionDataset(data_root, va_file, transform=transform)
    
    # Print VA distribution statistics
    print("\n" + "="*60)
    print("DATASET VA DISTRIBUTION")
    print("="*60)
    all_v = torch.tensor([dataset[i][1].item() for i in range(len(dataset))])
    all_a = torch.tensor([dataset[i][2].item() for i in range(len(dataset))])
    print(f"Total samples: {len(dataset)}")
    print(f"Valence: min={all_v.min():.3f}, max={all_v.max():.3f}, mean={all_v.mean():.3f}")
    print(f"Arousal: min={all_a.min():.3f}, max={all_a.max():.3f}, mean={all_a.mean():.3f}")
    
    # Quadrant distribution
    high_v_high_a = ((all_v > 0.5) & (all_a > 0.5)).sum().item()
    low_v_high_a = ((all_v <= 0.5) & (all_a > 0.5)).sum().item()
    low_v_low_a = ((all_v <= 0.5) & (all_a <= 0.5)).sum().item()
    high_v_low_a = ((all_v > 0.5) & (all_a <= 0.5)).sum().item()
    print(f"\nQuadrant distribution:")
    print(f"  High V, High A: {high_v_high_a} ({100*high_v_high_a/len(dataset):.1f}%)")
    print(f"  Low V, High A:  {low_v_high_a} ({100*low_v_high_a/len(dataset):.1f}%)")
    print(f"  Low V, Low A:   {low_v_low_a} ({100*low_v_low_a/len(dataset):.1f}%)")
    print(f"  High V, Low A:  {high_v_low_a} ({100*high_v_low_a/len(dataset):.1f}%)")
    print("="*60 + "\n")
    
    os.makedirs("GAN/Samples_TrueProgressiveGAN", exist_ok=True)
    os.makedirs("GAN/Weights", exist_ok=True)
    
    best_loss = float('inf')
    global_epoch = 0
    
    # Loss history for plotting
    loss_history = {
        'epochs': [],
        'g_loss': [],
        'd_loss': [],
        'recon_loss': [],
        'resolution': []
    }
    
    # RESTART FROM STAGE: Set to resume from specific stage after architecture change
    # Set to 0 for full training, or stage index to resume (e.g., 2 for 16×16)
    START_STAGE_IDX = 0  # START FROM SCRATCH - previous checkpoints collapsed
    
    # If resuming, load the checkpoint from previous stage
    if START_STAGE_IDX > 0:
        prev_stage_idx = START_STAGE_IDX - 1
        prev_checkpoint_gen = f"GAN/Weights/true_progressive_gen_stage{prev_stage_idx}.pth"
        prev_checkpoint_dis = f"GAN/Weights/true_progressive_dis_stage{prev_stage_idx}.pth"
        
        if os.path.exists(prev_checkpoint_gen):
            print(f"\n{'='*60}")
            print(f"RESUMING FROM STAGE {prev_stage_idx} CHECKPOINT")
            print(f"{'='*60}")
            generator.load_state_dict(torch.load(prev_checkpoint_gen, map_location=device), strict=False)
            discriminator.load_state_dict(torch.load(prev_checkpoint_dis, map_location=device), strict=False)
            print(f"✓ Loaded stage {prev_stage_idx} weights (strict=False for new ResBlocks)")
            # Calculate global_epoch offset
            # Stage 0: fade_in_epochs + stabilization_epochs_4x4
            # Stage 1+: fade_in_epochs + stabilization_epochs per stage
            for i in range(START_STAGE_IDX):
                if i == 0:
                    global_epoch += fade_in_epochs + stabilization_epochs_4x4
                else:
                    global_epoch += fade_in_epochs + stabilization_epochs
            print(f"✓ Starting from global epoch {global_epoch}")
        else:
            print(f"Warning: Checkpoint {prev_checkpoint_gen} not found. Starting from scratch.")
            START_STAGE_IDX = 0
    
    # Train each resolution stage
    for stage_idx, resolution in enumerate(resolutions):
        # Skip stages before START_STAGE_IDX
        if stage_idx < START_STAGE_IDX:
            continue
            
        stage = get_stage_from_resolution(resolution)
        print(f"\n{'='*60}")
        print(f"STAGE {stage_idx}: Training at {resolution}×{resolution}")
        print(f"{'='*60}")
        
        # Adaptive batch size for 16GB VRAM - larger batches for stability
        if resolution == 4:
            stage_batch_size = 32  # 32 at 4×4 (fast, stable)
        elif resolution == 8:
            stage_batch_size = 24  # 24 at 8×8
        elif resolution == 16:
            stage_batch_size = 20  # 20 at 16×16
        elif resolution == 32:
            stage_batch_size = 16  # 16 at 32×32
        elif resolution == 64:
            stage_batch_size = 14  # 14 at 64×64
        else:
            stage_batch_size = 12  # 12 at 128×128
        stage_batch_size = min(stage_batch_size, len(dataset))  # Don't exceed dataset size
        dataloader = DataLoader(dataset, batch_size=stage_batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, persistent_workers=True)
        print(f"Batch size for {resolution}×{resolution}: {stage_batch_size}")
        
        # Determine epochs for this stage
        stage_stab_epochs = stabilization_epochs_4x4 if resolution == 4 else stabilization_epochs
        
        # Phase 1: Fade-in new layer
        print(f"\n--- Phase 1: Fade-in {resolution}×{resolution} layer ({fade_in_epochs} epochs) ---")
        for fade_epoch in range(fade_in_epochs):
            alpha = fade_epoch / fade_in_epochs  # 0 → 1
            global_epoch += 1
            train_epoch(
                generator, discriminator, ema_generator,
                optimizer_G, optimizer_D, lpips_loss,
                dataloader, latent_dim, device,
                global_epoch, resolution, stage, alpha,
                is_fade_in=True,
                loss_history=loss_history,
                scheduler_G=scheduler_G,
                scheduler_D=scheduler_D
            )
        
        # Phase 2: Stabilization at current resolution
        print(f"\n--- Phase 2: Stabilize {resolution}×{resolution} ({stage_stab_epochs} epochs) ---")
        for stab_epoch in range(stage_stab_epochs):
            global_epoch += 1
            train_epoch(
                generator, discriminator, ema_generator,
                optimizer_G, optimizer_D, lpips_loss,
                dataloader, latent_dim, device,
                global_epoch, resolution, stage, alpha,
                is_fade_in=False,
                loss_history=loss_history,
                scheduler_G=scheduler_G,
                scheduler_D=scheduler_D
            )
            
            # Save checkpoint at end of each stage
            if stab_epoch == stage_stab_epochs - 1:
                torch.save(generator.state_dict(), f"GAN/Weights/true_progressive_gen_stage{stage_idx}.pth")
                torch.save(discriminator.state_dict(), f"GAN/Weights/true_progressive_dis_stage{stage_idx}.pth")
                print(f"✓ Saved checkpoint for stage {stage_idx} ({resolution}×{resolution})")
                
                # Plot and save loss curves for this stage
                plot_loss_curves(loss_history, stage_idx, resolution)
            
            # Save latest checkpoint every epoch (overwriting) to allow resume without wasting space
            torch.save(generator.state_dict(), "GAN/Weights/true_progressive_gen_latest.pth")
            torch.save(discriminator.state_dict(), "GAN/Weights/true_progressive_dis_latest.pth")
    
    # Save final complete loss curves
    plot_loss_curves(loss_history, stage_idx='final', resolution='all')
    
    print("\n" + "="*60)
    print("TRUE PROGRESSIVE TRAINING COMPLETE!")
    print("="*60)


def train_epoch(
    generator, discriminator, ema_generator,
    optimizer_G, optimizer_D, lpips_loss,
    dataloader, latent_dim, device,
    epoch, resolution, stage, alpha,
    is_fade_in,
    loss_history=None,
    scheduler_G=None,
    scheduler_D=None
):
    """Train one epoch at current resolution with optional fade-in"""
    
    generator.train()
    discriminator.train()
    
    epoch_g_loss = 0.0
    epoch_d_loss = 0.0
    epoch_recon_loss = 0.0
    epoch_va_loss = 0.0
    # epoch_afm_loss = 0.0 # Removed
    num_batches = 0
    
    phase_name = f"Fade-in α={alpha:.2f}" if is_fade_in else "Stable"
    
    # DiffAugment policy: Disabled (User request: emotion sensitive to color, physics sensitive to shifts)
    # policy = ''
    
    for batch_idx, (real_imgs, v, a) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        v = v.to(device)
        a = a.to(device)
        
        # Check for NaNs in data
        if torch.isnan(real_imgs).any() or torch.isnan(v).any() or torch.isnan(a).any():
            print(f"Warning: NaN detected in input data at batch {batch_idx}! Skipping...")
            continue
            
        batch = real_imgs.size(0)
        
        # Downsample images to current resolution
        if resolution < 128:
            real_imgs = F.interpolate(real_imgs, size=(resolution, resolution), mode='bilinear', align_corners=False)
        
        # Generate latent codes
        z = torch.randn(batch, latent_dim, device=device)
        
        # ==================== Train Discriminator ====================
        optimizer_D.zero_grad()
        
        # Generate fake images
        # Use progressive generation (no manual interpolation needed)
        fake_imgs = generator(z, v, a, stage=stage, alpha=alpha, return_eru_features=False)
        
        # Apply DiffAugment to BOTH real and fake images
        real_imgs_aug = real_imgs # DiffAugment(real_imgs, policy=policy)
        fake_imgs_aug = fake_imgs # DiffAugment(fake_imgs, policy=policy)
        
        # Discriminator predictions at current stage
        # Map training stage (0=4x4...5=128x128) to discriminator stage (5=4x4...0=128x128)
        disc_stage = 5 - stage
        
        real_logits, real_features, real_va_pred = discriminator(real_imgs_aug, v, a, stage=disc_stage, alpha=alpha, return_eru_features=True)
        fake_logits, fake_features_d, _ = discriminator(fake_imgs_aug.detach(), v, a, stage=disc_stage, alpha=alpha, return_eru_features=True)
        
        # Hinge loss - more stable than WGAN
        d_loss = hinge_loss_dis(real_logits, fake_logits)
        
        # VA auxiliary loss on real images
        va_target = torch.stack([v, a], dim=1)
        d_va_loss = F.mse_loss(real_va_pred, va_target)
        
        # Gradient Penalty (R1) - Essential for stability
        # Compute on real images (augmented)
        real_imgs_aug.requires_grad = True
        real_logits_grad, _, _ = discriminator(real_imgs_aug, v, a, stage=disc_stage, alpha=alpha)
        
        # R1 penalty applied every 16 iterations (like successful 64×64 GAN)
        if batch_idx % 16 == 0:
            grads = torch.autograd.grad(outputs=real_logits_grad.sum(), inputs=real_imgs_aug, create_graph=True, retain_graph=True)[0]
            
            # Safe norm calculation
            # Flatten gradients: [B, C, H, W] -> [B, -1]
            grads = grads.view(grads.size(0), -1)
            # Add epsilon inside norm to prevent NaN if grad is 0 (unlikely but safe)
            grad_norm = torch.sqrt(torch.sum(grads ** 2, dim=1) + 1e-8)
            grad_penalty = (grad_norm ** 2).mean()
        else:
            grad_penalty = torch.tensor(0.0, device=device)
        
        # Safety check for NaN
        if torch.isnan(d_loss) or torch.isnan(grad_penalty):
            print(f"Warning: NaN detected in D loss. d_loss={d_loss.item()}, gp={grad_penalty.item()}")
            print(f"  Debug D: real_logits_mean={real_logits.mean().item()}, fake_logits_mean={fake_logits.mean().item()}")
            print(f"  Debug D: real_logits_grad_norm={real_logits_grad.norm().item()}")
            # Zero out loss to prevent crash, but don't update
            d_total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            # Reset optimizer state to clear bad momentum
            optimizer_D.zero_grad()
        else:
            # R1 weight = 5.0 (stronger regularization to prevent D dominance at high res)
            gp_weight = 5.0
            d_total_loss = d_loss + 0.5 * d_va_loss + gp_weight * grad_penalty
        
        optimizer_D.zero_grad()
        if not torch.isnan(d_total_loss) and d_total_loss.item() != 0.0:
            d_total_loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_D.step()
        
        # ==================== Train Generator ====================
        optimizer_G.zero_grad()
        
        # Generate fake images
        fake_imgs = generator(z, v, a, stage=stage, alpha=alpha, return_eru_features=False)
        
        # Augment fake images for discriminator
        fake_imgs_aug = fake_imgs # DiffAugment(fake_imgs, policy=policy)
        
        # Discriminator predictions - get features for feature matching
        fake_logits, fake_features_g, fake_va_pred = discriminator(fake_imgs_aug, v, a, stage=disc_stage, alpha=alpha, return_eru_features=True)
        
        # Hinge loss for generator
        g_adv_loss = hinge_loss_gen(fake_logits)
        
        # Feature Matching Loss - match discriminator intermediate features
        # This teaches generator to create images with similar structure to real images
        # WITHOUT requiring pixel-perfect reconstruction from random z
        feature_match_loss = torch.tensor(0.0, device=device)
        if real_features is not None and fake_features_g is not None:
            # Match features at each discriminator layer
            for real_feat, fake_feat in zip(real_features, fake_features_g):
                feature_match_loss += F.l1_loss(fake_feat.mean(0), real_feat.mean(0).detach())
            feature_match_loss = feature_match_loss / len(real_features)
        
        # Remove pixel-level reconstruction (meaningless with random z)
        # Remove LPIPS (also comparing unmatched images)
        
        # VA auxiliary loss
        g_va_loss = F.mse_loss(fake_va_pred, va_target)
        
        # Diversity loss
        diversity_loss = va_diversity_loss(fake_imgs, va_target, batch)
        
        # Color diversity loss - prevent gray collapse
        color_loss = color_diversity_loss(fake_imgs)
        
        # AFM Loss (New) - REMOVED
        # Disable AFM during fade-in to prevent instability
        # if is_fade_in:
        #     afm_loss = torch.tensor(0.0, device=device)
        # else:
        #     afm_loss = afm_loss_fn(real_attentions, fake_attentions)
        #     # Safety clamp for AFM
        #     if torch.isnan(afm_loss) or afm_loss.item() > 100:
        #         print(f"Warning: AFM loss explosion/NaN: {afm_loss.item()}")
        #         afm_loss = torch.tensor(0.0, device=device)
        
        # Loss balance - PURE CONDITIONAL GAN with feature matching
        # Goal: Learn landscape structures from data, V/A associations naturally
        # NO pixel reconstruction (random z can't match specific real image)
        stage_progress = stage / 5.0  # 0 to 1
        
        # Resolution-based feature matching - reduce at higher resolutions
        if resolution <= 16:
            feature_weight = 10.0  # Strong at low res for structure learning
        elif resolution <= 64:
            feature_weight = 5.0   # Moderate at mid res
        else:
            feature_weight = 2.0   # Low at high res for detail freedom
        
        # Weights carefully balanced:
        adv_weight = 1.0           # Primary: fool discriminator
        va_weight = 3.0            # Strong: V/A conditioning via ERU attention
        diversity_weight = 1.0     # Strong: prevent mode collapse
        color_diversity_weight = 2.0  # Strong: prevent gray collapse
        
        # Progressive LPIPS for vivid colors and sharp textures (EARLIER start)
        # LPIPS requires minimum 32×32 resolution (AlexNet architecture limitation)
        # Start after 50 epochs (earlier) AND at 32×32+ resolution
        if epoch < 50 or resolution < 32:
            lpips_weight = 0.0  # Pure adversarial + feature matching first
        elif epoch < 150:
            # Gradual ramp from 0 to 0.02 over 100 epochs (higher weight)
            progress = (epoch - 50) / 100
            lpips_weight = 0.02 * progress
        else:
            lpips_weight = 0.02  # Full perceptual loss for vivid details (4x stronger)
        
        # Compute LPIPS loss if needed
        if lpips_weight > 0:
            # Compare fake images to real images in batch (perceptual similarity)
            lpips_value = lpips_loss(fake_imgs, real_imgs)
        else:
            lpips_value = torch.tensor(0.0, device=device)
        
        g_total_loss = (
            adv_weight * g_adv_loss +              # Fool discriminator
            feature_weight * feature_match_loss +  # Match real landscape structure
            va_weight * g_va_loss +                # V/A conditioning
            diversity_weight * diversity_loss +    # Prevent identical outputs
            color_diversity_weight * color_loss +  # Prevent gray collapse
            lpips_weight * lpips_value             # Perceptual realism (progressive)
        )
        
        if torch.isnan(g_total_loss):
             print(f"Warning: NaN detected in G loss. g_adv={g_adv_loss.item()}, FM={feature_match_loss.item()}")
             print(f"  Debug G: fake_logits_mean={fake_logits.mean().item()}")
             # Skip step and reset optimizer
             optimizer_G.zero_grad()
        else:
            g_total_loss.backward()
            # Gradient clipping
            g_grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()
            
            # Update EMA
            ema_generator.update(generator)
            
            # Monitor gradient norms every 50 batches
            if batch_idx % 50 == 0:
                d_grad_norm = sum(p.grad.norm().item() for p in discriminator.parameters() if p.grad is not None)
                print(f"  Grad norms - G: {g_grad_norm:.3f}, D: {d_grad_norm:.3f}")
        
        # Accumulate losses (only if not NaN)
        if not torch.isnan(g_total_loss):
            epoch_g_loss += g_total_loss.item()
        if not torch.isnan(d_total_loss):
            epoch_d_loss += d_total_loss.item()
        epoch_recon_loss += feature_match_loss.item()  # Track feature matching instead
        epoch_va_loss += g_va_loss.item()
        num_batches += 1
        
        # Print progress ONLY on first batch
        if batch_idx == 0:
            print(f"Epoch [{epoch}] {resolution}×{resolution} {phase_name} - "
                  f"G={g_total_loss.item():.4f}, D={d_total_loss.item():.4f}, "
                  f"FM={feature_match_loss.item():.4f}")
    
    # Save sample grid ONLY every 10 epochs (using EMA for better quality)
    if epoch % 10 == 0:
        save_sample_grid(ema_generator, latent_dim, device, epoch, resolution)
    
    # Step LR schedulers at end of epoch
    if scheduler_G is not None and epoch >= 200:  # Start decay after epoch 200
        scheduler_G.step()
    if scheduler_D is not None and epoch >= 200:
        scheduler_D.step()
    
    # Print epoch summary ONLY every 10 epochs
    if epoch % 10 == 0 and num_batches > 0:
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{resolution}×{resolution} Summary - {phase_name}")
        print(f"G_loss: {epoch_g_loss/num_batches:.4f}, D_loss: {epoch_d_loss/num_batches:.4f}")
        print(f"Feature Match: {epoch_recon_loss/num_batches:.4f}")
        print(f"{'='*60}\n")
    
    # Record loss history
    if loss_history is not None and num_batches > 0:
        loss_history['epochs'].append(epoch)
        loss_history['g_loss'].append(epoch_g_loss / num_batches)
        loss_history['d_loss'].append(epoch_d_loss / num_batches)
        loss_history['recon_loss'].append(epoch_recon_loss / num_batches)
        loss_history['resolution'].append(resolution)


def plot_loss_curves(loss_history, stage_idx, resolution):
    """Plot and save loss curves for the training history"""
    import matplotlib.pyplot as plt
    
    if len(loss_history['epochs']) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training Progress - Stage {stage_idx} ({resolution}×{resolution})', fontsize=16)
    
    epochs = loss_history['epochs']
    
    # Plot 1: Generator Loss
    axes[0, 0].plot(epochs, loss_history['g_loss'], 'b-', linewidth=2, label='G Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Generator Loss')
    axes[0, 0].set_title('Generator Loss Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Discriminator Loss
    axes[0, 1].plot(epochs, loss_history['d_loss'], 'r-', linewidth=2, label='D Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Discriminator Loss')
    axes[0, 1].set_title('Discriminator Loss Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Reconstruction Loss
    axes[1, 0].plot(epochs, loss_history['recon_loss'], 'g-', linewidth=2, label='Recon Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Reconstruction Loss')
    axes[1, 0].set_title('Reconstruction Loss Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: All Losses Combined
    axes[1, 1].plot(epochs, loss_history['g_loss'], 'b-', linewidth=2, label='G Loss', alpha=0.7)
    axes[1, 1].plot(epochs, loss_history['d_loss'], 'r-', linewidth=2, label='D Loss', alpha=0.7)
    axes[1, 1].plot(epochs, loss_history['recon_loss'], 'g-', linewidth=2, label='Recon Loss', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('All Losses Combined')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Add vertical lines to show resolution changes
    resolutions = loss_history['resolution']
    prev_res = resolutions[0] if resolutions else 4
    for i, res in enumerate(resolutions):
        if res != prev_res:
            for ax in axes.flatten():
                ax.axvline(x=epochs[i], color='gray', linestyle='--', alpha=0.5, linewidth=1)
            prev_res = res
    
    plt.tight_layout()
    plt.savefig(f"GAN/Samples_TrueProgressiveGAN/loss_curves_stage{stage_idx}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved loss curves to loss_curves_stage{stage_idx}.png")


def save_sample_grid(ema_generator, latent_dim, device, epoch, resolution):
    """Save 3×3 grid of generated images at different VA values (using EMA)"""
    import matplotlib.pyplot as plt
    from Models.generator import Generator
    
    # Create temporary model with EMA weights for better quality samples
    temp_gen = Generator(z_dim=latent_dim, emotion_dim=2).to(device)
    for name, param in temp_gen.named_parameters():
        if name in ema_generator.shadow:
            param.data = ema_generator.shadow[name].clone()
    temp_gen.eval()
    
    with torch.no_grad():
        save_dir = "GAN/Samples_TrueProgressiveGAN"
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle(f"Epoch {epoch} - {resolution}×{resolution} Progressive GAN", fontsize=16)
        
        # 3×3 grid of VA values
        va_grid = [(0.2, 0.2), (0.2, 0.5), (0.2, 0.8),
                   (0.5, 0.2), (0.5, 0.5), (0.5, 0.8),
                   (0.8, 0.2), (0.8, 0.5), (0.8, 0.8)]
        
        # Use DIFFERENT latent codes for each image to show diversity
        z_samples = torch.randn(9, latent_dim, device=device)
        
        for idx, (v_val, a_val) in enumerate(va_grid):
            z = z_samples[idx:idx+1]  # Use different z for each image
            v_t = torch.tensor([v_val], device=device)
            a_t = torch.tensor([a_val], device=device)
            
            fake_img = temp_gen(z, v_t, a_t, stage=get_stage_from_resolution(resolution), alpha=1.0, return_eru_features=False)
            
            # Upscale to 128×128 for visualization if needed
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_progressive_gan(
        data_root=r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Data\Landscape",
        va_file=r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Data\EmotionLabel\all_photos_valence_arousal.csv",
        vae_weights="GAN/Weights/vae_best.pth",
        batch_size=16,
        latent_dim=128,
        lr_g=0.0001,
        lr_d=0.000001,  # 100× slower
        n_critic=5,  # WGAN standard
        afm_lambda=100.0,  # Paper's λ
        device=device
    )
