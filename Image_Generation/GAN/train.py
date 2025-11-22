import os
import torch
from Models.generator import Generator
from Models.discriminator import Discriminator
from Utils.dataloader import EmotionDataset
from Utils.losses import AFMLoss, WGAN_GPLoss, FeatureMatchingLoss, ResNetPerceptualLoss, TextureLoss
from Utils.ema import EMA
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F


def train(
    img_dir,
    label_csv,
    epochs=100,  # Total epochs
    batch_size=16,
    lr_g=0.0002,
    lr_d=0.00005,
    lambda_gp=5,
    n_critic=1,
    mode="complex",  # "complex" (current setup) or "baseline" (simple 64x64)
    use_afm=True,     # Whether to include AFM loss in G loss
    device=None,
):
    # Initialize device
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Initialize models
    G = Generator().to(device)
    D = Discriminator().to(device)
    
    # Initialize EMA for generator (decay=0.999 means 99.9% old, 0.1% new)
    G_ema = EMA(G, decay=0.999)

    # Optimizers with separate learning rates
    opt_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    
    # Learning rate schedulers - reduce LR when training plateaus
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(opt_G, mode='min', factor=0.5, patience=200, verbose=True)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(opt_D, mode='min', factor=0.5, patience=200, verbose=True)

    # Loss functions
    wgan_gp = WGAN_GPLoss(lambda_gp=lambda_gp)
    perceptual_loss = ResNetPerceptualLoss().to(device)
    feature_matching_loss = FeatureMatchingLoss()
    afm_loss_fn = AFMLoss().to(device)
    texture_loss = TextureLoss().to(device)
    
    # Tracking metrics for plotting
    history = {
        'epoch': [],
        'D_real': [],  # Discriminator score on real images
        'D_fake': [],  # Discriminator score on fake images
        'loss_D': [],  # Total discriminator loss
        'loss_G': [],  # Total generator loss
        'va_loss': []  # VA auxiliary loss
    }

    # Data loading without augmentation (color affects emotion)
    # For baseline mode we still load at 128x128 then downsample once; this
    # keeps the dataset pipeline consistent with complex mode.
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    dataset = EmotionDataset(img_dir, label_csv, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    best_loss_G = float("inf")  # Initialize the best generator loss

    # Training loop (1..epochs for cleaner logging and sampling)
    for epoch in range(1, epochs + 1):
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        num_batches = 0
        
        for i, (real_imgs, v, a) in enumerate(dataloader):
            real_imgs, v, a = real_imgs.to(device), v.to(device), a.to(device)
            z = torch.randn(real_imgs.size(0), 100).to(device)

            if mode == "baseline":
                # Fixed 64x64 training, no progressive growing or LR tricks
                stage = 4  # Generator stage for 64x64 (4*2**4)
                stage_epoch = 0
                current_n_critic = n_critic
                lr_scale = 1.0
                resolution = 64
                # Downsample real images once to 64x64
                real_imgs = F.interpolate(
                    real_imgs,
                    size=(resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                # Complex progressive mode (existing behavior)
                stage = min(epoch // 10, 5)
                stage_epoch = epoch % 10

                if stage_epoch < 5 and stage > 0:
                    current_n_critic = 10
                    lr_scale = 0.2
                else:
                    current_n_critic = n_critic
                    lr_scale = 1.0

                for param_group in opt_G.param_groups:
                    param_group['lr'] = lr_g * lr_scale
                for param_group in opt_D.param_groups:
                    param_group['lr'] = lr_d * lr_scale

                resolution = 4 * (2**stage)
                real_imgs = F.interpolate(
                    real_imgs,
                    size=(resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                )

            # --- Train Discriminator (multiple times per generator update) ---
            for _ in range(current_n_critic):
                D.train()
                G.eval()

                with torch.no_grad():
                    fake_imgs = G(z, v, a, stage=stage)

                D_real, real_features, va_pred_real = D(real_imgs, v, a, stage=stage)
                D_fake, fake_features, va_pred_fake = D(fake_imgs.detach(), v, a, stage=stage)

                # Optional: patch-level critic to encourage local structure
                patch_size = 32 if resolution >= 32 else resolution
                _, _, H, W = real_imgs.shape
                if H >= patch_size and W >= patch_size:
                    top = torch.randint(0, H - patch_size + 1, (1,), device=device).item()
                    left = torch.randint(0, W - patch_size + 1, (1,), device=device).item()

                    real_patches = real_imgs[:, :, top:top+patch_size, left:left+patch_size]
                    fake_patches = fake_imgs[:, :, top:top+patch_size, left:left+patch_size]

                    D_real_p, _, _ = D(real_patches, v, a, stage=stage)
                    D_fake_p, _, _ = D(fake_patches, v, a, stage=stage)
                else:
                    D_real_p, D_fake_p = None, None

                real_features = [
                    (rf.detach(), v.detach(), a.detach()) for (rf, v, a) in real_features
                ]
                fake_features = [
                    (ff.detach(), v.detach(), a.detach()) for (ff, v, a) in fake_features
                ]

                loss_adv = -torch.mean(D_real) + torch.mean(D_fake)
                if D_real_p is not None and D_fake_p is not None:
                    loss_adv_patch = -torch.mean(D_real_p) + torch.mean(D_fake_p)
                    loss_adv = loss_adv + 0.5 * loss_adv_patch
                loss_gp = wgan_gp(D, real_imgs, fake_imgs, v, a, stage=stage)
                
                # VA Auxiliary Loss: Discriminator predicts VA from real images only
                va_target = torch.stack([v, a], dim=1)  # [B, 2]
                loss_va = F.mse_loss(va_pred_real, va_target)
                
                loss_D = loss_adv + loss_gp + 0.5 * loss_va

                opt_D.zero_grad()
                loss_D.backward()
                # Aggressive gradient clipping during stage transitions
                grad_clip = 0.5 if (stage_epoch < 5 and stage > 0) else 1.0
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=grad_clip)
                opt_D.step()

            # --- Train Generator (once per n_critic discriminator updates) ---
            D.eval()
            G.train()

            fake_imgs = G(z, v, a, stage=stage)
            D_fake, fake_features, va_pred_fake = D(fake_imgs, v, a, stage=stage)

            # Patch-level critic for generator as well
            patch_size = 32 if resolution >= 32 else resolution
            _, _, H, W = fake_imgs.shape
            if H >= patch_size and W >= patch_size:
                top = torch.randint(0, H - patch_size + 1, (1,), device=device).item()
                left = torch.randint(0, W - patch_size + 1, (1,), device=device).item()

                fake_patches = fake_imgs[:, :, top:top+patch_size, left:left+patch_size]
                D_fake_p, _, _ = D(fake_patches, v, a, stage=stage)
            else:
                D_fake_p = None

            # Adversarial loss
            loss_G_adv = -torch.mean(D_fake)
            if D_fake_p is not None:
                loss_G_adv_patch = -torch.mean(D_fake_p)
                loss_G_adv = loss_G_adv + 0.5 * loss_G_adv_patch
            
            # VA Auxiliary Loss: Generator must produce images where VA is predictable
            va_target = torch.stack([v, a], dim=1)  # [B, 2]
            loss_G_va = F.mse_loss(va_pred_fake, va_target) * 2.0  # 2x weight to strongly encourage VA encoding
            
            # Perceptual and feature-matching losses
            # ResNet18-based perceptual loss on real vs fake images
            loss_G_perceptual = perceptual_loss(real_imgs, fake_imgs)

            loss_G_fm = feature_matching_loss(real_features, fake_features)
            
            loss_G_texture = texture_loss(fake_imgs)

            # AFM loss (emotion-aware feature matching on discriminator features)
            loss_G_afm = afm_loss_fn(real_features, fake_features) if (use_afm and mode != "baseline") else torch.tensor(0.0).to(device)

            if mode == "baseline":
                # Baseline: adversarial + VA + ResNet-perceptual + FM + TEXTURE
                loss_G = (
                    loss_G_adv
                    + 0.5 * loss_G_va
                    + 0.3 * loss_G_perceptual
                    + 0.1 * loss_G_fm
                    + 2.0 * loss_G_texture
                )
            else:
                # Original complex generator loss with texture/gradient regularizers
                fake_std_per_channel = torch.std(fake_imgs, dim=[2, 3])
                real_std_per_channel = torch.std(real_imgs, dim=[2, 3])
                loss_texture_diversity = F.l1_loss(fake_std_per_channel, real_std_per_channel)

                fake_grad_x = torch.abs(fake_imgs[:, :, :, :-1] - fake_imgs[:, :, :, 1:])
                fake_grad_y = torch.abs(fake_imgs[:, :, :-1, :] - fake_imgs[:, :, 1:, :])
                real_grad_x = torch.abs(real_imgs[:, :, :, :-1] - real_imgs[:, :, :, 1:])
                real_grad_y = torch.abs(real_imgs[:, :, :-1, :] - real_imgs[:, :, 1:, :])
                loss_gradient = F.l1_loss(fake_grad_x.mean(), real_grad_x.mean()) + F.l1_loss(real_grad_y.mean(), fake_grad_y.mean())

                perceptual_weight = {0: 0.0, 1: 0.0, 2: 1.0, 3: 3.0, 4: 7.0, 5: 10.0}[stage]
                texture_weight = 10.0 if stage >= 3 else 5.0

                loss_G = (
                    loss_G_adv
                    + perceptual_weight * loss_G_perceptual
                    + 0.1 * loss_G_fm
                    + loss_G_va
                    + texture_weight * (loss_texture_diversity + loss_gradient)
                    + 0.5 * loss_G_afm
                )
            
            # Debug: Print discriminator outputs occasionally
            if i == 0 and epoch % 5 == 0:
                print(f"  DEBUG ({mode}): D_real={torch.mean(D_real).item():.2f}, D_fake={torch.mean(D_fake).item():.2f}, n_crit={current_n_critic}, lr={opt_G.param_groups[0]['lr']:.5f}")
                print(f"  VA_PRED: pred_v={va_pred_fake[0,0].item():.2f}, true_v={v[0].item():.2f}, pred_a={va_pred_fake[0,1].item():.2f}, true_a={a[0].item():.2f}")
                print(f"  LOSS: Adv={loss_G_adv.item():.2f}, VA={loss_G_va.item():.3f}, Perc={loss_G_perceptual.item():.3f}, FM={loss_G_fm.item():.3f}, Texture={loss_G_texture.item():.3f}, AFM={loss_G_afm.item():.3f}")
                if mode == "complex" and resolution >= 16:
                    print(f"  (complex extras active at res {resolution}x{resolution})")
                # Print ERU scale values to check if they're learning
                for name, param in G.named_parameters():
                    if 'scale_v' in name or 'scale_a' in name:
                        print(f"  ERU: {name}={param.data.item():.3f}, grad={'YES' if param.grad is not None else 'NO'}")
                # Print sample VA values being used
                print(f"  VA: v_sample={v[0].item():.2f}, a_sample={a[0].item():.2f}")

            opt_G.zero_grad()
            loss_G.backward()
            # Aggressive gradient clipping during stage transitions
            grad_clip = 0.5 if (mode == "complex" and stage_epoch < 5 and stage > 0) else 1.0
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=grad_clip)
            opt_G.step()
            
            # Update EMA after each generator step
            G_ema.update(G)
            
            # Accumulate losses for epoch average
            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            num_batches += 1

            # Save the best generator model
            if loss_G.item() < best_loss_G:
                best_loss_G = loss_G.item()
                torch.save(
                    G.state_dict(),
                    r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Weights\generator_best.pth",
                )

            # Logging
            if i % 50 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}], Batch {i}: D_loss={loss_D.item():.3f}, G_loss={loss_G.item():.3f}, Stage={stage}, LR_G={opt_G.param_groups[0]['lr']:.6f}"
                )
        
        # Calculate average losses for the epoch
        avg_loss_G = epoch_loss_G / num_batches
        avg_loss_D = epoch_loss_D / num_batches
        
        # Track metrics for plotting
        with torch.no_grad():
            # Sample a batch to get D scores
            sample_real = real_imgs[:4]
            sample_z = torch.randn(4, 100).to(device)
            sample_v = v[:4]
            sample_a = a[:4]
            sample_fake = G(sample_z, sample_v, sample_a, stage=stage)
            d_real_score, _, _ = D(sample_real, sample_v, sample_a, stage=stage)
            d_fake_score, _, _ = D(sample_fake, sample_v, sample_a, stage=stage)
            
            history['epoch'].append(epoch)
            history['D_real'].append(d_real_score.mean().item())
            history['D_fake'].append(d_fake_score.mean().item())
            history['loss_D'].append(avg_loss_D)
            history['loss_G'].append(avg_loss_G)
            history['va_loss'].append(loss_G_va.item() if 'loss_G_va' in locals() else 0.0)
        
        # Update learning rate schedulers after each epoch
        scheduler_G.step(avg_loss_G)
        scheduler_D.step(avg_loss_D)
        
        # Epoch summary and sample generation
        if epoch % 10 == 0:
            print(f"\n=== Epoch {epoch} Summary ===")
            print(f"Best G Loss so far: {best_loss_G:.3f}")
            if mode == "baseline":
                print("Mode: baseline (fixed 64x64, stage 4)")
                print("Current Stage: 4 (Resolution: 64x64)")
            else:
                print(f"Mode: complex (progressive). Current Stage: {stage} (Resolution: {4 * (2**stage)}x{4 * (2**stage)})")
            print(f"Learning Rate: G={opt_G.param_groups[0]['lr']:.6f}, D={opt_D.param_groups[0]['lr']:.6f}")
            # Check ERU scale evolution
            print(f"ERU Scales: ", end="")
            for name, param in G.named_parameters():
                if 'scale' in name:
                    print(f"{name.split('.')[-1]}={param.data.item():.3f} ", end="")
            print("\n")
        
        # Generate sample images at regular intervals (every 10 epochs)
        should_generate = (epoch % 10 == 0)
        if should_generate:
            import os
            import matplotlib.pyplot as plt
            
            save_dir = r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Samples"
            os.makedirs(save_dir, exist_ok=True)
            
            # Use EMA generator for sample generation (better quality)
            G.eval()
            G_ema.apply_shadow(G)
            
            with torch.no_grad():
                # Generate 9 images with different VA values
                fig, axes = plt.subplots(3, 3, figsize=(12, 12))
                valences = [2.0, 5.0, 8.0]
                arousals = [2.0, 5.0, 8.0]
                
                # CRITICAL: Use SAME z for all VA combinations to isolate VA effect
                z_fixed = torch.randn(1, 100).to(device)
                
                for i, v in enumerate(valences):
                    for j, a in enumerate(arousals):
                        v_tensor = torch.tensor([v / 9.0]).float().to(device)
                        a_tensor = torch.tensor([a / 9.0]).float().to(device)
                        
                        img = G(z_fixed, v_tensor, a_tensor, stage=stage)
                        img = img.squeeze().permute(1, 2, 0).cpu().numpy()
                        img = (img * 0.5 + 0.5).clip(0, 1)
                        
                        axes[i, j].imshow(img)
                        axes[i, j].set_title(f"V={v:.1f}, A={a:.1f}")
                        axes[i, j].axis('off')
                
                if mode == "baseline":
                    title_res = "64x64"
                else:
                    title_res = f"{4 * (2**stage)}x{4 * (2**stage)}"
                plt.suptitle(f"Epoch {epoch} - Mode {mode} - Stage {stage} ({title_res}) [EMA] - SAME Z", fontsize=16)
                plt.tight_layout()
                plt.savefig(f"{save_dir}/epoch_{epoch:04d}.png")
                plt.close()
                print(f"Sample images saved to {save_dir}/epoch_{epoch:04d}.png")
                print(f"  (Using SAME noise z for all VA combinations to isolate VA effect)")
            
            # Restore original weights for continued training
            G_ema.restore(G)
            G.train()

    # Save final models (both regular and EMA versions)
    torch.save(
        G.state_dict(),
        r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Weights\generator_final.pth",
    )
    
    # Save EMA generator (usually produces better images)
    G_ema.apply_shadow(G)
    torch.save(
        G.state_dict(),
        r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Weights\generator_final_ema.pth",
    )
    G_ema.restore(G)
    
    torch.save(
        D.state_dict(),
        r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Weights\discriminator_final.pth",
    )
    
    # Plot training curves
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Discriminator Scores (D_real vs D_fake)
    axes[0, 0].plot(history['epoch'], history['D_real'], label='D(real)', color='blue')
    axes[0, 0].plot(history['epoch'], history['D_fake'], label='D(fake)', color='red')
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Discriminator Score')
    axes[0, 0].set_title('Discriminator Scores Over Training')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Discriminator Loss
    axes[0, 1].plot(history['epoch'], history['loss_D'], label='D Loss', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Discriminator Loss Over Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Generator Loss
    axes[1, 0].plot(history['epoch'], history['loss_G'], label='G Loss', color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Generator Loss Over Training')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: VA Auxiliary Loss
    axes[1, 1].plot(history['epoch'], history['va_loss'], label='VA Loss', color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('VA Auxiliary Loss Over Training')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(r'C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\training_curves.png', dpi=600)
    print("\nTraining curves saved to training_curves.png")
    plt.close()


if __name__ == "__main__":
    # Use the balanced VA subset for quicker experimentation
    img_dir = r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Data\Landscape"
    
    # Master CSV with all 10,766 VA values - can use with any image subset
    label_csv = r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Data\EmotionLabel\all_photos_valence_arousal.csv"

    # Train the model
    # Default to baseline mode so you can quickly compare it to the
    # existing complex configuration by changing the "mode" argument.
    mode = "baseline"  # change to "complex" to use progressive training
    use_afm = True

    print(f"Starting training with landscape images in {mode} mode...")
    if mode == "baseline":
        print("Baseline config: fixed 64x64, simple loss (adv + VA + perceptual + FM + AFM)")
    else:
        print("Complex config: progressive stages, perceptual (0→1→3→7→10), texture+gradient diversity, AFM")

    train(img_dir, label_csv, mode=mode, use_afm=use_afm)
    print("Training complete.")
