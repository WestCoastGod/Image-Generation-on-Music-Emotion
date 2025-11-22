import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.vae import VAEGAN
from Utils.dataloader import EmotionDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def vae_loss(recon_x, x, mu, logvar, kld_weight=0.001, free_bits=0.5):
    """VAE loss with free bits: reconstruction + constrained KL divergence."""
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # Per-dimension KLD
    kld_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    # Free bits: only penalize KLD that exceeds free_bits per dimension
    kld_loss = torch.mean(torch.max(kld_per_dim - free_bits, torch.zeros_like(kld_per_dim)))
    
    return recon_loss + kld_weight * kld_loss, recon_loss, kld_loss


def train_vae(
    img_dir,
    label_csv,
    epochs=100,
    batch_size=16,
    lr=0.0003,
    latent_dim=128,
    kld_weight=0.00001,
    device=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = VAEGAN(latent_dim=latent_dim).to(device)
    
    # DON'T reinitialize! ConditionalBatchNorm has careful init (gamma=1, beta=0)
    # Xavier with gain=0.1 destroys this → gamma≈0.1 → all features vanish → gray blur
    
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Upgraded to 128x128
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    dataset = EmotionDataset(img_dir, label_csv, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)

    best_loss = float("inf")
    warmup_epochs = 30
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kld = 0.0
        num_batches = 0
        
        current_kld_weight = min(kld_weight * (epoch / warmup_epochs), kld_weight) if epoch <= warmup_epochs else kld_weight
        
        for i, (real_imgs, v, a) in enumerate(dataloader):
            real_imgs, v, a = real_imgs.to(device), v.to(device), a.to(device)
            
            # Use REAL VA values from dataset
            # VAE learns: "Given (z, v, a), reconstruct the image"
            # ConditionalBatchNorm stays flexible
            # GAN later refines the V/A conditioning
            recon_imgs, mu, logvar = model(real_imgs, v, a)
            
            loss, recon_loss, kld_loss = vae_loss(recon_imgs, real_imgs, mu, logvar, current_kld_weight, free_bits=0.0)  # No free bits - force meaningful latent
            
            # Check for NaN before VA predictor
            if torch.isnan(loss) or torch.isnan(recon_loss) or torch.isnan(kld_loss):
                print(f"NaN in VAE loss at epoch {epoch}, batch {i}")
                print(f"  mu stats: min={mu.min().item():.3f}, max={mu.max().item():.3f}, mean={mu.mean().item():.3f}")
                print(f"  logvar stats: min={logvar.min().item():.3f}, max={logvar.max().item():.3f}, mean={logvar.mean().item():.3f}")
                print(f"  recon_loss={recon_loss.item():.4f}, kld_loss={kld_loss.item():.4f}")
                raise ValueError("Training diverged with NaN in VAE loss")
            
            total_loss = loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
            optimizer.step()
            
            if torch.isnan(total_loss):
                print(f"NaN detected at epoch {epoch}, batch {i}")
                print(f"  mu: min={mu.min().item():.3f}, max={mu.max().item():.3f}")
                print(f"  logvar: min={logvar.min().item():.3f}, max={logvar.max().item():.3f}")
                print(f"  recon: min={recon_imgs.min().item():.3f}, max={recon_imgs.max().item():.3f}")
                raise ValueError("Training diverged with NaN")
            
            epoch_loss += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_kld += kld_loss.item()
            num_batches += 1

            if i % 20 == 0:
                print(f"Epoch [{epoch}/{epochs}], Batch {i}: Loss={total_loss.item():.4f}, Recon={recon_loss.item():.4f}, KLD={kld_loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches
        avg_recon = epoch_recon / num_batches
        avg_kld = epoch_kld / num_batches
        
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), r"GAN\Weights\vae_best.pth")
        
        if epoch % 10 == 0:
            print(f"\n=== Epoch {epoch} Summary ===")
            print(f"Avg Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KLD: {avg_kld:.6f})")
            print(f"Best Loss: {best_loss:.4f}")
            print(f"KLD Weight: {current_kld_weight:.6f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        if epoch % 5 == 0:
            save_dir = r"GAN\Samples_VAE"
            os.makedirs(save_dir, exist_ok=True)
            
            model.eval()
            
            with torch.no_grad():
                fig, axes = plt.subplots(3, 3, figsize=(12, 12))
                valences = [2.0, 5.0, 8.0]
                arousals = [2.0, 5.0, 8.0]
                
                z_fixed = torch.randn(1, latent_dim).to(device)
                
                for i, v in enumerate(valences):
                    for j, a in enumerate(arousals):
                        # Use varying VA values to see if VAE learned any conditioning
                        v_tensor = torch.tensor([v / 9.0], dtype=torch.float32, device=device)
                        a_tensor = torch.tensor([a / 9.0], dtype=torch.float32, device=device)
                        
                        # Generate using decoder directly at full 128x128 resolution
                        img = model.decoder(z_fixed, v_tensor, a_tensor, stage=5, alpha=1.0)
                        img = img.squeeze().permute(1, 2, 0).cpu().numpy()
                        img = (img * 0.5 + 0.5).clip(0, 1)
                        
                        axes[i, j].imshow(img)
                        axes[i, j].set_title(f"V={v:.1f}, A={a:.1f}", fontsize=10)
                        axes[i, j].axis('off')
                
                plt.suptitle(f"Epoch {epoch} - VA-Conditioned VAE", fontsize=16)
                plt.tight_layout()
                plt.savefig(f"{save_dir}/epoch_{epoch:04d}.png")
                plt.close()
                print(f"Sample images saved to {save_dir}/epoch_{epoch:04d}.png\n")
            
            model.train()

    torch.save(model.state_dict(), r"GAN\Weights\vae_final.pth")
    
    print("\nVAE training complete!")


if __name__ == "__main__":
    img_dir = r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Data\Landscape"
    label_csv = r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Data\EmotionLabel\all_photos_valence_arousal.csv"

    print("Starting VA-Conditioned VAE training...")
    print("VAE learns to reconstruct images, so it naturally preserves textures and details!")
    
    train_vae(img_dir, label_csv, epochs=300, batch_size=8, lr=0.0003, latent_dim=128, kld_weight=0.0001)  # Smaller batch + higher KLD for features
