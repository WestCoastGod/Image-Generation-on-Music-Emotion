import os
import torch
import torch.nn as nn
from Models.stylegan2_generator import StyleGAN2Generator
from Models.discriminator import Discriminator
from Utils.dataloader import EmotionDataset
from Utils.losses import WGAN_GPLoss, ResNetPerceptualLoss, FeatureMatchingLoss
from Utils.ema import EMA
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F


def train_stylegan(
    img_dir,
    label_csv,
    epochs=50,
    batch_size=16,
    lr_g=0.0001,
    lr_d=0.0001,
    lambda_gp=10,
    n_critic=1,
    device=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    G = StyleGAN2Generator(z_dim=100, w_dim=512, img_resolution=64).to(device)
    D = Discriminator().to(device)
    
    G_ema = EMA(G, decay=0.999)

    opt_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.0, 0.99))
    opt_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.0, 0.99))
    
    for param in G.parameters():
        if len(param.shape) >= 2:
            nn.init.kaiming_normal_(param, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

    wgan_gp = WGAN_GPLoss(lambda_gp=lambda_gp)
    perceptual_loss = ResNetPerceptualLoss().to(device)
    feature_matching_loss = FeatureMatchingLoss()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    dataset = EmotionDataset(img_dir, label_csv, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=4, pin_memory=True, persistent_workers=True)

    best_loss_G = float("inf")
    stage = 4

    for epoch in range(1, epochs + 1):
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        num_batches = 0
        
        for i, (real_imgs, v, a) in enumerate(dataloader):
            real_imgs, v, a = real_imgs.to(device), v.to(device), a.to(device)
            z = torch.randn(real_imgs.size(0), 100).to(device)

            for _ in range(n_critic):
                D.train()
                G.eval()

                with torch.no_grad():
                    fake_imgs = G(z, v, a, stage=stage)

                D_real, real_features, va_pred_real = D(real_imgs, v, a, stage=stage)
                D_fake, fake_features, _ = D(fake_imgs.detach(), v, a, stage=stage)

                real_features = [(rf.detach(), v.detach(), a.detach()) for (rf, v, a) in real_features]
                fake_features = [(ff.detach(), v.detach(), a.detach()) for (ff, v, a) in fake_features]

                loss_adv = -torch.mean(D_real) + torch.mean(D_fake)
                loss_gp = wgan_gp(D, real_imgs, fake_imgs, v, a, stage=stage)
                
                va_target = torch.stack([v, a], dim=1)
                loss_va_d = F.mse_loss(va_pred_real, va_target)
                
                loss_D = loss_adv + loss_gp + 0.5 * loss_va_d

                opt_D.zero_grad()
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
                opt_D.step()

            D.eval()
            G.train()

            fake_imgs = G(z, v, a, stage=stage)
            D_fake, fake_features, va_pred_fake = D(fake_imgs, v, a, stage=stage)

            loss_G_adv = -torch.mean(D_fake)
            
            va_target = torch.stack([v, a], dim=1)
            loss_G_va = F.mse_loss(va_pred_fake, va_target) * 1.0
            
            loss_G_perceptual = perceptual_loss(real_imgs, fake_imgs)
            loss_G_fm = feature_matching_loss(real_features, fake_features)
            
            loss_G = (
                loss_G_adv
                + loss_G_va
                + 1.0 * loss_G_perceptual
                + 0.2 * loss_G_fm
            )

            opt_G.zero_grad()
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            opt_G.step()
            
            G_ema.update(G)
            
            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            num_batches += 1

            if loss_G.item() < best_loss_G:
                best_loss_G = loss_G.item()
                torch.save(G.state_dict(), r"GAN\Weights\stylegan_generator_best.pth")

            if i % 50 == 0:
                print(f"Epoch [{epoch}/{epochs}], Batch {i}: D_loss={loss_D.item():.3f}, G_loss={loss_G.item():.3f}")
        
        avg_loss_G = epoch_loss_G / num_batches
        avg_loss_D = epoch_loss_D / num_batches
        
        if epoch % 10 == 0:
            print(f"\n=== Epoch {epoch} Summary ===")
            print(f"Best G Loss: {best_loss_G:.3f}")
            print(f"Avg G Loss: {avg_loss_G:.3f}, Avg D Loss: {avg_loss_D:.3f}")
        
        if epoch % 5 == 0:
            import matplotlib.pyplot as plt
            
            save_dir = r"GAN\Samples_StyleGAN"
            os.makedirs(save_dir, exist_ok=True)
            
            G.eval()
            G_ema.apply_shadow(G)
            
            with torch.no_grad():
                fig, axes = plt.subplots(3, 3, figsize=(12, 12))
                valences = [2.0, 5.0, 8.0]
                arousals = [2.0, 5.0, 8.0]
                
                z_fixed = torch.randn(1, 100).to(device)
                
                for i, v in enumerate(valences):
                    for j, a in enumerate(arousals):
                        v_tensor = torch.tensor([v / 9.0]).float().to(device)
                        a_tensor = torch.tensor([a / 9.0]).float().to(device)
                        
                        img = G(z_fixed, v_tensor, a_tensor, stage=stage)
                        img = img.squeeze().permute(1, 2, 0).cpu().numpy()
                        img = (img * 0.5 + 0.5).clip(0, 1)
                        
                        axes[i, j].imshow(img)
                        axes[i, j].set_title(f"V={v:.1f}, A={a:.1f}", fontsize=10)
                        axes[i, j].axis('off')
                
                plt.suptitle(f"Epoch {epoch} - StyleGAN2 + VA Conditioning [EMA]", fontsize=16)
                plt.tight_layout()
                plt.savefig(f"{save_dir}/epoch_{epoch:04d}.png")
                plt.close()
                print(f"Sample images saved to {save_dir}/epoch_{epoch:04d}.png")
            
            G_ema.restore(G)
            G.train()

    torch.save(G.state_dict(), r"GAN\Weights\stylegan_generator_final.pth")
    
    G_ema.apply_shadow(G)
    torch.save(G.state_dict(), r"GAN\Weights\stylegan_generator_final_ema.pth")
    G_ema.restore(G)
    
    torch.save(D.state_dict(), r"GAN\Weights\stylegan_discriminator_final.pth")
    
    print("\nStyleGAN training complete.")


if __name__ == "__main__":
    img_dir = r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Data\variate_images_for_test"
    label_csv = r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\GAN\Data\EmotionLabel\all_photos_valence_arousal.csv"

    print("Starting StyleGAN2 training with VA conditioning...")
    train_stylegan(img_dir, label_csv, epochs=50)
