import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalBatchNorm2d(nn.Module):
    """Conditional Batch Normalization (CBN) / AdaIN
    Modulates features based on emotion (v, a)
    """
    def __init__(self, num_features, num_conditions=2):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Linear(num_conditions, num_features * 2)
        # Initialize scale to 1, shift to 0
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()
        self.embed.bias.data.zero_()

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class VAEEncoder(nn.Module):
    """Simple encoder without ERU - for VAE warm-start only."""
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Progressive encoding: 128x128 -> 4x4 (NO ERUs for clean reconstruction)
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 128x128 -> 64x64
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.2),
        )
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1),  # 64x64 -> 32x32
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.2),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.2),
        )
        
        # Output: 512 x 4 x 4 = 8192
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        
    def forward(self, x, v=None, a=None):
        # v, a ignored during VAE training (kept for compatibility)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        # Constrain logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-10, max=2)
        
        return mu, logvar


class ResidualBlock(nn.Module):
    """Residual block with conditional batch norm"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.cbn1 = ConditionalBatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.cbn2 = ConditionalBatchNorm2d(channels)
        
    def forward(self, x, y):
        residual = x
        out = self.conv1(x)
        out = self.cbn1(out, y)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.cbn2(out, y)
        return F.relu(out + residual)


class VAEDecoder(nn.Module):
    """Decoder WITH Conditional Batch Norm (AdaIN) - SIMPLE VERSION
    Light architecture matching VAE that already produces good results.
    NO ResidualBlocks - they cause discriminator dominance.
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.ReLU()
        )
        
        # Stage 0: 4x4 (Initial) - Simple conv with CBN
        self.conv_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.cbn_4 = ConditionalBatchNorm2d(512)
        
        # Stage 1: 4x4 -> 8x8
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(512, 256, 3, padding=1)
        self.cbn1 = ConditionalBatchNorm2d(256)
        
        # Stage 2: 8x8 -> 16x16
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.cbn2 = ConditionalBatchNorm2d(128)
        
        # Stage 3: 16x16 -> 32x32
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.cbn3 = ConditionalBatchNorm2d(64)
        
        # Stage 4: 32x32 -> 64x64
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2d(64, 32, 3, padding=1)
        self.cbn4 = ConditionalBatchNorm2d(32)
        
        # Stage 5: 64x64 -> 128x128
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        self.cbn5 = ConditionalBatchNorm2d(32)
        
        # Final RGB conversion (no modulation needed here)
        self.to_rgb = nn.Sequential(
            nn.Conv2d(32, 3, 1),  # 1x1 conv for RGB
            nn.Tanh()
        )

        # Progressive RGB heads (for lower resolutions)
        self.to_rgb_4 = nn.Sequential(nn.Conv2d(512, 3, 1), nn.Tanh())
        self.to_rgb_8 = nn.Sequential(nn.Conv2d(256, 3, 1), nn.Tanh())
        self.to_rgb_16 = nn.Sequential(nn.Conv2d(128, 3, 1), nn.Tanh())
        self.to_rgb_32 = nn.Sequential(nn.Conv2d(64, 3, 1), nn.Tanh())
        self.to_rgb_64 = nn.Sequential(nn.Conv2d(32, 3, 1), nn.Tanh())
        
    def forward(self, z, v, a, stage=5, alpha=1.0, return_eru_features=False):
        # Combine v, a into a single condition vector
        # v, a are [B], make them [B, 2]
        if v.dim() == 1:
            y = torch.stack([v, a], dim=1)
        else:
            y = torch.cat([v, a], dim=1)
            
        # Stage 0: 4x4
        x = self.fc(z)
        x = x.view(x.size(0), 512, 4, 4)
        x = F.relu(self.cbn_4(self.conv_4(x), y))
        
        # If we are at stage 0 (4x4), return here
        if stage == 0:
            rgb = self.to_rgb_4(x)
            if return_eru_features: return rgb, {}
            return rgb

        # Stage 1: 8x8
        x_prev = x # Save for fade-in
        x = self.up1(x)
        x = F.relu(self.cbn1(self.conv1(x), y))
            
        if stage == 1:
            rgb = self.to_rgb_8(x)
            if alpha < 1.0:
                # Fade in: (1-alpha) * upsample(prev_rgb) + alpha * curr_rgb
                rgb_prev = self.to_rgb_4(x_prev)
                rgb_prev = F.interpolate(rgb_prev, scale_factor=2, mode='nearest')
                rgb = (1 - alpha) * rgb_prev + alpha * rgb
            if return_eru_features: return rgb, {}
            return rgb
        
        # Stage 2: 16x16
        x_prev = x
        x = self.up2(x)
        x = F.relu(self.cbn2(self.conv2(x), y))
            
        if stage == 2:
            rgb = self.to_rgb_16(x)
            if alpha < 1.0:
                rgb_prev = self.to_rgb_8(x_prev)
                rgb_prev = F.interpolate(rgb_prev, scale_factor=2, mode='nearest')
                rgb = (1 - alpha) * rgb_prev + alpha * rgb
            if return_eru_features: return rgb, {}
            return rgb
        
        # Stage 3: 32x32
        x_prev = x
        x = self.up3(x)
        x = F.relu(self.cbn3(self.conv3(x), y))
            
        if stage == 3:
            rgb = self.to_rgb_32(x)
            if alpha < 1.0:
                rgb_prev = self.to_rgb_16(x_prev)
                rgb_prev = F.interpolate(rgb_prev, scale_factor=2, mode='nearest')
                rgb = (1 - alpha) * rgb_prev + alpha * rgb
            if return_eru_features: return rgb, {}
            return rgb
        
        # Stage 4: 64x64
        x_prev = x
        x = self.up4(x)
        x = F.relu(self.cbn4(self.conv4(x), y))
            
        if stage == 4:
            rgb = self.to_rgb_64(x)
            if alpha < 1.0:
                rgb_prev = self.to_rgb_32(x_prev)
                rgb_prev = F.interpolate(rgb_prev, scale_factor=2, mode='nearest')
                rgb = (1 - alpha) * rgb_prev + alpha * rgb
            if return_eru_features: return rgb, {}
            return rgb
        
        # Stage 5: 128x128
        x_prev = x
        x = self.up5(x)
        x = F.relu(self.cbn5(self.conv5(x), y))
        
        rgb = self.to_rgb(x)  # Convert to RGB
        if alpha < 1.0:
            rgb_prev = self.to_rgb_64(x_prev)
            rgb_prev = F.interpolate(rgb_prev, scale_factor=2, mode='nearest')
            rgb = (1 - alpha) * rgb_prev + alpha * rgb
        
        if return_eru_features: return rgb, {}
        return rgb


class VAEGAN(nn.Module):
    """VA-Conditioned Variational Autoencoder."""
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = VAEDecoder(latent_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, v, a):
        mu, logvar = self.encoder(x, v, a)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, v, a, stage=5, alpha=1.0)  # Full 128x128 resolution
        return recon, mu, logvar
    
    def generate(self, z, v, a):
        """Generate image from latent code and VA values."""
        return self.decoder(z, v, a, stage=5, alpha=1.0)
