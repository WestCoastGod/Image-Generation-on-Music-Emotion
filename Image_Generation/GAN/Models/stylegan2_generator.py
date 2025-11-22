import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate from StyleGAN2."""
    def __init__(self, in_features, out_features, bias=True, lr_mul=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.lr_mul = lr_mul
        self.scale = (1 / math.sqrt(in_features)) * lr_mul

    def forward(self, x):
        weight = self.weight * self.scale
        bias = self.bias * self.lr_mul if self.bias is not None else None
        return F.linear(x, weight, bias)


class MappingNetwork(nn.Module):
    """Maps latent z to intermediate latent w."""
    def __init__(self, z_dim=512, w_dim=512, n_layers=8):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_dim = z_dim if i == 0 else w_dim
            layers.append(EqualizedLinear(in_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.mapping = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.mapping(z)


class VAConditioningModule(nn.Module):
    """Injects VA conditioning into w latent space with moderate modulation."""
    def __init__(self, w_dim=512, va_dim=2):
        super().__init__()
        self.va_embed = nn.Sequential(
            EqualizedLinear(va_dim, w_dim),
            nn.LeakyReLU(0.2),
            EqualizedLinear(w_dim, w_dim),
        )
        self.scale = nn.Parameter(torch.ones(1) * 1.0)
        
    def forward(self, w, v, a):
        va = torch.stack([v, a], dim=1)
        va_latent = self.va_embed(va)
        return w + self.scale * va_latent


class NoiseInjection(nn.Module):
    """Learnable noise injection."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        return x + self.weight * noise


class ModulatedConv2d(nn.Module):
    """Modulated convolution from StyleGAN2."""
    def __init__(self, in_channels, out_channels, kernel_size, w_dim, demodulate=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.style_weight = EqualizedLinear(w_dim, in_channels, bias=True)
        
    def forward(self, x, w):
        batch, in_c, height, width = x.shape
        
        style = self.style_weight(w).view(batch, 1, in_c, 1, 1)
        weight = self.weight.unsqueeze(0) * self.scale
        weight = weight * (style + 1)
        
        if self.demodulate:
            d = torch.rsqrt((weight ** 2).sum([2, 3, 4], keepdim=True) + 1e-8)
            weight = weight * d
            
        x = x.view(1, batch * in_c, height, width)
        weight = weight.view(batch * self.out_channels, in_c, self.kernel_size, self.kernel_size)
        
        x = F.conv2d(x, weight, padding=self.kernel_size // 2, groups=batch)
        x = x.view(batch, self.out_channels, height, width)
        
        return x


class StyleBlock(nn.Module):
    """StyleGAN2 synthesis block with modulated conv and noise."""
    def __init__(self, in_channels, out_channels, w_dim, upsample=True):
        super().__init__()
        self.upsample = upsample
        
        self.conv1 = ModulatedConv2d(in_channels, out_channels, 3, w_dim)
        self.noise1 = NoiseInjection()
        self.bias1 = nn.Parameter(torch.zeros(out_channels))
        self.activation1 = nn.LeakyReLU(0.2)
        
        self.conv2 = ModulatedConv2d(out_channels, out_channels, 3, w_dim)
        self.noise2 = NoiseInjection()
        self.bias2 = nn.Parameter(torch.zeros(out_channels))
        self.activation2 = nn.LeakyReLU(0.2)
        
        self.va_modulation = nn.Sequential(
            nn.Linear(2, out_channels * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(out_channels * 2, out_channels),
            nn.Tanh()
        )
        
    def forward(self, x, w, v, a):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            
        x = self.conv1(x, w)
        x = self.noise1(x)
        x = x + self.bias1.view(1, -1, 1, 1)
        
        va = torch.stack([v, a], dim=1)
        va_scale = self.va_modulation(va).view(x.shape[0], x.shape[1], 1, 1)
        x = x * (1.0 + 0.3 * va_scale)
        
        x = self.activation1(x)
        
        x = self.conv2(x, w)
        x = self.noise2(x)
        x = x + self.bias2.view(1, -1, 1, 1)
        x = self.activation2(x)
        
        return x


class StyleGAN2Generator(nn.Module):
    """Simplified StyleGAN2-like generator with VA conditioning."""
    def __init__(self, z_dim=100, w_dim=512, img_resolution=64, img_channels=3):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        
        self.mapping = MappingNetwork(z_dim, w_dim, n_layers=4)
        self.va_conditioning = VAConditioningModule(w_dim)
        
        self.const = nn.Parameter(torch.randn(1, 512, 4, 4))
        
        self.blocks = nn.ModuleList([
            StyleBlock(512, 512, w_dim, upsample=False),
            StyleBlock(512, 384, w_dim, upsample=True),
            StyleBlock(384, 256, w_dim, upsample=True),
            StyleBlock(256, 128, w_dim, upsample=True),
            StyleBlock(128, 64, w_dim, upsample=True),
        ])
        
        self.to_rgb = ModulatedConv2d(64, img_channels, 1, w_dim, demodulate=False)
        
    def forward(self, z, v, a, stage=None):
        batch = z.shape[0]
        
        w = self.mapping(z)
        w = self.va_conditioning(w, v, a)
        
        x = self.const.repeat(batch, 1, 1, 1)
        
        num_blocks = len(self.blocks) if stage is None else min(stage + 1, len(self.blocks))
        for i in range(num_blocks):
            x = self.blocks[i](x, w, v, a)
            
        x = self.to_rgb(x, w)
        x = torch.tanh(x)
        
        return x
