import torch
import torch.nn as nn
import torch.nn.functional as F
from .eru import EmotionalResidualUnit


class PixelNorm(nn.Module):
    """Pixel-wise feature normalization (paper requirement)"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x, epsilon=1e-8):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + epsilon)


class Generator(nn.Module):
    def __init__(self, z_dim=100, emotion_dim=2):
        super().__init__()
        self.progressive_blocks = nn.ModuleList()
        self.to_rgb = nn.ModuleList()

        # Initial block (4x4)
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(z_dim + emotion_dim, 512, 4, 1, 0),
            PixelNorm(),
            nn.ReLU(),
        )
        self.progressive_blocks.append(EmotionalResidualUnit(512, use_spectral_norm=False))
        self.to_rgb.append(nn.Conv2d(512, 3, 3, padding=1))

        # Progressively increase resolution with deeper residual stacks
        # Each stage now has: PixelShuffle upsampling → Conv → BN → ReLU, followed by two
        # residual-style conv blocks and an ERU for stronger capacity.
        channels = [512, 256, 128, 64, 32, 32]  # Stage 0-5 output channels (limit to 128x128)
        for i in range(1, 6):  # 8x8 → 128x128 (removed 256x256 for memory)
            in_ch = channels[i - 1]
            out_ch = channels[i]

            block_layers = [
                # PixelShuffle upsampling (ZERO artifacts - SOTA approach)
                nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1),  # Generate 4x channels
                nn.PixelShuffle(2),  # Rearrange to 2x spatial resolution
                PixelNorm(),
                nn.ReLU(),
            ]

            # First residual-style conv block
            block_layers.extend(
                [
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    PixelNorm(),
                    nn.ReLU(),
                ]
            )

            # Second residual-style conv block
            block_layers.extend(
                [
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    PixelNorm(),
                    nn.ReLU(),
                ]
            )
            
            # Third conv block for deeper feature learning (NEW - critical for 128×128 quality)
            block_layers.extend(
                [
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    PixelNorm(),
                    nn.ReLU(),
                    EmotionalResidualUnit(out_ch, use_spectral_norm=False),
                ]
            )

            self.progressive_blocks.append(nn.Sequential(*block_layers))
            self.to_rgb.append(nn.Conv2d(out_ch, 3, 3, padding=1))

    def forward(self, z, v, a, stage=5, alpha=1.0, return_eru_features=False):  # stage: 0(4x4) → 5(128x128)
        # Expand dimensions of v and a to match z
        v_scalar = v.unsqueeze(1) if v.dim() == 1 else v  # [B] -> [B, 1]
        a_scalar = a.unsqueeze(1) if a.dim() == 1 else a  # [B] -> [B, 1]
        v_flat = v if v.dim() == 1 else v.squeeze(1)  # ERU needs [B] format
        a_flat = a if a.dim() == 1 else a.squeeze(1)

        cond = torch.cat([z, v_scalar, a_scalar], dim=1).unsqueeze(-1).unsqueeze(-1)
        x = self.initial(cond)
        
        eru_features = []  # For AFM loss (not used currently, but available)

        # Progressive fade-in support (currently always alpha=1.0 for stability)
        for i in range(stage + 1):
            block = self.progressive_blocks[i]

            # Process all blocks uniformly
            if isinstance(block, EmotionalResidualUnit):
                # ERU returns (output, v_attention, a_attention)
                x, v_att, a_att = block(x, v_flat, a_flat)
                if return_eru_features:
                    eru_features.append((v_att, a_att))
            else:
                # Iterate through each layer in the Sequential block
                for layer in block:
                    if isinstance(layer, EmotionalResidualUnit):
                        x, v_att, a_att = layer(x, v_flat, a_flat)
                        if return_eru_features:
                            eru_features.append((v_att, a_att))
                    else:
                        x = layer(x)

        # Note: alpha fade-in not implemented for ERU generator (always stable at alpha=1.0)
        # Progressive training controls resolution via stage parameter
        return torch.tanh(self.to_rgb[stage](x))
