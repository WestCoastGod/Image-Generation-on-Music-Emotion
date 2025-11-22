import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from .eru import EmotionalResidualUnit


class MinibatchStdDev(nn.Module):
    """Minibatch standard deviation layer for improved diversity"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        batch_size, _, height, width = x.shape
        # Calculate std across batch
        std = torch.std(x, dim=0, keepdim=True)
        # Average std across channels and spatial dimensions
        mean_std = std.mean().expand(batch_size, 1, height, width)
        # Add epsilon to avoid numerical issues
        mean_std = mean_std + 1e-8
        # Concatenate as new channel
        return torch.cat([x, mean_std], dim=1)


class Discriminator(nn.Module):
    """Discriminator with ERUs (paper architecture)"""
    def __init__(self):
        super().__init__()
        self.progressive_blocks = nn.ModuleList()
        self.from_rgb = nn.ModuleList()
        self.minibatch_std = MinibatchStdDev()
        # Balanced capacity: 2× generator at each stage (prevents D from crushing G)
        channels = [64, 64, 128, 256, 512, 512]  # [128×128, 64×64, 32×32, 16×16, 8×8, 4×4]

        # 1. Initialize the from_rgb layers (With Spectral Norm for stability)
        for ch in channels:
            conv = spectral_norm(nn.Conv2d(3, ch, 1))
            nn.init.normal_(conv.weight, 0, 0.02)
            nn.init.zeros_(conv.bias)
            self.from_rgb.append(conv)

        # 2. Initialize progressive blocks (With Spectral Norm + ERUs)
        for i in range(len(channels)):
            in_ch = channels[i]
            out_ch = channels[min(i + 1, len(channels) - 1)]
            
            block = []
            
            # ERU for emotion-aware features (paper architecture)
            block.append(EmotionalResidualUnit(in_ch, use_spectral_norm=True))
            
            # First conv block (in_ch -> in_ch) - ERU outputs in_ch
            conv1 = spectral_norm(nn.Conv2d(in_ch, in_ch, 3, padding=1))
            nn.init.normal_(conv1.weight, 0, 0.02)
            nn.init.zeros_(conv1.bias)
            block.extend([conv1, nn.LeakyReLU(0.2)])
            
            # Second conv block (in_ch -> out_ch) - transition to next stage channels
            conv2 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1))
            nn.init.normal_(conv2.weight, 0, 0.02)
            nn.init.zeros_(conv2.bias)
            block.extend([conv2, nn.LeakyReLU(0.2)])
            
            # Downsample at the END of the block (except the last 4x4 block)
            if i < len(channels) - 1:
                block.append(nn.AvgPool2d(2))
                
            self.progressive_blocks.append(nn.ModuleList(block))

        # 3. Dynamically adapt to input size (minibatch std adds 1 channel)
        # NOTE: NO SIGMOID for WGAN - we need raw critic scores, not probabilities!
        final_conv = spectral_norm(nn.Conv2d(513, 1, 1))
        nn.init.normal_(final_conv.weight, 0, 0.02)
        nn.init.zeros_(final_conv.bias)
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Key modification: Pool any size to 1x1
            final_conv,  # 512 + 1 from minibatch std
        )
        
        # VA Auxiliary Classifier: Predict VA values from image features
        # Add dropout to prevent learning from subtle cues - forces strong visual VA encoding
        va_head_conv = spectral_norm(nn.Conv2d(513, 2, 1))  # Output: [valence, arousal]
        nn.init.normal_(va_head_conv.weight, 0, 0.02)
        nn.init.zeros_(va_head_conv.bias)
        self.va_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(0.3),  # Force reliance on strong visual features
            va_head_conv,
            nn.Sigmoid()  # Output in [0, 1] range like input VA
        )

    def forward(self, x, v, a, stage=5, alpha=1.0, return_eru_features=False):  # Limit to stage 5 (128x128)
        # Collect ERU attention maps for AFM loss
        v_attentions = []
        a_attentions = []
        features = []
        stage = min(stage, len(self.progressive_blocks) - 1)
        
        # Format v, a for ERU
        v_flat = v if v.dim() == 1 else v.squeeze(1)
        a_flat = a if a.dim() == 1 else a.squeeze(1)

        # Handle Fade-in
        # If alpha < 1.0 and we are not at the lowest resolution (stage 5), we need to blend
        # The "current" stage is 'stage'. The "previous" (lower res) stage is 'stage + 1'.
        if alpha < 1.0 and stage < len(self.progressive_blocks) - 1:
            # Path A: New resolution (Current stage)
            # x is at current resolution (e.g., 8x8)
            x_new = self.from_rgb[stage](x)
            # Collect from_rgb output
            if return_eru_features:
                features.append(x_new)
            # Pass through the current block (which usually includes downsampling at the END)
            x_new_out = x_new
            for layer in self.progressive_blocks[stage]:
                if isinstance(layer, EmotionalResidualUnit):
                    x_new_out, v_att, a_att = layer(x_new_out, v_flat, a_flat)
                    if return_eru_features:
                        v_attentions.append(v_att)
                        a_attentions.append(a_att)
                elif isinstance(layer, nn.AvgPool2d) and x_new_out.size(2) <= 2:
                    continue
                else:
                    x_new_out = layer(x_new_out)
                    # Collect features after each conv layer (not just after blocks)
                    if return_eru_features and isinstance(layer, (nn.Conv2d, nn.Linear)):
                        features.append(x_new_out)
            
            # Path B: Old resolution (Previous stage)
            # Downsample input image to match previous resolution
            x_down = F.avg_pool2d(x, kernel_size=2)
            # Use the from_rgb of the previous stage
            x_old = self.from_rgb[stage + 1](x_down)
            
            # Blend
            x = (1 - alpha) * x_old + alpha * x_new_out
            
            # Continue from the NEXT stage
            start_block_idx = stage + 1
        else:
            # Standard path
            x = self.from_rgb[stage](x)
            # Collect from_rgb output
            if return_eru_features:
                features.append(x)
            start_block_idx = stage

        # 2. Process blocks from the current stage to the highest resolution
        for i in range(start_block_idx, len(self.progressive_blocks)):
            block = self.progressive_blocks[i]
            for layer in block:
                if isinstance(layer, EmotionalResidualUnit):
                    # Collect ERU attention maps for AFM loss
                    x, v_att, a_att = layer(x, v_flat, a_flat)
                    if return_eru_features:
                        v_attentions.append(v_att)
                        a_attentions.append(a_att)
                elif isinstance(layer, nn.AvgPool2d) and x.size(2) <= 2:
                    continue
                else:
                    x = layer(x)
                    if return_eru_features and isinstance(layer, (nn.Conv2d, nn.Linear)):
                        features.append(x)
        
        # 3. Add minibatch standard deviation before final layer
        x = self.minibatch_std(x)

        # 4. Final classification
        out = self.final(x)
        
        # Return: (logits, ERU_attentions, features)
        # ERU_attentions = (v_attentions, a_attentions) for AFM loss
        eru_attentions = (v_attentions, a_attentions) if return_eru_features else ([], [])
        return out, eru_attentions, features
