import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class EmotionalResidualUnit(nn.Module):
    """Paper's ERU: Spatial VA concatenation + Soft Attention
    
    Architecture from paper:
    1. Create spatial VA maps (H×W×1) filled with v/a values
    2. Concat with input: [V,X] and [A,X]
    3. Conv + Sigmoid to get attention maps v and a
    4. Element-wise multiply and add: m = (v⊗X) ⊕ (a⊗X)
    5. Concat m with [V,A] and final conv + tanh
    """
    def __init__(self, in_channels, use_spectral_norm=False, stage=3):
        super().__init__()
        self.in_channels = in_channels
        self.stage = stage
        
        # Helper for spectral norm
        def maybe_sn(layer):
            return spectral_norm(layer) if use_spectral_norm else layer
        
        # Step 1: V concatenation branch - Conv([V, X]) → sigmoid
        # Input: (in_channels + 1), Output: in_channels
        self.conv_v = maybe_sn(nn.Conv2d(in_channels + 1, in_channels, 3, padding=1))
        
        # Step 2: A concatenation branch - Conv([A, X]) → sigmoid
        self.conv_a = maybe_sn(nn.Conv2d(in_channels + 1, in_channels, 3, padding=1))
        
        # Step 3: Final combination - Conv([m, V, A]) → tanh
        # Input: (in_channels + 2), Output: in_channels
        self.conv_final = maybe_sn(nn.Conv2d(in_channels + 2, in_channels, 3, padding=1))
        
        # Initialize weights
        for module in [self.conv_v, self.conv_a, self.conv_final]:
            if hasattr(module, 'weight'):
                nn.init.xavier_normal_(module.weight, gain=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, x, v, a):
        """Forward pass following paper's exact architecture
        
        Args:
            x: Input features [B, C, H, W]
            v: Valence values [B] (normalized to [0,1])
            a: Arousal values [B] (normalized to [0,1])
        
        Returns:
            y: Output features [B, C, H, W] with emotion modulation
            v_map, a_map: Attention maps (for AFM loss)
        """
        B, C, H, W = x.shape
        
        # Create spatial emotion maps filled with constant VA values
        # Paper: "width and height are equal to input X"
        V_map = v.view(B, 1, 1, 1).expand(B, 1, H, W)  # [B, 1, H, W]
        A_map = a.view(B, 1, 1, 1).expand(B, 1, H, W)  # [B, 1, H, W]
        
        # Step 1: Valence attention - Eq. (1) in paper
        # v = sigmoid(Conv([V, X]))
        concat_v = torch.cat([V_map, x], dim=1)  # [B, C+1, H, W]
        v_attention = torch.sigmoid(self.conv_v(concat_v))  # [B, C, H, W]
        
        # Note: Paper mentions "sum = 1.0 like soft attention experimentally"
        # but this is too restrictive for VAE reconstruction - removed
        
        # Step 2: Arousal attention - Eq. (1) in paper
        # a = sigmoid(Conv([A, X]))
        concat_a = torch.cat([A_map, x], dim=1)  # [B, C+1, H, W]
        a_attention = torch.sigmoid(self.conv_a(concat_a))  # [B, C, H, W]
        
        # Step 3: Combine attention with input - Conservative scaling to prevent saturation
        # m = X + 0.2 * (v ⊗ X + a ⊗ X) - prevents over-amplification at higher resolutions
        # Total amplification: 1.0x + 0.2x = 1.2x max (safe from saturation cascade)
        m = x + 0.2 * ((v_attention * x) + (a_attention * x))  # [B, C, H, W]
        
        # Step 4: Final output - Eq. (3) in paper
        # Y = tanh(Conv([m, V, A]))
        concat_final = torch.cat([m, V_map, A_map], dim=1)  # [B, C+2, H, W]
        y = torch.tanh(self.conv_final(concat_final))  # [B, C, H, W]
        
        # Return output and attention maps (for AFM loss computation)
        return y, v_attention, a_attention
