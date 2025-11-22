import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import lpips


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for texture learning"""
    def __init__(self):
        super().__init__()
        # Load pretrained VGG16 (only for loss calculation, not for generation)
        vgg = models.vgg16(pretrained=True).features
        # Use conv3_3 features (layer 15)
        self.feature_extractor = nn.Sequential(*list(vgg[:16])).eval()
        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, real_imgs, fake_imgs):
        # Extract features
        real_features = self.feature_extractor(real_imgs)
        fake_features = self.feature_extractor(fake_imgs)
        # MSE loss on features
        return F.mse_loss(fake_features, real_features)


class LPIPSPerceptualLoss(nn.Module):
    """LPIPS perceptual loss - industry standard for image quality.
    Strongly penalizes blur and better correlates with human perception.
    """
    def __init__(self):
        super().__init__()
        # Use AlexNet backbone (fastest, good performance)
        self.lpips_fn = lpips.LPIPS(net='alex').eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad = False
    
    def forward(self, fake_imgs, real_imgs):
        # LPIPS expects [-1, 1] range (same as our Tanh output)
        return self.lpips_fn(fake_imgs, real_imgs).mean()


class ResNetPerceptualLoss(nn.Module):
    """Perceptual loss using ImageNet-pretrained ResNet18 mid-level features."""

    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Use stem + layer1 + layer2 for mid-level structure
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        ).eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        # ImageNet normalization stats
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, real_imgs, fake_imgs):
        # Inputs are in [-1, 1]; map to [0, 1]
        real = (real_imgs + 1.0) / 2.0
        fake = (fake_imgs + 1.0) / 2.0
        # Normalize for ResNet
        real = (real - self.mean) / self.std
        fake = (fake - self.mean) / self.std
        real_feat = self.feature_extractor(real)
        fake_feat = self.feature_extractor(fake)
        return F.l1_loss(fake_feat, real_feat)


class TextureLoss(nn.Module):
    """Penalizes smooth outputs by rewarding high spatial gradients."""
    def __init__(self):
        super().__init__()
        
    def forward(self, fake_imgs):
        dx = torch.abs(fake_imgs[:, :, :, 1:] - fake_imgs[:, :, :, :-1])
        dy = torch.abs(fake_imgs[:, :, 1:, :] - fake_imgs[:, :, :-1, :])
        
        gradient_mag = dx.mean() + dy.mean()
        
        return -gradient_mag


class FeatureMatchingLoss(nn.Module):
    """Match discriminator intermediate features for diversity"""
    def __init__(self):
        super().__init__()
    
    def forward(self, real_features, fake_features):
        """real/fake_features: List of (feature_map, v, a) from discriminator"""
        loss = 0
        for (real_f, _, _), (fake_f, _, _) in zip(real_features, fake_features):
            # Match feature statistics
            real_mean = torch.mean(real_f, dim=[2, 3])
            fake_mean = torch.mean(fake_f, dim=[2, 3])
            loss += F.l1_loss(fake_mean, real_mean)
        return loss / len(real_features)


class AFMLoss(nn.Module):
    """
    Affective Feature Matching Loss (Paper Eq. 4 & 5)
    Computes L1 distance between internal attention maps (v, a) of ERUs.
    """
    def __init__(self):
        super().__init__()

    def forward(self, real_attentions, fake_attentions):
        """
        Args:
            real_attentions: Dict {eru_name: (v_attn, a_attn)} from discriminator on real images
            fake_attentions: Dict {eru_name: (v_attn, a_attn)} from discriminator on fake images
        """
        loss_v = 0.0
        loss_a = 0.0
        count = 0

        # Iterate over common ERUs (e.g., eru1, eru2...)
        for key in real_attentions.keys():
            if key in fake_attentions:
                real_v, real_a = real_attentions[key]
                fake_v, fake_a = fake_attentions[key]
                
                # Eq. 4: L1 distance of Valence attention maps
                loss_v += F.l1_loss(fake_v, real_v)
                
                # Eq. 5: L1 distance of Arousal attention maps
                loss_a += F.l1_loss(fake_a, real_a)
                
                count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=list(real_attentions.values())[0][0].device)

        # Average over all ERUs
        return (loss_v + loss_a) / count


class WGAN_GPLoss(nn.Module):
    def __init__(self, lambda_gp=10):
        super().__init__()
        self.lambda_gp = lambda_gp

    def gradient_penalty(self, D, real_imgs, fake_imgs, v, a, stage=6):
        """WGAN-GP gradient penalty"""
        alpha = torch.rand(real_imgs.size(0), 1, 1, 1).to(real_imgs.device)
        interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(
            True
        )
        d_interpolates, _, _ = D(interpolates, v, a, stage=stage)  # Unpack 3 values: validity, features, va_pred

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp

    def forward(self, D, real_imgs, fake_imgs, v, a, stage=6):
        gp = self.gradient_penalty(D, real_imgs, fake_imgs, v, a, stage=stage)
        return gp
