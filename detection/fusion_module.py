import torch
import torch.nn as nn
import torch.nn.functional as F

class SaliencyFusion(nn.Module):
    """
    Fuses RGB image (3 channels) with saliency map (1 channel)
    into a 4-channel tensor for downstream detection models.
    Also supports lightweight feature refinement via attention.
    """

    def __init__(self, use_attention=True):
        super(SaliencyFusion, self).__init__()
        self.use_attention = use_attention

        # Small convolutional refinement layers
        self.refine = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Optional attention weighting
        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(4, 8, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 1, kernel_size=1),
                nn.Sigmoid()
            )

        # Final feature projection to 3 channels (compatible with YOLO/Faster-RCNN)
        self.project = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, rgb_img, saliency_map):
        """
        rgb_img: (B, 3, H, W)
        saliency_map: (B, 1, H, W)
        """
        # Normalize saliency to [0, 1]
        saliency_norm = torch.clamp(saliency_map, 0, 1)

        # Concatenate RGB + saliency → (B, 4, H, W)
        fused = torch.cat((rgb_img, saliency_norm), dim=1)

        # Optional attention weighting
        if self.use_attention:
            attn = self.attention(fused)
            fused = fused * (1 + attn)  # enhance salient regions

        # Feature refinement and projection
        refined = self.refine(fused)
        output = self.project(refined)

        return output  # (B, 3, H, W)
