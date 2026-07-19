"""Attention-weighted fusion of RGB + saliency into a 3-channel detector input."""

import torch
import torch.nn as nn


class SaliencyFusion(nn.Module):
    def __init__(self, use_attention=True):
        super().__init__()
        self.use_attention = use_attention

        self.refine = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(4, 8, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 1, kernel_size=1),
                nn.Sigmoid(),
            )
        self.project = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, rgb_img, saliency_map):
        fused = torch.cat((rgb_img, torch.clamp(saliency_map, 0, 1)), dim=1)
        if self.use_attention:
            fused = fused * (1 + self.attention(fused))
        return self.project(self.refine(fused))
