"""Unified RGB-D test/inference dataset.

Single source of truth for loading RGB + depth pairs (replaces the two
diverging CustomDataset copies that used to live here and in
joint_dataset.py).

Each item returns:
    rgb       - (3, 224, 224) float tensor in [0, 1]  (un-normalized)
    depth     - (3, 224, 224) float tensor in [0, 1]
    name      - image file name (e.g. "000008_left.jpg")
    orig_size - LongTensor [width, height] of the original RGB image

The cmSalGAN checkpoint was trained on ImageNet-normalized RGB, so callers
feeding cmSalGAN should pass the rgb tensor through `normalize_imagenet`.
The detector / fusion path consumes the raw [0, 1] tensor directly
(torchvision detection models normalize internally).
"""

import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from common.config import load_config

INPUT_SIZE = load_config()["saliency"]["input_size"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def normalize_imagenet(rgb_tensor):
    """ImageNet-normalize a (B, 3, H, W) or (3, H, W) tensor in [0, 1]."""
    return _normalize(rgb_tensor)


class CustomDataset(Dataset):
    def __init__(self, root):
        self.rgb_dir = os.path.join(root, "RGB")
        self.depth_dir = os.path.join(root, "depth")
        if not os.path.isdir(self.rgb_dir):
            raise FileNotFoundError(f"RGB directory not found: {self.rgb_dir}")
        self.images = sorted(os.listdir(self.rgb_dir))
        self.transform = T.Compose([
            T.Resize((INPUT_SIZE, INPUT_SIZE)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        rgb_path = os.path.join(self.rgb_dir, name)
        depth_path = os.path.join(self.depth_dir, name)

        rgb_img = Image.open(rgb_path).convert("RGB")
        orig_w, orig_h = rgb_img.size

        if os.path.exists(depth_path):
            depth_img = Image.open(depth_path).convert("L")
        else:
            depth_img = Image.new("L", rgb_img.size)

        rgb = self.transform(rgb_img)
        depth = self.transform(depth_img).repeat(3, 1, 1)

        return rgb, depth, name, torch.tensor([orig_w, orig_h], dtype=torch.long)
