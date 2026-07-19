"""RGB-D dataset for saliency generation and detection."""

import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from saliency_driving.config import load_config

INPUT_SIZE = load_config()["saliency"]["input_size"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def normalize_imagenet(rgb_tensor):
    """ImageNet-normalize a [0, 1] tensor of shape (B, 3, H, W) or (3, H, W)."""
    return _normalize(rgb_tensor)


class CustomDataset(Dataset):
    """Returns (rgb, depth, name, orig_size); rgb/depth are (3, S, S) in [0, 1]."""

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
