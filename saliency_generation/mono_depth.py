"""Monocular depth estimation (MiDaS small) for arbitrary RGB images.

Removes the RGB-D dataset requirement: any photo, dashcam frame, or webcam
capture can be pushed through the full saliency pipeline. The model is
loaded lazily from torch.hub on first use (~66 MB download, cached).

Requires the optional `timm` package (pip install timm). `available()`
reports whether the feature can be used without raising.
"""

import os
import sys

import cv2
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common import get_device, get_logger

log = get_logger(__name__)

_model = None
_transform = None


def available() -> bool:
    """True if the optional timm dependency for MiDaS is installed."""
    try:
        import timm  # noqa: F401
        return True
    except ImportError:
        return False


def _load():
    global _model, _transform
    if _model is not None:
        return
    if not available():
        raise ImportError(
            "Monocular depth needs the `timm` package: pip install timm")
    device = get_device()
    log.info("Loading MiDaS_small from torch.hub (first run downloads ~66 MB)...")
    _model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    _model.to(device).eval()
    transforms_hub = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    _transform = transforms_hub.small_transform
    log.info("MiDaS_small ready on %s", device)


def estimate_depth(rgb: np.ndarray) -> np.ndarray:
    """Estimate depth for an RGB image (H, W, 3, uint8).

    Returns a float32 map in [0, 1] at the input resolution. MiDaS predicts
    inverse relative depth, so HIGHER values mean CLOSER — matching the
    convention of the NJUD/NLPR depth maps used to train cmSalGAN.
    """
    _load()
    device = get_device()
    inp = _transform(rgb).to(device)
    with torch.no_grad():
        pred = _model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=rgb.shape[:2],
            mode="bicubic", align_corners=False).squeeze()
    depth = pred.cpu().numpy()
    lo, hi = float(depth.min()), float(depth.max())
    if hi - lo < 1e-8:
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - lo) / (hi - lo)).astype(np.float32)


def depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    """[0, 1] float depth → uint8 grayscale image for saving/display."""
    return (np.clip(depth, 0, 1) * 255).astype(np.uint8)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """[0, 1] float depth → BGR colormap visualization."""
    return cv2.applyColorMap(depth_to_uint8(depth), cv2.COLORMAP_MAGMA)
