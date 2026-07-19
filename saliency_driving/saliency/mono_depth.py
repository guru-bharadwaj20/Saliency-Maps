"""Monocular depth estimation (MiDaS small) for arbitrary RGB images.

MiDaS predicts inverse relative depth, so higher values mean closer, matching
the NJUD/NLPR depth convention cmSalGAN was trained on. Requires the optional
`timm` package; use `available()` to check before calling.
"""

import cv2
import numpy as np
import torch

from saliency_driving.config import get_device
from saliency_driving.log import get_logger

log = get_logger(__name__)

_model = None
_transform = None


def available() -> bool:
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
        raise ImportError("Monocular depth needs the `timm` package: pip install timm")
    device = get_device()
    log.info("Loading MiDaS_small from torch.hub (first run downloads ~66 MB)...")
    _model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    _model.to(device).eval()
    _transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
    log.info("MiDaS_small ready on %s", device)


def estimate_depth(rgb: np.ndarray) -> np.ndarray:
    """Depth map in [0, 1] at the input resolution (higher = closer)."""
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
    return (np.clip(depth, 0, 1) * 255).astype(np.uint8)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(depth_to_uint8(depth), cv2.COLORMAP_MAGMA)
