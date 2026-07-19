"""Spectral-residual saliency (Hou & Zhang, 2007) — a checkpoint-free fallback."""

import cv2
import numpy as np

WORK_SIZE = 64


def spectral_residual_saliency(rgb: np.ndarray) -> np.ndarray:
    """Saliency map in [0, 1] at the input resolution for an RGB or gray image."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    h, w = gray.shape

    small = cv2.resize(gray, (WORK_SIZE, WORK_SIZE)).astype(np.float32)
    fft = np.fft.fft2(small)
    log_amplitude = np.log(np.abs(fft) + 1e-8)
    phase = np.angle(fft)
    residual = log_amplitude - cv2.blur(log_amplitude, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (9, 9), 2.5)

    lo, hi = float(sal.min()), float(sal.max())
    if hi - lo < 1e-12:
        return np.zeros((h, w), dtype=np.float32)
    sal = (sal - lo) / (hi - lo)
    return cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
