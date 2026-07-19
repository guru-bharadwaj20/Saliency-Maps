"""Classical saliency fallback: spectral residual (Hou & Zhang, CVPR 2007).

Used for arbitrary images when the cmSalGAN checkpoint is not available, so
the single-image pipeline and web demo always produce a real (if simpler)
saliency signal. Every output produced this way is labeled as heuristic —
never presented as cmSalGAN output.
"""

import cv2
import numpy as np

WORK_SIZE = 64  # spectral residual operates on a small image by design


def spectral_residual_saliency(rgb: np.ndarray) -> np.ndarray:
    """Saliency map for an RGB or grayscale image. Returns float32 [0, 1] at
    the input resolution."""
    if rgb.ndim == 3:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb
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
