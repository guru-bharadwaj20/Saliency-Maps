import numpy as np

from saliency_driving.saliency import mono_depth
from saliency_driving.saliency.heuristic import spectral_residual_saliency


def test_spectral_residual_shape_and_range():
    rng = np.random.RandomState(0)
    img = rng.randint(0, 255, (120, 160, 3), np.uint8)
    sal = spectral_residual_saliency(img)
    assert sal.shape == (120, 160)
    assert sal.dtype == np.float32
    assert 0.0 <= sal.min() and sal.max() <= 1.0


def test_spectral_residual_highlights_structure():
    img = np.zeros((128, 128), np.uint8)
    img[50:80, 50:80] = 255
    sal = spectral_residual_saliency(img)
    assert sal[55:75, 55:75].mean() > sal[:30, :30].mean()


def test_spectral_residual_flat_image():
    sal = spectral_residual_saliency(np.full((64, 64), 128, np.uint8))
    assert np.isfinite(sal).all()


def test_mono_depth_available_is_bool():
    assert isinstance(mono_depth.available(), bool)


def test_depth_to_uint8():
    d = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
    u = mono_depth.depth_to_uint8(d)
    assert u.dtype == np.uint8 and u.min() == 0 and u.max() == 255
