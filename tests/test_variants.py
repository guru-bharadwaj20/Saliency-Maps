import pytest
import torch

from saliency_driving.train import FusedDetectionDataset


def _dataset(tiny_split, source):
    return FusedDetectionDataset(root_dir=str(tiny_split.parent), split=tiny_split.name,
                                 saliency_dir=None, saliency_source=source)


def test_gt_saliency_source(tiny_split):
    ds = _dataset(tiny_split, "gt")
    _, sal, target, _ = ds[0]
    assert sal.shape[0] == 1
    assert float(sal.max()) > 0.5
    assert len(target["boxes"]) > 0


def test_zero_saliency_source(tiny_split):
    ds = _dataset(tiny_split, "zero")
    _, sal, _, _ = ds[0]
    assert torch.all(sal == 0)


def test_precomputed_missing_falls_back_to_zero(tiny_split):
    ds = _dataset(tiny_split, "precomputed")
    _, sal, _, _ = ds[0]
    assert torch.all(sal == 0)


def test_invalid_fusion_mode_rejected():
    from saliency_driving.train import train
    with pytest.raises(ValueError, match="fusion_mode"):
        train(fusion_mode="bogus")
