import pytest
import torch

from datasets.saliency_dataset import INPUT_SIZE, CustomDataset, normalize_imagenet


def test_shapes_and_range(tiny_split):
    ds = CustomDataset(str(tiny_split))
    assert len(ds) == 2
    rgb, depth, name, orig_size = ds[0]
    assert rgb.shape == (3, INPUT_SIZE, INPUT_SIZE)
    assert depth.shape == (3, INPUT_SIZE, INPUT_SIZE)
    assert 0.0 <= rgb.min() and rgb.max() <= 1.0


def test_original_size_preserved(tiny_split):
    ds = CustomDataset(str(tiny_split))
    _, _, _, orig_size = ds[0]
    assert orig_size.tolist() == [60, 40]  # (w, h), not the resized 224


def test_normalize_imagenet(tiny_split):
    ds = CustomDataset(str(tiny_split))
    rgb, *_ = ds[0]
    norm = normalize_imagenet(rgb)
    assert norm.shape == rgb.shape
    assert norm.min() < 0  # normalization must shift the range


def test_missing_depth_fallback(tiny_split):
    (tiny_split / "depth" / "img_a.jpg").unlink()
    ds = CustomDataset(str(tiny_split))
    rgb, depth, name, _ = ds[0]
    assert name == "img_a.jpg"
    assert torch.all(depth == 0)  # placeholder depth


def test_missing_rgb_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CustomDataset(str(tmp_path))
