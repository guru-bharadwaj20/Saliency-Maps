import numpy as np
import torch

from saliency_driving.config import (
    DEFAULTS,
    ROOT,
    _merge,
    dataset_name,
    get_device,
    load_config,
    resolve_path,
    set_seed,
)


def test_load_config_has_all_sections():
    cfg = load_config()
    for section in ("paths", "saliency", "training", "detection", "simulation"):
        assert section in cfg


def test_merge_deep_override():
    merged = _merge(DEFAULTS, {"training": {"lr": 0.5}})
    assert merged["training"]["lr"] == 0.5
    assert merged["training"]["epochs"] == DEFAULTS["training"]["epochs"]


def test_dataset_env_override(monkeypatch):
    monkeypatch.setenv("DATASETS", "NLPR,STEREO")
    assert dataset_name() == "NLPR"
    monkeypatch.delenv("DATASETS")
    assert dataset_name() == load_config()["dataset"]


def test_resolve_path_is_absolute():
    p = resolve_path("data")
    assert p.is_absolute()
    assert p == ROOT / "data"


def test_get_device_valid():
    assert get_device() in ("cpu", "cuda")


def test_set_seed_reproducible():
    set_seed(123)
    a = (torch.rand(3), np.random.rand(3))
    set_seed(123)
    b = (torch.rand(3), np.random.rand(3))
    assert torch.allclose(a[0], b[0])
    assert np.allclose(a[1], b[1])
