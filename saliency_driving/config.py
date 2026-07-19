"""Configuration loading, seeding, and path/device helpers."""

import copy
import os
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"

DEFAULTS: dict = {
    "dataset": "NJUD",
    "device": "auto",
    "seed": 42,
    "paths": {
        "data": "data",
        "results": "results",
        "saliency_maps": "results/saliency_maps",
        "detection_outputs": "results/detection_outputs",
        "simulation_outputs": "results/simulation_outputs",
        "tensorboard": "results/tensorboard",
        "cmsalgan_ckpt": "models/cmSalGAN.ckpt",
        "detector_best": "models/detector_best.pt",
        "checkpoints": "models/checkpoints",
    },
    "saliency": {"input_size": 224, "batch_size": 1},
    "training": {
        "batch_size": 2,
        "epochs": 3,
        "lr": 1.0e-4,
        "num_workers": 0,
        "fusion_mode": "attention",
        "saliency_source": "precomputed",
    },
    "detection": {
        "score_threshold": 0.5,
        "iou_threshold": 0.5,
        "min_box_area_fraction": 0.001,
    },
    "simulation": {
        "window": [640, 480],
        "step_size": 5.0,
        "steps_per_frame": 15,
        "goal_radius": 12,
        "smoothing": 0.8,
        "goal_weight": 1.0,
        "saliency_weight": 2.2,
        "detection_weight": 3.5,
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


@lru_cache(maxsize=1)
def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        return _merge(DEFAULTS, user_cfg)
    return copy.deepcopy(DEFAULTS)


def dataset_name(cfg: dict = None) -> str:
    """Active dataset; the DATASETS env var overrides config.yaml."""
    cfg = cfg or load_config()
    return os.environ.get("DATASETS", cfg["dataset"]).split(",")[0]


def resolve_path(key: str, cfg: dict = None) -> Path:
    """Absolute path for a paths.* config entry."""
    cfg = cfg or load_config()
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def get_device(cfg: dict = None) -> str:
    cfg = cfg or load_config()
    device = cfg.get("device", "auto")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def set_seed(seed: int = None) -> int:
    """Seed python, numpy, and torch RNGs; returns the seed used."""
    if seed is None:
        seed = load_config().get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
