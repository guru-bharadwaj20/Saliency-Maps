import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def square_gt():
    """100x100 float GT mask with a 40x40 foreground square."""
    gt = np.zeros((100, 100), np.float32)
    gt[30:70, 30:70] = 1.0
    return gt


@pytest.fixture
def tiny_split(tmp_path):
    """Minimal dataset split dir: RGB/, depth/, GT/ with two 60x40 images."""
    rgb_dir = tmp_path / "RGB"
    depth_dir = tmp_path / "depth"
    gt_dir = tmp_path / "GT"
    for d in (rgb_dir, depth_dir, gt_dir):
        d.mkdir()
    rng = np.random.RandomState(0)
    for name in ("img_a", "img_b"):
        Image.fromarray(rng.randint(0, 255, (40, 60, 3), np.uint8)).save(rgb_dir / f"{name}.jpg")
        Image.fromarray(rng.randint(0, 255, (40, 60), np.uint8)).save(depth_dir / f"{name}.jpg")
        mask = np.zeros((40, 60), np.uint8)
        mask[10:30, 20:50] = 255
        Image.fromarray(mask).save(gt_dir / f"{name}.png")
    return tmp_path
