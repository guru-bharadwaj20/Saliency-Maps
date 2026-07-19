"""Assemble the simulation frames into an animated GIF for the README.

Usage:
    python scripts/make_demo_gif.py [--dataset NJUD] [--out docs/demo.gif]

Run `python main.py --simulate` first so the frames exist.
"""

import argparse
import glob
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import get_logger, load_config, resolve_path

log = get_logger(__name__)


def make_gif(dataset="NJUD", out_path=None, frame_ms=400, width=480):
    cfg = load_config()
    frames_dir = resolve_path("simulation_outputs", cfg) / dataset
    frame_files = sorted(glob.glob(str(frames_dir / f"{dataset}_frame_*.png")))
    if not frame_files:
        raise RuntimeError(
            f"No simulation frames in {frames_dir}. Run `python main.py --simulate` first.")

    out_path = Path(out_path) if out_path else ROOT / "docs" / "demo.gif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for f in frame_files:
        img = Image.open(f).convert("RGB")
        h = int(img.height * width / img.width)
        frames.append(img.resize((width, h), Image.LANCZOS))

    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=frame_ms, loop=0, optimize=True)
    size_kb = out_path.stat().st_size // 1024
    log.info("GIF written: %s (%d frames, %d KB)", out_path, len(frames), size_kb)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    make_gif(dataset=args.dataset or load_config()["dataset"], out_path=args.out)
