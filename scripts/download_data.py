"""Download datasets and the cmSalGAN checkpoint from Google Drive (needs gdown).

    python scripts/download_data.py --what model | dataset | edge | all
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "model": {"id": "1j18BvmGEUip1NSlK3N4t66jU_WeV2tCF",
              "target": ROOT / "models" / "cmSalGAN.ckpt"},
    "dataset": {"id": "1YENRxUxAcFQhxcesxaWHEM3BipcIayX1",
                "target": ROOT / "downloads" / "train_dataset.zip"},
    "edge": {"id": "1J8z_LH2KvHYZEXApcwLgV8KoqBrxAJ5d",
             "target": ROOT / "downloads" / "edge_dataset.zip"},
}


def download(what: str):
    try:
        import gdown
    except ImportError:
        sys.exit("gdown is required: pip install gdown")

    target = FILES[what]["target"]
    if target.exists():
        print(f"[skip] {target} already exists")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {what} -> {target}")
    gdown.download(f"https://drive.google.com/uc?id={FILES[what]['id']}", str(target), quiet=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--what", choices=[*FILES, "all"], default="model")
    args = parser.parse_args()
    for key in (FILES if args.what == "all" else [args.what]):
        download(key)
