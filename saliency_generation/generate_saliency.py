import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common import get_device, get_logger, load_config, resolve_path, set_seed
from datasets.saliency_dataset import CustomDataset, normalize_imagenet
from models.cmSalGAN_model import UnetGAN

log = get_logger(__name__)


def generate_saliency_for_dataset(model, dataset_name, split="test", cfg=None):
    cfg = cfg or load_config()
    device = get_device(cfg)
    dataset_path = resolve_path("data", cfg) / dataset_name / split
    save_path = resolve_path("saliency_maps", cfg) / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)
    dataloader = DataLoader(CustomDataset(str(dataset_path)),
                            batch_size=cfg["saliency"]["batch_size"], shuffle=False)
    log.info("Generating saliency maps for %s/%s (%d images)",
             dataset_name, split, len(dataloader))
    for batch in tqdm(dataloader, total=len(dataloader)):
        img_rgb, img_dep, img_name, orig_size = batch
        out_name = os.path.splitext(img_name[0])[0] + ".png"
        img_rgb = normalize_imagenet(img_rgb).to(device)
        img_dep = img_dep.to(device)
        orig_w, orig_h = int(orig_size[0][0]), int(orig_size[0][1])
        with torch.no_grad():
            pred = model([img_rgb, img_dep], mode=2)
            pred = F.interpolate(pred, size=(orig_h, orig_w), mode="bilinear", align_corners=True)
            pred_img = np.squeeze(pred.cpu().numpy())
            pred_img = np.clip(pred_img, 0.0, 1.0) * 255.0
            cv2.imwrite(str(save_path / out_name), pred_img.astype(np.uint8))
    log.info("Completed | Saved to: %s", save_path)


def main(dataset=None, splits=("test", "train")):
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device(cfg)
    model_path = resolve_path("cmsalgan_ckpt", cfg)
    if not model_path.exists():
        raise FileNotFoundError(
            f"cmSalGAN checkpoint not found at {model_path}. "
            "Download it (python scripts/download_data.py --what model) "
            "and place it in models/.")
    datasets = [dataset] if dataset else os.environ.get("DATASETS", cfg["dataset"]).split(",")
    log.info("Loading cmSalGAN model from %s (device: %s)", model_path, device)
    model = UnetGAN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    for ds_name in datasets:
        for split in splits:
            if (resolve_path("data", cfg) / ds_name / split / "RGB").is_dir():
                generate_saliency_for_dataset(model, ds_name, split, cfg)
    log.info("Saliency map generation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="Dataset name, e.g. NJUD")
    args = parser.parse_args()
    main(dataset=args.dataset)
