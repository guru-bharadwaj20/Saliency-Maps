import sys, os
import argparse
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.cmSalGAN_model import UnetGAN
from datasets.saliency_dataset import CustomDataset, normalize_imagenet

BASE_DATA_PATH = os.path.join(ROOT_DIR, "data")
SAVE_BASE_PATH = os.path.join(ROOT_DIR, "results", "saliency_maps")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "cmSalGAN.ckpt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1


def generate_saliency_for_dataset(model, dataset_name, split="test"):
    dataset_path = os.path.join(BASE_DATA_PATH, dataset_name, split)
    save_path = os.path.join(SAVE_BASE_PATH, dataset_name)
    os.makedirs(save_path, exist_ok=True)
    dataloader = DataLoader(CustomDataset(dataset_path), batch_size=BATCH_SIZE, shuffle=False)
    print(f"\nGenerating saliency maps for {dataset_name}/{split} ({len(dataloader)} images)")
    for batch in tqdm(dataloader, total=len(dataloader)):
        img_rgb, img_dep, img_name, orig_size = batch
        out_name = os.path.splitext(img_name[0])[0] + ".png"
        img_rgb = normalize_imagenet(img_rgb).to(DEVICE)
        img_dep = img_dep.to(DEVICE)
        orig_w, orig_h = int(orig_size[0][0]), int(orig_size[0][1])
        with torch.no_grad():
            pred = model([img_rgb, img_dep], mode=2)
            pred = F.interpolate(pred, size=(orig_h, orig_w), mode="bilinear", align_corners=True)
            pred_img = np.squeeze(pred.cpu().numpy())
            pred_img = np.clip(pred_img, 0.0, 1.0) * 255.0
            cv2.imwrite(os.path.join(save_path, out_name), pred_img.astype(np.uint8))
    print(f"Completed | Saved to: {save_path}")


def main(dataset=None, splits=("test", "train")):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"cmSalGAN checkpoint not found at {MODEL_PATH}. "
            "Download it (see README) and place it in models/.")
    datasets = [dataset] if dataset else os.environ.get("DATASETS", "NJUD").split(",")
    print(f"Loading cmSalGAN model from {MODEL_PATH}...")
    model = UnetGAN().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    for dataset_name in datasets:
        for split in splits:
            split_dir = os.path.join(BASE_DATA_PATH, dataset_name, split)
            if os.path.isdir(os.path.join(split_dir, "RGB")):
                generate_saliency_for_dataset(model, dataset_name, split)
    print("\nSaliency map generation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="Dataset name, e.g. NJUD")
    args = parser.parse_args()
    main(dataset=args.dataset)
