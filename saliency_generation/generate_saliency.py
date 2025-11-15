import sys, os
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
from datasets.saliency_dataset import CustomDataset

DATASETS = ["NJUD"]
BASE_DATA_PATH = "./data"
SAVE_BASE_PATH = "./results/saliency_maps"
MODEL_PATH = "./models/cmSalGAN.ckpt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1

def generate_saliency_for_dataset(model, dataset_name):
    dataset_path = os.path.join(BASE_DATA_PATH, dataset_name, "test")
    save_path = os.path.join(SAVE_BASE_PATH, dataset_name)
    os.makedirs(save_path, exist_ok=True)
    dataloader = DataLoader(CustomDataset(dataset_path), batch_size=BATCH_SIZE, shuffle=False)
    print(f"\nGenerating Saliency Maps!")
    for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        img_rgb, img_dep, img_name, size = batch
        img_name = img_name[0].split(".")[0] + ".png"
        img_rgb, img_dep = img_rgb.to(DEVICE), img_dep.to(DEVICE)
        with torch.no_grad():
            pred = model([img_rgb, img_dep], mode=2)
            pred_resized = F.interpolate(pred, [size[1], size[0]], mode="bilinear", align_corners=True)
            pred_img = np.squeeze(pred_resized.cpu().numpy()) * 255.0
            cv2.imwrite(os.path.join(save_path, img_name), pred_img)
    print(f"Completed | Saved to: {save_path}")

def main():
    print(f"Loading cmSalGAN model from {MODEL_PATH}...")
    model = UnetGAN().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    for dataset_name in DATASETS:
        generate_saliency_for_dataset(model, dataset_name)
    print("\nSaliency map Generation completed!")

if __name__ == "__main__":
    main()
