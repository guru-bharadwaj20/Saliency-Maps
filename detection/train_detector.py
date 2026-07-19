import sys, os, time
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from detection.fusion_module import SaliencyFusion
from detection.annotations import load_gt_boxes, NUM_CLASSES
from datasets.saliency_dataset import CustomDataset, INPUT_SIZE

DATASET_NAME = os.environ.get("DATASETS", "NJUD").split(",")[0]
BASE_DATA_PATH = os.path.join(ROOT_DIR, "data")
PRECOMPUTED_SAL_PATH = os.path.join(ROOT_DIR, "results", "saliency_maps", DATASET_NAME)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 3
LR = 1e-4
NUM_WORKERS = 0
SAVE_DIR = os.path.join(ROOT_DIR, "models", "checkpoints")
BEST_MODEL_PATH = os.path.join(ROOT_DIR, "models", "detector_best.pt")


class FusedDetectionDataset(Dataset):
    """RGB + precomputed saliency + boxes derived from GT saliency masks.

    If a precomputed saliency map is missing the sample falls back to a
    zero map (and says so once), so training degrades gracefully instead of
    silently training on garbage.
    """

    def __init__(self, root_dir, split="train", saliency_dir=None):
        self.root_dir = os.path.join(root_dir, split)
        self.custom = CustomDataset(self.root_dir)
        self.saliency_dir = saliency_dir
        self.gt_dir = os.path.join(self.root_dir, "GT")
        self._warned_missing_sal = False

    def __len__(self):
        return len(self.custom)

    def __getitem__(self, idx):
        rgb, dep, img_name, orig_size = self.custom[idx]
        stem = os.path.splitext(img_name)[0]

        sal_path = os.path.join(self.saliency_dir or "", stem + ".png")
        if self.saliency_dir and os.path.exists(sal_path):
            sal = Image.open(sal_path).convert("L")
            sal = transforms.ToTensor()(sal)
            sal = transforms.Resize(rgb.shape[1:])(sal)
        else:
            if not self._warned_missing_sal:
                print(f"[train] WARNING: no precomputed saliency map for {stem} "
                      f"(looked in {self.saliency_dir}); using zeros. "
                      "Run `python main.py --saliency` first for real fusion input.")
                self._warned_missing_sal = True
            sal = torch.zeros((1, rgb.shape[1], rgb.shape[2]))

        # Boxes from the GT mask, scaled to the tensor resolution the model sees.
        boxes, labels = load_gt_boxes(self.gt_dir, img_name,
                                      target_size=(INPUT_SIZE, INPUT_SIZE))
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes
            else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels
            else torch.zeros((0,), dtype=torch.int64),
        }
        return rgb, sal, target, stem


def get_fasterrcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model.to(DEVICE)


def collate_fn(batch):
    rgbs = torch.stack([b[0] for b in batch], dim=0)
    sals = torch.stack([b[1] for b in batch], dim=0)
    targets = [b[2] for b in batch]
    names = [b[3] for b in batch]
    return rgbs, sals, targets, names


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    dataset = FusedDetectionDataset(
        root_dir=os.path.join(BASE_DATA_PATH, DATASET_NAME),
        split="train",
        saliency_dir=PRECOMPUTED_SAL_PATH)
    n_annotated = sum(1 for i in range(len(dataset)) if len(dataset[i][2]["boxes"]) > 0)
    print(f"Dataset: {DATASET_NAME}/train — {len(dataset)} images, "
          f"{n_annotated} with GT-derived boxes")
    if n_annotated == 0:
        raise RuntimeError(
            "No ground-truth boxes could be derived from GT masks — "
            f"check that {os.path.join(BASE_DATA_PATH, DATASET_NAME, 'train', 'GT')} "
            "contains binary saliency masks.")

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, collate_fn=collate_fn)
    model = get_fasterrcnn_model(NUM_CLASSES)
    model.train()
    fusion = SaliencyFusion(use_attention=True).to(DEVICE)
    fusion.train()
    params = [p for p in model.parameters() if p.requires_grad] + list(fusion.parameters())
    optimizer = optim.Adam(params, lr=LR)

    best_loss = float("inf")
    best_epoch = -1
    print(f"Starting training on {DEVICE}...")
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        start_time = time.time()
        pbar = tqdm(data_loader)
        for rgbs, sals, targets, names in pbar:
            rgbs, sals = rgbs.to(DEVICE), sals.to(DEVICE)
            fused = fusion(rgbs, sals)
            tv_targets = [{"boxes": t["boxes"].to(DEVICE),
                           "labels": t["labels"].to(DEVICE)} for t in targets]
            losses = model(list(fused), tv_targets)
            loss = sum(losses.values()) if isinstance(losses, dict) else losses
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_description(f"Epoch {epoch + 1}/{NUM_EPOCHS} Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(data_loader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1} — avg loss: {avg_loss:.4f} — time: {elapsed:.1f}s")

        checkpoint = {
            "detector": model.state_dict(),
            "fusion": fusion.state_dict(),
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "num_classes": NUM_CLASSES,
        }
        ckpt_path = os.path.join(SAVE_DIR, f"detector_epoch_{epoch + 1}.pth")
        torch.save(checkpoint, ckpt_path)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            torch.save(checkpoint, BEST_MODEL_PATH)
            print(f"Best model updated (Epoch {best_epoch}, Loss {best_loss:.4f})")

    print(f"Training finished — Best model: Epoch {best_epoch} | Loss: {best_loss:.4f}")
    print(f"Saved to {BEST_MODEL_PATH} (contains detector + fusion weights)")


if __name__ == "__main__":
    train()
