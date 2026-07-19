"""Train Faster R-CNN (+ optional saliency fusion) on GT-derived boxes."""

import os
import time

import torch
import torch.optim as optim
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_ResNet50_FPN_Weights,
    FastRCNNPredictor,
)
from tqdm import tqdm

from saliency_driving.annotations import NUM_CLASSES, load_gt_boxes
from saliency_driving.config import (
    dataset_name,
    get_device,
    load_config,
    resolve_path,
    set_seed,
)
from saliency_driving.datasets import INPUT_SIZE, CustomDataset
from saliency_driving.fusion import SaliencyFusion
from saliency_driving.log import get_logger

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

log = get_logger(__name__)


class FusedDetectionDataset(Dataset):
    """RGB + saliency + GT-derived boxes. saliency_source: precomputed | gt | zero."""

    def __init__(self, root_dir, split="train", saliency_dir=None,
                 saliency_source="precomputed"):
        self.root_dir = os.path.join(root_dir, split)
        self.custom = CustomDataset(self.root_dir)
        self.saliency_dir = saliency_dir
        self.saliency_source = saliency_source
        self.gt_dir = os.path.join(self.root_dir, "GT")
        self._warned_missing_sal = False

    def __len__(self):
        return len(self.custom)

    def _load_saliency(self, stem, shape):
        if self.saliency_source == "zero":
            return torch.zeros((1, *shape))
        if self.saliency_source == "gt":
            path = os.path.join(self.gt_dir, stem + ".png")
        else:
            path = os.path.join(self.saliency_dir or "", stem + ".png")
        if os.path.exists(path):
            sal = transforms.ToTensor()(Image.open(path).convert("L"))
            return transforms.Resize(shape)(sal)
        if not self._warned_missing_sal:
            log.warning("No %s saliency map for %s (looked at %s); using zeros. "
                        "Run `python main.py --saliency` first for real fusion input.",
                        self.saliency_source, stem, path)
            self._warned_missing_sal = True
        return torch.zeros((1, *shape))

    def __getitem__(self, idx):
        rgb, dep, img_name, orig_size = self.custom[idx]
        stem = os.path.splitext(img_name)[0]
        sal = self._load_saliency(stem, rgb.shape[1:])
        boxes, labels = load_gt_boxes(self.gt_dir, img_name,
                                      target_size=(INPUT_SIZE, INPUT_SIZE))
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes
            else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels
            else torch.zeros((0,), dtype=torch.int64),
        }
        return rgb, sal, target, stem


def get_fasterrcnn_model(num_classes, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model.to(device)


def collate_fn(batch):
    rgbs = torch.stack([b[0] for b in batch], dim=0)
    sals = torch.stack([b[1] for b in batch], dim=0)
    targets = [b[2] for b in batch]
    names = [b[3] for b in batch]
    return rgbs, sals, targets, names


def train(fusion_mode=None, saliency_source=None, epochs=None, out_path=None):
    """Train the detector (+ optional fusion) and return a run summary dict."""
    cfg = load_config()
    seed = set_seed(cfg["seed"])
    device = get_device(cfg)
    ds_name = dataset_name(cfg)
    tcfg = cfg["training"]
    fusion_mode = fusion_mode or tcfg["fusion_mode"]
    saliency_source = saliency_source or tcfg["saliency_source"]
    epochs = epochs or tcfg["epochs"]
    if fusion_mode not in ("attention", "plain", "none"):
        raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

    save_dir = resolve_path("checkpoints", cfg)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = out_path or resolve_path("detector_best", cfg)
    saliency_dir = resolve_path("saliency_maps", cfg) / ds_name

    dataset = FusedDetectionDataset(
        root_dir=str(resolve_path("data", cfg) / ds_name),
        split="train",
        saliency_dir=str(saliency_dir),
        saliency_source=saliency_source)
    n_annotated = sum(1 for i in range(len(dataset)) if len(dataset[i][2]["boxes"]) > 0)
    log.info("Dataset: %s/train — %d images, %d with GT-derived boxes "
             "(fusion=%s, saliency=%s, seed=%d, device=%s)",
             ds_name, len(dataset), n_annotated, fusion_mode, saliency_source, seed, device)
    if n_annotated == 0:
        gt_dir = resolve_path("data", cfg) / ds_name / "train" / "GT"
        raise RuntimeError(
            "No ground-truth boxes could be derived from GT masks — check that "
            f"{gt_dir} contains binary saliency masks.")

    data_loader = DataLoader(dataset, batch_size=tcfg["batch_size"], shuffle=True,
                             num_workers=tcfg["num_workers"], collate_fn=collate_fn)
    model = get_fasterrcnn_model(NUM_CLASSES, device)
    model.train()
    fusion = None
    params = [p for p in model.parameters() if p.requires_grad]
    if fusion_mode != "none":
        fusion = SaliencyFusion(use_attention=(fusion_mode == "attention")).to(device)
        fusion.train()
        params += list(fusion.parameters())
    optimizer = optim.Adam(params, lr=tcfg["lr"])

    writer = None
    if SummaryWriter is not None:
        tb_dir = resolve_path("tensorboard", cfg) / time.strftime(
            f"{ds_name}_{fusion_mode}_{saliency_source}_%Y%m%d_%H%M%S")
        writer = SummaryWriter(log_dir=str(tb_dir))
        log.info("TensorBoard logs: %s", tb_dir)
    else:
        log.info("tensorboard not installed — skipping loss curves.")

    best_loss = float("inf")
    best_epoch = -1
    global_step = 0
    train_start = time.time()
    log.info("Starting training: %d epochs, batch %d, lr %g",
             epochs, tcfg["batch_size"], tcfg["lr"])
    for epoch in range(epochs):
        epoch_loss = 0.0
        start_time = time.time()
        pbar = tqdm(data_loader)
        for rgbs, sals, targets, _names in pbar:
            rgbs, sals = rgbs.to(device), sals.to(device)
            fused = fusion(rgbs, sals) if fusion is not None else rgbs
            tv_targets = [{"boxes": t["boxes"].to(device),
                           "labels": t["labels"].to(device)} for t in targets]
            losses = model(list(fused), tv_targets)
            loss = sum(losses.values()) if isinstance(losses, dict) else losses
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_loss = loss.item()
            epoch_loss += step_loss
            if writer:
                writer.add_scalar("train/loss_step", step_loss, global_step)
            global_step += 1
            pbar.set_description(f"Epoch {epoch + 1}/{epochs} Loss: {step_loss:.4f}")

        avg_loss = epoch_loss / len(data_loader)
        elapsed = time.time() - start_time
        if writer:
            writer.add_scalar("train/loss_epoch", avg_loss, epoch + 1)
        log.info("Epoch %d — avg loss: %.4f — time: %.1fs", epoch + 1, avg_loss, elapsed)

        checkpoint = {
            "detector": model.state_dict(),
            "fusion": fusion.state_dict() if fusion is not None else None,
            "fusion_mode": fusion_mode,
            "saliency_source": saliency_source,
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "num_classes": NUM_CLASSES,
            "seed": seed,
        }
        torch.save(checkpoint, save_dir / f"detector_epoch_{epoch + 1}.pth")
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            torch.save(checkpoint, best_model_path)
            log.info("Best model updated (Epoch %d, Loss %.4f)", best_epoch, best_loss)

    if writer:
        writer.close()
    total_time = time.time() - train_start
    log.info("Training finished — Best model: Epoch %d | Loss: %.4f", best_epoch, best_loss)
    log.info("Saved to %s", best_model_path)
    return {"best_epoch": best_epoch, "best_loss": best_loss,
            "fusion_mode": fusion_mode, "saliency_source": saliency_source,
            "out_path": str(best_model_path), "train_time_s": total_time}


if __name__ == "__main__":
    train()
