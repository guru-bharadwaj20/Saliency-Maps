import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common import dataset_name, get_device, get_logger, load_config, resolve_path, set_seed
from datasets.saliency_dataset import INPUT_SIZE, CustomDataset, normalize_imagenet
from detection.annotations import NUM_CLASSES, load_gt_boxes_for_split
from detection.fusion_module import SaliencyFusion
from detection.metrics import evaluate_detections, evaluate_saliency, format_report
from models.cmSalGAN_model import UnetGAN

log = get_logger(__name__)


def load_models(cfg, device, model_path=None):
    """Load the trained detector, its fusion module (if any), and cmSalGAN."""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model_path = model_path or resolve_path("detector_best", cfg)
    if not os.path.exists(str(model_path)):
        raise FileNotFoundError(
            f"Trained detector not found at {model_path}. Run `python main.py --train` first.")
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "detector" not in checkpoint:
        raise RuntimeError(
            f"{model_path} is an old-format checkpoint without fusion weights. "
            "Re-train with `python main.py --train`.")

    num_classes = checkpoint.get("num_classes", NUM_CLASSES)
    detector = fasterrcnn_resnet50_fpn(weights=None)
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    detector.load_state_dict(checkpoint["detector"])
    detector.to(device).eval()

    fusion = None
    fusion_mode = checkpoint.get("fusion_mode", "attention")
    if fusion_mode != "none":
        fusion = SaliencyFusion(use_attention=(fusion_mode == "attention")).to(device)
        fusion.load_state_dict(checkpoint["fusion"])
        fusion.eval()

    saliency_model = None
    sal_ckpt = resolve_path("cmsalgan_ckpt", cfg)
    if fusion is not None:
        if sal_ckpt.exists():
            saliency_model = UnetGAN().to(device)
            saliency_model.load_state_dict(torch.load(sal_ckpt, map_location=device))
            saliency_model.eval()
        else:
            log.warning("%s not found — falling back to precomputed saliency maps "
                        "in results/saliency_maps/.", sal_ckpt)

    return detector, fusion, checkpoint, saliency_model


def get_saliency(source, saliency_model, img_rgb, img_dep, stem, sal_root, gt_dir, device):
    """Saliency at tensor resolution for one image, per the chosen source.

    precomputed — cmSalGAN live if the checkpoint exists, else the saved map
    gt          — the ground-truth mask (oracle condition, label it as such)
    zero        — all-zero map (control)
    """
    if source == "zero":
        return torch.zeros((1, 1, INPUT_SIZE, INPUT_SIZE), device=device)
    if source == "gt":
        path = os.path.join(gt_dir, stem + ".png")
    else:  # precomputed
        if saliency_model is not None:
            with torch.no_grad():
                sal = saliency_model([normalize_imagenet(img_rgb), img_dep], mode=2)
            return torch.clamp(sal, 0, 1)
        path = os.path.join(sal_root, stem + ".png")
    if os.path.exists(path):
        sal = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        sal = cv2.resize(sal, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
        return torch.from_numpy(sal)[None, None].to(device)
    return torch.zeros((1, 1, INPUT_SIZE, INPUT_SIZE), device=device)


def visualize_results(rgb_path, boxes, save_path, score_threshold):
    """Draw predicted boxes (original-image coordinates) on the original RGB."""
    img = cv2.imread(rgb_path)
    if img is None:
        return
    for box in boxes:
        x1, y1, x2, y2, score, _cls = box
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"salient {score:.2f}", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(save_path, img)


def run_inference(model_path=None, saliency_source=None, save_images=True):
    """Run the full test pipeline. Returns (preds, checkpoint_info, timing).

    preds: {image_stem: [[x1, y1, x2, y2, score, cls], ...]} in ORIGINAL
    image coordinates — the same space as the saved saliency maps and GT.
    Also writes results/detection_outputs/<DATASET>_detections.json.

    saliency_source overrides the source stored in the checkpoint
    (precomputed | gt | zero) — e.g. to test an oracle-trained model on
    realistic cmSalGAN maps.
    """
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device(cfg)
    ds_name = dataset_name(cfg)
    data_path = resolve_path("data", cfg) / ds_name / "test"
    save_path = resolve_path("detection_outputs", cfg)
    sal_root = resolve_path("saliency_maps", cfg) / ds_name
    gt_dir = data_path / "GT"
    score_threshold = cfg["detection"]["score_threshold"]

    save_path.mkdir(parents=True, exist_ok=True)
    detector, fusion, checkpoint, saliency_model = load_models(cfg, device, model_path)
    source = saliency_source or checkpoint.get("saliency_source", "precomputed")
    log.info("Testing with fusion=%s, saliency=%s",
             checkpoint.get("fusion_mode", "attention"), source)
    test_loader = DataLoader(CustomDataset(str(data_path)), batch_size=1, shuffle=False)

    preds = {}
    total_time = 0.0
    for batch in tqdm(test_loader, desc="Testing"):
        img_rgb, img_dep, img_name, orig_size = batch
        img_rgb, img_dep = img_rgb.to(device), img_dep.to(device)
        stem = os.path.splitext(img_name[0])[0]
        orig_w, orig_h = int(orig_size[0][0]), int(orig_size[0][1])

        start = time.perf_counter()
        with torch.no_grad():
            if fusion is not None:
                sal = get_saliency(source, saliency_model, img_rgb, img_dep,
                                   stem, str(sal_root), str(gt_dir), device)
                fused = fusion(img_rgb, sal)
            else:
                fused = img_rgb
            outputs = detector(list(fused))
        total_time += time.perf_counter() - start

        det = outputs[0]
        boxes = det["boxes"].cpu().numpy()
        scores = det["scores"].cpu().numpy()
        labels = det["labels"].cpu().numpy()

        # Scale from tensor space (224x224) back to original image space.
        sx, sy = orig_w / INPUT_SIZE, orig_h / INPUT_SIZE
        pred_boxes = [[float(b[0] * sx), float(b[1] * sy),
                       float(b[2] * sx), float(b[3] * sy),
                       float(scores[j]), int(labels[j])]
                      for j, b in enumerate(boxes)]
        preds[stem] = pred_boxes

        if save_images:
            visualize_results(str(data_path / "RGB" / img_name[0]), pred_boxes,
                              str(save_path / f"{stem}.jpg"), score_threshold)

    preds_path = save_path / f"{ds_name}_detections.json"
    with open(preds_path, "w") as f:
        json.dump(preds, f)
    log.info("Predictions saved to %s", preds_path)

    timing = {"avg_inference_ms": (total_time / max(len(test_loader), 1)) * 1000.0,
              "num_images": len(test_loader), "device": device}
    info = {"epoch": checkpoint.get("epoch", -1),
            "avg_loss": checkpoint.get("avg_loss", float("nan"))}
    return preds, info, timing


def evaluate(preds, timing=None):
    cfg = load_config()
    ds_name = dataset_name(cfg)
    data_path = resolve_path("data", cfg) / ds_name / "test"
    sal_root = resolve_path("saliency_maps", cfg) / ds_name
    gt_boxes = load_gt_boxes_for_split(str(data_path))
    detection_results = evaluate_detections(
        preds, gt_boxes,
        iou_threshold=cfg["detection"]["iou_threshold"],
        score_threshold=cfg["detection"]["score_threshold"]) if gt_boxes else None
    saliency_results = evaluate_saliency(str(sal_root), str(data_path / "GT"))
    return format_report(saliency_results, detection_results, timing)


if __name__ == "__main__":
    preds, info, timing = run_inference()
    log.info("Best checkpoint: epoch %s | train loss %.4f", info["epoch"], info["avg_loss"])
    print(evaluate(preds, timing))
