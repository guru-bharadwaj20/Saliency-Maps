import sys, os, json, time
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.cmSalGAN_model import UnetGAN
from datasets.saliency_dataset import CustomDataset, normalize_imagenet, INPUT_SIZE
from detection.fusion_module import SaliencyFusion
from detection.annotations import load_gt_boxes_for_split, NUM_CLASSES
from detection.metrics import evaluate_saliency, evaluate_detections, format_report

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_NAME = os.environ.get("DATASETS", "NJUD").split(",")[0]
MODEL_PATH = os.path.join(ROOT_DIR, "models", "detector_best.pt")
SAL_MODEL_PATH = os.path.join(ROOT_DIR, "models", "cmSalGAN.ckpt")
DATA_PATH = os.path.join(ROOT_DIR, "data", DATASET_NAME, "test")
SAVE_PATH = os.path.join(ROOT_DIR, "results", "detection_outputs")
SAL_ROOT = os.path.join(ROOT_DIR, "results", "saliency_maps", DATASET_NAME)
SCORE_THRESHOLD = 0.5


def load_models():
    """Load cmSalGAN, the trained fusion module, and the trained detector."""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained detector not found at {MODEL_PATH}. Run `python main.py --train` first.")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    if not isinstance(checkpoint, dict) or "detector" not in checkpoint:
        raise RuntimeError(
            f"{MODEL_PATH} is an old-format checkpoint without fusion weights. "
            "Re-train with `python main.py --train`.")

    num_classes = checkpoint.get("num_classes", NUM_CLASSES)
    detector = fasterrcnn_resnet50_fpn(weights=None)
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    detector.load_state_dict(checkpoint["detector"])
    detector.to(DEVICE).eval()

    fusion = SaliencyFusion(use_attention=True).to(DEVICE)
    fusion.load_state_dict(checkpoint["fusion"])
    fusion.eval()

    saliency_model = None
    if os.path.exists(SAL_MODEL_PATH):
        saliency_model = UnetGAN().to(DEVICE)
        saliency_model.load_state_dict(torch.load(SAL_MODEL_PATH, map_location=DEVICE))
        saliency_model.eval()
    else:
        print(f"[test] WARNING: {SAL_MODEL_PATH} not found — "
              "falling back to precomputed saliency maps in results/saliency_maps/.")

    return detector, fusion, checkpoint, saliency_model


def get_saliency(saliency_model, img_rgb, img_dep, stem):
    """Saliency at tensor resolution: cmSalGAN if available, else precomputed map."""
    if saliency_model is not None:
        with torch.no_grad():
            sal = saliency_model([normalize_imagenet(img_rgb), img_dep], mode=2)
        return torch.clamp(sal, 0, 1)
    sal_path = os.path.join(SAL_ROOT, stem + ".png")
    if os.path.exists(sal_path):
        sal = cv2.imread(sal_path, cv2.IMREAD_GRAYSCALE)
        sal = cv2.resize(sal, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
        return torch.from_numpy(sal)[None, None].to(DEVICE)
    return torch.zeros((1, 1, INPUT_SIZE, INPUT_SIZE), device=DEVICE)


def visualize_results(rgb_path, boxes, save_path, score_threshold=SCORE_THRESHOLD):
    """Draw predicted boxes (original-image coordinates) on the original RGB."""
    img = cv2.imread(rgb_path)
    if img is None:
        return
    for box in boxes:
        x1, y1, x2, y2, score, cls = box
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"salient {score:.2f}", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(save_path, img)


def run_inference():
    """Run the full test pipeline. Returns (preds, checkpoint_info, timing).

    preds: {image_stem: [[x1, y1, x2, y2, score, cls], ...]} in ORIGINAL
    image coordinates — the same space as the saved saliency maps and GT.
    Also writes results/detection_outputs/<DATASET>_detections.json.
    """
    os.makedirs(SAVE_PATH, exist_ok=True)
    detector, fusion, checkpoint, saliency_model = load_models()
    test_loader = DataLoader(CustomDataset(DATA_PATH), batch_size=1, shuffle=False)

    preds = {}
    total_time = 0.0
    for batch in tqdm(test_loader, desc="Testing"):
        img_rgb, img_dep, img_name, orig_size = batch
        img_rgb, img_dep = img_rgb.to(DEVICE), img_dep.to(DEVICE)
        stem = os.path.splitext(img_name[0])[0]
        orig_w, orig_h = int(orig_size[0][0]), int(orig_size[0][1])

        start = time.perf_counter()
        sal = get_saliency(saliency_model, img_rgb, img_dep, stem)
        with torch.no_grad():
            fused = fusion(img_rgb, sal)
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

        visualize_results(os.path.join(DATA_PATH, "RGB", img_name[0]),
                          pred_boxes, os.path.join(SAVE_PATH, f"{stem}.jpg"))

    preds_path = os.path.join(SAVE_PATH, f"{DATASET_NAME}_detections.json")
    with open(preds_path, "w") as f:
        json.dump(preds, f)
    print(f"Predictions saved to {preds_path}")

    timing = {"avg_inference_ms": (total_time / max(len(test_loader), 1)) * 1000.0,
              "num_images": len(test_loader), "device": DEVICE}
    info = {"epoch": checkpoint.get("epoch", -1), "avg_loss": checkpoint.get("avg_loss", float("nan"))}
    return preds, info, timing


def evaluate(preds, timing=None):
    gt_boxes = load_gt_boxes_for_split(DATA_PATH)
    detection_results = evaluate_detections(preds, gt_boxes) if gt_boxes else None
    saliency_results = evaluate_saliency(SAL_ROOT, os.path.join(DATA_PATH, "GT"))
    return format_report(saliency_results, detection_results, timing)


if __name__ == "__main__":
    preds, info, timing = run_inference()
    print(f"\nBest checkpoint: epoch {info['epoch']} | train loss {info['avg_loss']:.4f}")
    print(evaluate(preds, timing))
