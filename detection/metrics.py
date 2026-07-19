"""Real evaluation metrics for the saliency + detection pipeline.

Saliency metrics (predicted saliency map vs. ground-truth mask):
    - MAE                (lower is better)
    - Adaptive F-measure (beta^2 = 0.3, threshold = 2 * mean of prediction)
    - S-measure          (structure measure, Fan et al. 2017)

Detection metrics (predicted boxes vs. ground-truth boxes):
    - IoU
    - AP@0.5 (VOC-style, all-points interpolation)
    - Precision / Recall / F1 at a fixed score threshold
"""

import os
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Saliency metrics
# ---------------------------------------------------------------------------

def mae(pred, gt):
    """Mean absolute error between two [0, 1] float maps."""
    return float(np.mean(np.abs(pred - gt)))


def adaptive_fmeasure(pred, gt, beta_sq=0.3):
    """F-measure with the standard adaptive threshold (2 * mean saliency)."""
    thresh = min(2.0 * pred.mean(), 1.0)
    binary = pred >= thresh
    gt_bin = gt >= 0.5
    tp = np.logical_and(binary, gt_bin).sum()
    if tp == 0:
        return 0.0
    precision = tp / (binary.sum() + 1e-8)
    recall = tp / (gt_bin.sum() + 1e-8)
    return float((1 + beta_sq) * precision * recall / (beta_sq * precision + recall + 1e-8))


def _s_object(pred, gt):
    """Object-aware structural similarity component of S-measure."""
    def object_score(region):
        if region.size == 0:
            return 0.0
        x = region.mean()
        sigma = region.std()
        return 2.0 * x / (x * x + 1.0 + sigma + 1e-8)

    gt_bin = gt >= 0.5
    fg = pred[gt_bin]
    bg = 1.0 - pred[~gt_bin]
    u = gt_bin.mean()
    return u * object_score(fg) + (1 - u) * object_score(bg)


def _ssim_region(pred, gt):
    """SSIM-like score used inside the region term of S-measure."""
    n = pred.size
    if n == 0:
        return 0.0
    x, y = pred.mean(), gt.mean()
    sigma_x = pred.var()
    sigma_y = gt.var()
    sigma_xy = ((pred - x) * (gt - y)).sum() / max(n - 1, 1)
    alpha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x + sigma_y)
    if alpha != 0:
        return alpha / (beta + 1e-8)
    return 1.0 if beta == 0 else 0.0


def _s_region(pred, gt):
    """Region-aware component: split at GT centroid into 4 quadrants."""
    gt_bin = gt >= 0.5
    h, w = gt.shape
    if gt_bin.sum() == 0:
        cx, cy = w // 2, h // 2
    else:
        ys, xs = np.nonzero(gt_bin)
        cx, cy = int(xs.mean()) + 1, int(ys.mean()) + 1
    cx = min(max(cx, 1), w - 1)
    cy = min(max(cy, 1), h - 1)

    total = h * w
    score, weights = 0.0, []
    for (y0, y1, x0, x1) in [(0, cy, 0, cx), (0, cy, cx, w), (cy, h, 0, cx), (cy, h, cx, w)]:
        p_q, g_q = pred[y0:y1, x0:x1], gt[y0:y1, x0:x1]
        wgt = p_q.size / total
        score += wgt * _ssim_region(p_q, g_q)
    return score


def s_measure(pred, gt, alpha=0.5):
    """Structure measure (Fan et al., ICCV 2017)."""
    y = (gt >= 0.5).mean()
    if y == 0:      # GT entirely background
        return 1.0 - pred.mean()
    if y == 1:      # GT entirely foreground
        return float(pred.mean())
    score = alpha * _s_object(pred, gt) + (1 - alpha) * _s_region(pred, gt)
    return float(max(score, 0.0))


def evaluate_saliency(pred_dir, gt_dir):
    """Compare every predicted saliency map in pred_dir against gt_dir.

    Returns a dict of averaged metrics, or None if no pairs were found.
    """
    if not os.path.isdir(pred_dir) or not os.path.isdir(gt_dir):
        return None
    maes, fms, sms = [], [], []
    for name in sorted(os.listdir(pred_dir)):
        gt_path = os.path.join(gt_dir, os.path.splitext(name)[0] + ".png")
        pred_path = os.path.join(pred_dir, name)
        if not os.path.exists(gt_path):
            continue
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            continue
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        pred = pred.astype(np.float32) / 255.0
        gt = gt.astype(np.float32) / 255.0
        maes.append(mae(pred, gt))
        fms.append(adaptive_fmeasure(pred, gt))
        sms.append(s_measure(pred, gt))
    if not maes:
        return None
    return {
        "num_images": len(maes),
        "MAE": float(np.mean(maes)),
        "F-measure": float(np.mean(fms)),
        "S-measure": float(np.mean(sms)),
    }


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------

def compute_iou(box_a, box_b):
    """IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return float(inter / (area_a + area_b - inter + 1e-8))


def average_precision(preds, gts, iou_threshold=0.5):
    """VOC-style AP over a whole split (single foreground class).

    preds: {image_name: [[x1, y1, x2, y2, score], ...]}
    gts:   {image_name: [[x1, y1, x2, y2], ...]}
    """
    all_preds = []
    total_gt = 0
    for name, boxes in gts.items():
        total_gt += len(boxes)
    for name, boxes in preds.items():
        for b in boxes:
            all_preds.append((float(b[4]), name, [float(v) for v in b[:4]]))
    if total_gt == 0 or not all_preds:
        return 0.0, 0.0, 0.0

    all_preds.sort(key=lambda t: -t[0])
    matched = {name: np.zeros(len(boxes), dtype=bool) for name, boxes in gts.items()}
    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))
    for i, (_, name, box) in enumerate(all_preds):
        gt_boxes = gts.get(name, [])
        best_iou, best_j = 0.0, -1
        for j, gt_box in enumerate(gt_boxes):
            iou = compute_iou(box, gt_box)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold and not matched[name][best_j]:
            tp[i] = 1
            matched[name][best_j] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / total_gt
    precision = tp_cum / (tp_cum + fp_cum + 1e-8)

    # All-points interpolation: precision envelope, integrate over recall.
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap, float(recall[-1]), float(precision[-1])


def precision_recall_at(preds, gts, score_threshold=0.5, iou_threshold=0.5):
    """Precision/recall/F1 counting only predictions above score_threshold."""
    filtered = {n: [b for b in boxes if float(b[4]) >= score_threshold]
                for n, boxes in preds.items()}
    tp, fp, total_gt = 0, 0, 0
    for name, gt_boxes in gts.items():
        total_gt += len(gt_boxes)
        taken = np.zeros(len(gt_boxes), dtype=bool)
        for b in sorted(filtered.get(name, []), key=lambda x: -float(x[4])):
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gt_boxes):
                iou = compute_iou(b[:4], g)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_threshold and not taken[best_j]:
                tp += 1
                taken[best_j] = True
            else:
                fp += 1
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (total_gt + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"precision": float(precision), "recall": float(recall), "F1": float(f1),
            "TP": tp, "FP": fp, "total_GT": total_gt}


def evaluate_detections(preds, gts, iou_threshold=0.5, score_threshold=0.5):
    """Full detection report. preds boxes: [x1,y1,x2,y2,score,cls]; gts: [x1,y1,x2,y2]."""
    ap, recall, precision = average_precision(preds, gts, iou_threshold)
    fixed = precision_recall_at(preds, gts, score_threshold, iou_threshold)
    return {"AP@0.5": ap, "max_recall": recall, "final_precision": precision, **fixed}


def format_report(saliency_results, detection_results, timing=None):
    lines = ["===== Evaluation Results ====="]
    if saliency_results:
        lines.append(f"\nSaliency ({saliency_results['num_images']} images vs GT):")
        lines.append(f"  MAE:        {saliency_results['MAE']:.4f}")
        lines.append(f"  F-measure:  {saliency_results['F-measure']:.4f}")
        lines.append(f"  S-measure:  {saliency_results['S-measure']:.4f}")
    else:
        lines.append("\nSaliency: no GT/prediction pairs found — skipped.")
    if detection_results:
        lines.append(f"\nDetection (IoU >= 0.5, {detection_results['total_GT']} GT boxes):")
        lines.append(f"  AP@0.5:     {detection_results['AP@0.5']:.4f}")
        lines.append(f"  Precision:  {detection_results['precision']:.4f} (score >= 0.5)")
        lines.append(f"  Recall:     {detection_results['recall']:.4f}")
        lines.append(f"  F1:         {detection_results['F1']:.4f}")
    else:
        lines.append("\nDetection: no ground-truth boxes available — skipped.")
    if timing:
        lines.append(f"\nAvg inference time: {timing['avg_inference_ms']:.1f} ms/image "
                     f"({timing['num_images']} images, {timing['device']})")
    lines.append("==============================")
    return "\n".join(lines)
