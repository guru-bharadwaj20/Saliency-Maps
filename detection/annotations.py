"""Derive detection annotations from ground-truth saliency masks.

The NJUD/NLPR/STEREO datasets ship binary saliency masks (GT/) but no
bounding boxes. Boxes are derived per image by taking connected components
of the binarized mask; every component becomes one box with class label 1
("salient_object"). This gives the detector real, verifiable supervision.
"""

import os
import cv2
import numpy as np

CLASS_NAMES = ["__background__", "salient_object"]
NUM_CLASSES = len(CLASS_NAMES)

MIN_AREA_FRACTION = 0.001   # drop components smaller than 0.1% of the image
BINARIZE_THRESHOLD = 128


def boxes_from_gt_mask(mask, min_area_fraction=MIN_AREA_FRACTION):
    """Extract [x1, y1, x2, y2] boxes from a grayscale GT mask (np.uint8).

    Returns (boxes, labels); labels are all 1 (salient_object).
    """
    binary = (mask >= BINARIZE_THRESHOLD).astype(np.uint8)
    num, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = min_area_fraction * mask.shape[0] * mask.shape[1]
    boxes, labels = [], []
    for i in range(1, num):  # 0 is background
        x, y, w, h, area = stats[i]
        if area < min_area or w < 2 or h < 2:
            continue
        boxes.append([float(x), float(y), float(x + w), float(y + h)])
        labels.append(1)
    return boxes, labels


def load_gt_boxes(gt_dir, image_name, target_size=None):
    """Load boxes for one image from its GT mask file.

    image_name may carry any extension; the mask is looked up as <stem>.png.
    If target_size (w, h) is given, boxes are scaled from mask resolution
    to that size (e.g. the 224x224 tensor the detector actually sees).
    """
    stem = os.path.splitext(os.path.basename(image_name))[0]
    mask_path = os.path.join(gt_dir, stem + ".png")
    if not os.path.exists(mask_path):
        return [], []
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return [], []
    boxes, labels = boxes_from_gt_mask(mask)
    if target_size is not None and boxes:
        sx = target_size[0] / mask.shape[1]
        sy = target_size[1] / mask.shape[0]
        boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes]
    return boxes, labels


def load_gt_boxes_for_split(split_dir):
    """Boxes for every image of a split at original mask resolution.

    Returns {image_stem: [[x1, y1, x2, y2], ...]} — the format expected by
    detection.metrics.evaluate_detections.
    """
    gt_dir = os.path.join(split_dir, "GT")
    if not os.path.isdir(gt_dir):
        return {}
    result = {}
    for fname in sorted(os.listdir(gt_dir)):
        stem = os.path.splitext(fname)[0]
        boxes, _ = load_gt_boxes(gt_dir, fname)
        result[stem] = boxes
    return result
