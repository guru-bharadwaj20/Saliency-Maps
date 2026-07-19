import numpy as np
import pytest

from common.config import ROOT
from detection.annotations import (
    NUM_CLASSES,
    boxes_from_gt_mask,
    load_gt_boxes,
    load_gt_boxes_for_split,
)


def test_num_classes():
    assert NUM_CLASSES == 2  # background + salient_object


def test_single_component():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:60, 30:80] = 255
    boxes, labels = boxes_from_gt_mask(mask)
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]
    assert (x1, y1, x2, y2) == (30, 20, 80, 60)
    assert labels == [1]


def test_two_components():
    mask = np.zeros((100, 100), np.uint8)
    mask[10:30, 10:30] = 255
    mask[60:90, 60:90] = 255
    boxes, _ = boxes_from_gt_mask(mask)
    assert len(boxes) == 2


def test_tiny_components_dropped():
    mask = np.zeros((100, 100), np.uint8)
    mask[50, 50] = 255  # single pixel, below min area
    boxes, labels = boxes_from_gt_mask(mask)
    assert boxes == [] and labels == []


def test_empty_mask():
    boxes, labels = boxes_from_gt_mask(np.zeros((50, 50), np.uint8))
    assert boxes == [] and labels == []


def test_load_with_scaling(tiny_split):
    # mask is 60x40 with box (20,10,50,30); scale to 120x80 doubles coords
    boxes, labels = load_gt_boxes(str(tiny_split / "GT"), "img_a.jpg", target_size=(120, 80))
    assert len(boxes) == 1
    assert boxes[0] == [40.0, 20.0, 100.0, 60.0]


def test_missing_mask(tiny_split):
    boxes, labels = load_gt_boxes(str(tiny_split / "GT"), "nope.jpg")
    assert boxes == [] and labels == []


@pytest.mark.skipif(not (ROOT / "data" / "NJUD" / "test" / "GT").is_dir(),
                    reason="NJUD sample data not present")
def test_real_njud_split():
    result = load_gt_boxes_for_split(str(ROOT / "data" / "NJUD" / "test"))
    assert len(result) == 10
    assert sum(1 for v in result.values() if v) >= 8
