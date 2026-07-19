"""Slow integration test: one forward/backward through the full training path.

Run with:  pytest -m slow
Skipped in the default (fast) test run and in CI.
"""

import pytest
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from detection.annotations import NUM_CLASSES
from detection.fusion_module import SaliencyFusion


@pytest.mark.slow
def test_one_training_step(tiny_split):
    from torch.utils.data import DataLoader

    from detection.train_detector import FusedDetectionDataset, collate_fn

    ds = FusedDetectionDataset(root_dir=str(tiny_split.parent), split=tiny_split.name,
                               saliency_dir=None)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None,
                                                                 weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.train()
    fusion = SaliencyFusion(use_attention=True).train()

    rgbs, sals, targets, names = next(iter(loader))
    assert all(len(t["boxes"]) > 0 for t in targets), "GT-derived boxes expected"
    fused = fusion(rgbs, sals)
    losses = model(list(fused), [dict(t) for t in targets])
    loss = sum(losses.values())
    loss.backward()
    grad = next(fusion.parameters()).grad
    assert grad is not None and float(grad.abs().sum()) > 0
    assert torch.isfinite(loss)
