"""End-to-end inference on a single arbitrary RGB image.

Chain: RGB → monocular depth (MiDaS) → saliency (cmSalGAN, or spectral-
residual fallback) → optional fusion → Faster R-CNN detections → overlay,
plus an optional car-simulation render on the resulting saliency field.

This is the engine behind the Gradio demo (app.py) and the
`python main.py --infer <image>` CLI. Sources are always labeled honestly:
the result dict reports exactly which depth/saliency method produced it.
"""

import os
import sys

import cv2
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common import get_device, get_logger, load_config, resolve_path
from datasets.saliency_dataset import INPUT_SIZE, normalize_imagenet
from detection.test_detector import load_models
from models.cmSalGAN_model import UnetGAN
from saliency_generation import heuristic_saliency, mono_depth
from simulation.car_simulator import CarSimulator
from simulation.visualize import draw_car_state, draw_detections, overlay_saliency, scale_boxes

log = get_logger(__name__)


class SingleImagePipeline:
    """Loads all models once; call .run(rgb) per image."""

    def __init__(self, model_path=None, use_mono_depth=True):
        self.cfg = load_config()
        self.device = get_device(self.cfg)
        self.use_mono_depth = use_mono_depth and mono_depth.available()
        self.detector, self.fusion, self.checkpoint, self.saliency_model = \
            load_models(self.cfg, self.device, model_path)
        # load_models only loads cmSalGAN when fusion is present; the pipeline
        # wants it for saliency display/steering even with a no-fusion detector.
        sal_ckpt = resolve_path("cmsalgan_ckpt", self.cfg)
        if self.saliency_model is None and sal_ckpt.exists():
            self.saliency_model = UnetGAN().to(self.device)
            self.saliency_model.load_state_dict(
                torch.load(sal_ckpt, map_location=self.device))
            self.saliency_model.eval()
        log.info("Pipeline ready: fusion=%s, cmSalGAN=%s, mono_depth=%s, device=%s",
                 self.checkpoint.get("fusion_mode", "attention"),
                 self.saliency_model is not None, self.use_mono_depth, self.device)

    # ------------------------------------------------------------------
    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """(H, W, C) uint8 → (1, C, INPUT_SIZE, INPUT_SIZE) float [0, 1]."""
        resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        if resized.ndim == 2:
            resized = resized[..., None]
        t = torch.from_numpy(resized.astype(np.float32) / 255.0)
        return t.permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _estimate_saliency(self, rgb: np.ndarray, depth: np.ndarray):
        """Returns (saliency float [0,1] at original resolution, source label)."""
        if self.saliency_model is not None and depth is not None:
            rgb_t = normalize_imagenet(self._to_tensor(rgb))
            depth_t = self._to_tensor(
                (depth * 255).astype(np.uint8)).repeat(1, 3, 1, 1)
            with torch.no_grad():
                sal = self.saliency_model([rgb_t, depth_t], mode=2)
            sal = torch.clamp(sal, 0, 1).squeeze().cpu().numpy()
            sal = cv2.resize(sal, (rgb.shape[1], rgb.shape[0]))
            return sal.astype(np.float32), "cmSalGAN (RGB + MiDaS depth)"
        return (heuristic_saliency.spectral_residual_saliency(rgb),
                "spectral residual (heuristic fallback — no cmSalGAN checkpoint)")

    # ------------------------------------------------------------------
    def run(self, rgb: np.ndarray) -> dict:
        """rgb: (H, W, 3) uint8, RGB channel order. Returns a result dict."""
        h, w = rgb.shape[:2]

        depth = None
        depth_source = "unavailable (pip install timm for MiDaS)"
        if self.use_mono_depth:
            depth = mono_depth.estimate_depth(rgb)
            depth_source = "MiDaS_small (monocular)"

        saliency, sal_source = self._estimate_saliency(rgb, depth)

        rgb_t = self._to_tensor(rgb)
        with torch.no_grad():
            if self.fusion is not None:
                sal_small = cv2.resize(saliency, (INPUT_SIZE, INPUT_SIZE))
                sal_t = torch.from_numpy(sal_small)[None, None].to(self.device)
                det_in = self.fusion(rgb_t, sal_t)
            else:
                det_in = rgb_t
            outputs = self.detector(list(det_in))[0]

        sx, sy = w / INPUT_SIZE, h / INPUT_SIZE
        detections = [[float(b[0] * sx), float(b[1] * sy),
                       float(b[2] * sx), float(b[3] * sy),
                       float(s), int(c)]
                      for b, s, c in zip(outputs["boxes"].cpu().numpy(),
                                         outputs["scores"].cpu().numpy(),
                                         outputs["labels"].cpu().numpy(),
                                         strict=True)]

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        overlay = overlay_saliency(bgr, saliency)
        overlay = draw_detections(
            overlay, detections, self.cfg["detection"]["score_threshold"])

        return {"depth": depth, "depth_source": depth_source,
                "saliency": saliency, "saliency_source": sal_source,
                "detections": detections, "overlay_bgr": overlay,
                "fusion_mode": self.checkpoint.get("fusion_mode", "attention")}

    # ------------------------------------------------------------------
    def simulate(self, rgb: np.ndarray, result: dict, max_frames=12) -> np.ndarray:
        """Run the car simulation on this image's saliency field and render
        the final state (path, car, goal) over the image. Returns BGR."""
        window = tuple(self.cfg["simulation"]["window"])
        sal_win = cv2.resize(result["saliency"], window)
        dets_win = scale_boxes(result["detections"],
                               (rgb.shape[1], rgb.shape[0]), window)
        score_thr = self.cfg["detection"]["score_threshold"]
        dets_win = [d for d in dets_win if d[4] >= score_thr]

        sim = CarSimulator(cfg=self.cfg)
        for _ in range(max_frames):
            sim.update(sal_win, dets_win)
            if sim.goal_reached():
                break

        bgr = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), window)
        frame = overlay_saliency(bgr, sal_win)
        frame = draw_detections(frame, dets_win, score_thr)
        return draw_car_state(frame, sim.state(), sim.goal)
