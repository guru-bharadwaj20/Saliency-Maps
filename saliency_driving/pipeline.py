"""End-to-end inference on a single arbitrary RGB image.

RGB -> monocular depth (MiDaS) -> saliency (cmSalGAN or spectral-residual
fallback) -> optional fusion -> Faster R-CNN detections -> overlay, plus an
optional car-simulation render. Result dicts report the exact depth/saliency
source used. Backs both the Gradio demo and `main.py --infer`.
"""

import cv2
import numpy as np
import torch

from saliency_driving.config import get_device, load_config, resolve_path
from saliency_driving.datasets import INPUT_SIZE, normalize_imagenet
from saliency_driving.evaluate import load_models
from saliency_driving.log import get_logger
from saliency_driving.models import UnetGAN
from saliency_driving.saliency import heuristic, mono_depth
from saliency_driving.simulation.simulator import CarSimulator
from saliency_driving.simulation.visualize import (
    draw_car_state,
    draw_detections,
    overlay_saliency,
    scale_boxes,
)

log = get_logger(__name__)


class SingleImagePipeline:
    def __init__(self, model_path=None, use_mono_depth=True):
        self.cfg = load_config()
        self.device = get_device(self.cfg)
        self.use_mono_depth = use_mono_depth and mono_depth.available()
        self.detector, self.fusion, self.checkpoint, self.saliency_model = \
            load_models(self.cfg, self.device, model_path)
        sal_ckpt = resolve_path("cmsalgan_ckpt", self.cfg)
        if self.saliency_model is None and sal_ckpt.exists():
            self.saliency_model = UnetGAN().to(self.device)
            self.saliency_model.load_state_dict(torch.load(sal_ckpt, map_location=self.device))
            self.saliency_model.eval()
        log.info("Pipeline ready: fusion=%s, cmSalGAN=%s, mono_depth=%s, device=%s",
                 self.checkpoint.get("fusion_mode", "attention"),
                 self.saliency_model is not None, self.use_mono_depth, self.device)

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        if resized.ndim == 2:
            resized = resized[..., None]
        t = torch.from_numpy(resized.astype(np.float32) / 255.0)
        return t.permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _estimate_saliency(self, rgb: np.ndarray, depth: np.ndarray):
        if self.saliency_model is not None and depth is not None:
            rgb_t = normalize_imagenet(self._to_tensor(rgb))
            depth_t = self._to_tensor((depth * 255).astype(np.uint8)).repeat(1, 3, 1, 1)
            with torch.no_grad():
                sal = self.saliency_model([rgb_t, depth_t], mode=2)
            sal = torch.clamp(sal, 0, 1).squeeze().cpu().numpy()
            sal = cv2.resize(sal, (rgb.shape[1], rgb.shape[0]))
            return sal.astype(np.float32), "cmSalGAN (RGB + MiDaS depth)"
        return (heuristic.spectral_residual_saliency(rgb),
                "spectral residual (heuristic fallback — no cmSalGAN checkpoint)")

    def run(self, rgb: np.ndarray) -> dict:
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
        overlay = draw_detections(overlay, detections, self.cfg["detection"]["score_threshold"])

        return {"depth": depth, "depth_source": depth_source,
                "saliency": saliency, "saliency_source": sal_source,
                "detections": detections, "overlay_bgr": overlay,
                "fusion_mode": self.checkpoint.get("fusion_mode", "attention")}

    def simulate(self, rgb: np.ndarray, result: dict, max_frames=12) -> np.ndarray:
        window = tuple(self.cfg["simulation"]["window"])
        sal_win = cv2.resize(result["saliency"], window)
        dets_win = scale_boxes(result["detections"], (rgb.shape[1], rgb.shape[0]), window)
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
