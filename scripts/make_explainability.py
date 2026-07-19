"""Explainability grid: input RGB, saliency, fusion attention, and EigenCAM of
the detector backbone (baseline vs. fusion+attention). Writes docs/explainability.png.
Run scripts/run_ablation.py first. Usage: python scripts/make_explainability.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saliency_driving import dataset_name, get_device, get_logger, load_config, resolve_path
from saliency_driving.datasets import INPUT_SIZE
from saliency_driving.evaluate import load_models

log = get_logger(__name__)

PANEL = 224
N_IMAGES = 3
HEADERS = ["Input RGB", "cmSalGAN saliency", "Fusion attention",
           "EigenCAM: baseline", "EigenCAM: fusion+attn"]


def eigencam(detector, det_input):
    """First principal component of the backbone's layer4 activations, in [0, 1]."""
    acts = {}

    def hook(_m, _i, out):
        acts["layer4"] = out.detach()

    handle = detector.backbone.body.layer4.register_forward_hook(hook)
    with torch.no_grad():
        detector(list(det_input))
    handle.remove()

    a = acts["layer4"][0].cpu().numpy()
    c, h, w = a.shape
    x = a.reshape(c, -1).T
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    proj = x @ vt[0]
    if abs(proj.min()) > abs(proj.max()):
        proj = -proj
    cam = np.maximum(proj.reshape(h, w), 0)
    if cam.max() > 1e-8:
        cam = cam / cam.max()
    return cam


def heat_panel(gray01, base_bgr=None):
    heat = cv2.applyColorMap((np.clip(gray01, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (PANEL, PANEL))
    if base_bgr is not None:
        heat = cv2.addWeighted(cv2.resize(base_bgr, (PANEL, PANEL)), 0.45, heat, 0.55, 0)
    return heat


def main():
    cfg = load_config()
    device = get_device(cfg)
    ds_name = dataset_name(cfg)
    ckpt_dir = resolve_path("checkpoints", cfg)
    data_dir = resolve_path("data", cfg) / ds_name / "test" / "RGB"
    sal_dir = resolve_path("saliency_maps", cfg) / ds_name

    baseline_path = ckpt_dir / "ablation_baseline_no_fusion.pt"
    fusion_path = ckpt_dir / "ablation_fusion_attention.pt"
    for p in (baseline_path, fusion_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — run scripts/run_ablation.py first.")

    det_base, _, _, _ = load_models(cfg, device, baseline_path)
    det_fuse, fusion, _, _ = load_models(cfg, device, fusion_path)

    rows = []
    for img_path in sorted(data_dir.glob("*.jpg"))[:N_IMAGES]:
        stem = img_path.stem
        bgr = cv2.imread(str(img_path))
        rgb_small = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (INPUT_SIZE, INPUT_SIZE))
        rgb_t = (torch.from_numpy(rgb_small.astype(np.float32) / 255.0)
                 .permute(2, 0, 1).unsqueeze(0).to(device))

        sal_img = cv2.imread(str(sal_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        sal01 = (cv2.resize(sal_img, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
                 if sal_img is not None else np.zeros((INPUT_SIZE, INPUT_SIZE), np.float32))
        sal_t = torch.from_numpy(sal01)[None, None].to(device)

        with torch.no_grad():
            fused_in = torch.cat((rgb_t, torch.clamp(sal_t, 0, 1)), dim=1)
            attn = fusion.attention(fused_in).squeeze().cpu().numpy()
            fused = fusion(rgb_t, sal_t)

        panels = [cv2.resize(bgr, (PANEL, PANEL)),
                  heat_panel(sal01),
                  heat_panel(attn, bgr),
                  heat_panel(eigencam(det_base, rgb_t), bgr),
                  heat_panel(eigencam(det_fuse, fused), bgr)]
        rows.append(np.hstack(panels))
        log.info("Processed %s (attention range %.3f-%.3f)", stem, attn.min(), attn.max())

    header = np.full((28, PANEL * len(HEADERS), 3), 245, np.uint8)
    for i, text in enumerate(HEADERS):
        cv2.putText(header, text, (i * PANEL + 8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    grid = np.vstack([header] + rows)

    out_path = ROOT / "docs" / "explainability.png"
    out_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    log.info("Figure written: %s (%d KB)", out_path, out_path.stat().st_size // 1024)


if __name__ == "__main__":
    main()
