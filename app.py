"""Gradio web demo — try the full pipeline on any image.

Upload a photo → monocular depth (MiDaS) → saliency → detections → a car
simulation steered by that saliency field, all rendered side by side.

Run locally:      python app.py            (needs: pip install gradio timm)
Hugging Face:     push this repo to a Space (SDK: gradio); see
                  docs/DEPLOY_HF_SPACES.md
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import get_logger

log = get_logger(__name__)

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from detection.pipeline import SingleImagePipeline
        _pipeline = SingleImagePipeline()
    return _pipeline


def predict(image: np.ndarray):
    """image: RGB uint8 from Gradio. Returns (depth, saliency, detections,
    simulation) images plus a provenance report."""
    if image is None:
        return None, None, None, None, "Upload an image first."
    pipe = get_pipeline()
    result = pipe.run(image)
    sim_frame = pipe.simulate(image, result)

    if result["depth"] is not None:
        from saliency_generation.mono_depth import colorize_depth
        depth_vis = cv2.cvtColor(colorize_depth(result["depth"]), cv2.COLOR_BGR2RGB)
    else:
        depth_vis = np.zeros_like(image)

    sal_vis = cv2.applyColorMap(
        (result["saliency"] * 255).astype(np.uint8), cv2.COLORMAP_JET)
    sal_vis = cv2.cvtColor(sal_vis, cv2.COLOR_BGR2RGB)

    overlay = cv2.cvtColor(result["overlay_bgr"], cv2.COLOR_BGR2RGB)
    sim_rgb = cv2.cvtColor(sim_frame, cv2.COLOR_BGR2RGB)

    n_conf = sum(1 for d in result["detections"]
                 if d[4] >= pipe.cfg["detection"]["score_threshold"])
    report = (f"**Depth:** {result['depth_source']}  \n"
              f"**Saliency:** {result['saliency_source']}  \n"
              f"**Detector:** Faster R-CNN (fusion mode: {result['fusion_mode']}) — "
              f"{n_conf} confident detection(s)  \n"
              f"**Simulation:** potential-field steering on the saliency map "
              f"(white dot = car, orange = path, magenta star = goal)")
    return depth_vis, sal_vis, overlay, sim_rgb, report


def build_demo():
    import gradio as gr

    with gr.Blocks(title="Saliency-Aware Driving Demo") as demo:
        gr.Markdown(
            "# 🚗 Saliency-Aware Autonomous Driving — Live Demo\n"
            "Upload **any** photo. The pipeline estimates depth (MiDaS), computes a "
            "saliency map, runs the trained detector, and steers a simulated car "
            "through the scene using the saliency field as a cost surface.")
        with gr.Row():
            inp = gr.Image(label="Input image", type="numpy")
            depth_out = gr.Image(label="Monocular depth (MiDaS)")
        with gr.Row():
            sal_out = gr.Image(label="Saliency map")
            det_out = gr.Image(label="Saliency overlay + detections")
        with gr.Row():
            sim_out = gr.Image(label="Car simulation on this scene")
            report = gr.Markdown()
        btn = gr.Button("Run pipeline", variant="primary")
        btn.click(predict, inputs=inp,
                  outputs=[depth_out, sal_out, det_out, sim_out, report])
        gr.Markdown(
            "Sources are labeled honestly: without the cmSalGAN checkpoint the "
            "saliency falls back to the classical spectral-residual method, and "
            "the report says so.")
    return demo


if __name__ == "__main__":
    build_demo().launch()
