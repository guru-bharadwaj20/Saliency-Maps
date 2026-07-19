"""Gradio web demo: upload any image and see depth, saliency, detections, and a
car simulation of the scene. Run `python app.py`; deploy per docs/DEPLOY_HF_SPACES.md."""

import cv2
import numpy as np

from saliency_driving import get_logger

log = get_logger(__name__)

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from saliency_driving.pipeline import SingleImagePipeline
        _pipeline = SingleImagePipeline()
    return _pipeline


def predict(image: np.ndarray):
    if image is None:
        return None, None, None, None, "Upload an image first."
    pipe = get_pipeline()
    result = pipe.run(image)
    sim_frame = pipe.simulate(image, result)

    if result["depth"] is not None:
        from saliency_driving.saliency.mono_depth import colorize_depth
        depth_vis = cv2.cvtColor(colorize_depth(result["depth"]), cv2.COLOR_BGR2RGB)
    else:
        depth_vis = np.zeros_like(image)

    sal_vis = cv2.applyColorMap((result["saliency"] * 255).astype(np.uint8), cv2.COLORMAP_JET)
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
            "# Saliency-Aware Autonomous Driving — Live Demo\n"
            "Upload any photo. The pipeline estimates depth (MiDaS), computes a "
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
        gr.Button("Run pipeline", variant="primary").click(
            predict, inputs=inp, outputs=[depth_out, sal_out, det_out, sim_out, report])
    return demo


if __name__ == "__main__":
    build_demo().launch()
