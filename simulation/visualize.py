import os
import cv2
import json
import numpy as np
from typing import List, Tuple

HEATMAP_ALPHA = 0.55
DETECTION_COLOR = (0, 255, 0)
UNKNOWN_COLOR = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_image(rgb_path: str, size: Tuple[int, int] = None):
    img = cv2.imread(rgb_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {rgb_path}")
    if size:
        img = cv2.resize(img, size)
    return img


def load_saliency(sal_path: str, size: Tuple[int, int] = None):
    sal = cv2.imread(sal_path, cv2.IMREAD_GRAYSCALE)
    if sal is None:
        raise FileNotFoundError(f"Saliency map not found: {sal_path}")
    sal = sal.astype(np.float32) / 255.0
    if size:
        sal = cv2.resize(sal, size)
    return sal


def overlay_saliency(rgb_img: np.ndarray, saliency: np.ndarray, alpha=HEATMAP_ALPHA):
    heat = cv2.applyColorMap((saliency * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(rgb_img, 1 - alpha, heat, alpha, 0)


def draw_detections(img: np.ndarray, detections: List[List]):
    for det in detections:
        x1, y1, x2, y2, score, cls = det
        color = DETECTION_COLOR if cls in [0, 1, 2, 3] else UNKNOWN_COLOR
        label = f"{int(cls)}:{score:.2f}"
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, label, (int(x1), int(y1 - 5)), FONT, 0.5, color, 1)
    return img


def visualize_frame(rgb_path: str, sal_path: str, detections: List[List],
                    save_path: str = None):

    rgb = load_image(rgb_path)
    sal = load_saliency(sal_path, size=(rgb.shape[1], rgb.shape[0]))

    frame = overlay_saliency(rgb, sal)
    frame = draw_detections(frame, detections)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, frame)

    return frame


def visualize_sequence(dataset: str, saliency_dir: str, detections_json: str,
                       results_dir: str, path_trace, goal, car_heading):

    os.makedirs(results_dir, exist_ok=True)

    detections = {}
    if os.path.exists(detections_json):
        with open(detections_json, 'r') as f:
            detections = json.load(f)

    frame_paths = sorted(os.listdir(saliency_dir))
    if len(frame_paths) == 0:
        raise RuntimeError(f"No saliency maps found in {saliency_dir}")

    print(f"🎞 Generating visualization for dataset: {dataset}")

    H, W = 480, 640
    out_path = os.path.join(results_dir, f"{dataset}_simulation.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(out_path, fourcc, 15, (W, H))

    for idx, sal_name in enumerate(frame_paths):

        sal_path = os.path.join(saliency_dir, sal_name)
        rgb_guess = os.path.join("data", dataset, "test", "RGB", sal_name.replace(".png", ".jpg"))

        rgb = load_image(rgb_guess, size=(W, H)) if os.path.exists(rgb_guess) else np.zeros((H, W, 3), np.uint8)
        sal = load_saliency(sal_path, size=(W, H))

        det_list = detections.get(sal_name.split('.')[0], [])

        frame = overlay_saliency(rgb, sal)
        frame = draw_detections(frame, det_list)

        writer.write(frame)

        save_bn = f"{dataset}_frame_{idx:03d}.png"
        print(f"✅ Frame {idx + 1}/{len(frame_paths)} -> {os.path.join(results_dir, save_bn)}")

        cv2.imwrite(os.path.join(results_dir, save_bn), frame)

    writer.release()
    print(f"🎬 Final video saved to: {out_path}")
