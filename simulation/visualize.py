import json
import os
import sys

import cv2
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common import get_logger, load_config

log = get_logger(__name__)

WINDOW_W, WINDOW_H = load_config()["simulation"]["window"]
SCORE_THRESHOLD = load_config()["detection"]["score_threshold"]
HEATMAP_ALPHA = 0.45
DETECTION_COLOR = (0, 255, 0)
CAR_COLOR = (255, 255, 255)
PATH_COLOR = (0, 200, 255)
GOAL_COLOR = (255, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_image(rgb_path: str, size: tuple[int, int] = None):
    img = cv2.imread(rgb_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {rgb_path}")
    if size:
        img = cv2.resize(img, size)
    return img


def load_saliency(sal_path: str, size: tuple[int, int] = None):
    sal = cv2.imread(sal_path, cv2.IMREAD_GRAYSCALE)
    if sal is None:
        raise FileNotFoundError(f"Saliency map not found: {sal_path}")
    orig_shape = sal.shape  # (h, w) at native resolution
    sal = sal.astype(np.float32) / 255.0
    if size:
        sal = cv2.resize(sal, size)
    return sal, orig_shape


def scale_boxes(detections: list[list], src_size: tuple[int, int],
                dst_size: tuple[int, int]):
    """Scale [x1,y1,x2,y2,score,cls] boxes from src (w, h) to dst (w, h)."""
    sx = dst_size[0] / src_size[0]
    sy = dst_size[1] / src_size[1]
    return [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy, *b[4:]] for b in detections]


def overlay_saliency(rgb_img: np.ndarray, saliency: np.ndarray, alpha=HEATMAP_ALPHA):
    heat = cv2.applyColorMap((saliency * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(rgb_img, 1 - alpha, heat, alpha, 0)


def draw_detections(img: np.ndarray, detections: list[list], score_threshold=SCORE_THRESHOLD):
    for det in detections:
        x1, y1, x2, y2, score = det[:5]
        if score < score_threshold:
            continue
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), DETECTION_COLOR, 2)
        cv2.putText(img, f"{score:.2f}", (int(x1), max(int(y1) - 5, 12)),
                    FONT, 0.5, DETECTION_COLOR, 1)
    return img


def draw_car_state(img: np.ndarray, state: dict, goal: tuple[float, float]):
    """Draw path trace, car, heading arrow, and goal marker for one frame."""
    path = state.get("path", [])
    if len(path) > 1:
        pts = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=False, color=PATH_COLOR, thickness=2)

    gx, gy = int(goal[0]), int(goal[1])
    cv2.drawMarker(img, (gx, gy), GOAL_COLOR, cv2.MARKER_STAR, 18, 2)
    cv2.putText(img, "GOAL", (gx + 10, gy - 10), FONT, 0.5, GOAL_COLOR, 1)

    x, y = int(state["pos"][0]), int(state["pos"][1])
    cv2.circle(img, (x, y), 8, CAR_COLOR, -1)
    cv2.circle(img, (x, y), 8, (0, 0, 0), 2)
    hx, hy = state.get("heading", (1.0, 0.0))
    cv2.arrowedLine(img, (x, y), (int(x + hx * 25), int(y + hy * 25)),
                    CAR_COLOR, 2, tipLength=0.4)

    if state.get("reached"):
        cv2.putText(img, "GOAL REACHED", (WINDOW_W // 2 - 100, 40),
                    FONT, 0.9, (0, 255, 0), 2)
    return img


def visualize_sequence(dataset: str, saliency_dir: str, detections_json: str,
                       results_dir: str, car_states: list[dict] = None,
                       goal: tuple[float, float] = None, save_frames: bool = True):
    """Render saliency + detections (+ optional car simulation states) per frame
    and write an .avi video. car_states[i] corresponds to the i-th frame and
    holds window-space {pos, heading, path, reached}."""
    root = ROOT_DIR
    os.makedirs(results_dir, exist_ok=True)

    detections = {}
    if detections_json and os.path.exists(detections_json):
        with open(detections_json) as f:
            detections = json.load(f)
    else:
        log.warning("No detections file at %s — rendering without boxes.", detections_json)

    frame_names = sorted(f for f in os.listdir(saliency_dir) if f.endswith(".png"))
    if not frame_names:
        raise RuntimeError(f"No saliency maps found in {saliency_dir}")

    W, H = WINDOW_W, WINDOW_H
    out_path = os.path.join(results_dir, f"{dataset}_simulation.avi")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), 5, (W, H))

    log.info("Rendering %d frames for dataset: %s", len(frame_names), dataset)
    for idx, sal_name in enumerate(frame_names):
        stem = os.path.splitext(sal_name)[0]
        sal, orig_shape = load_saliency(os.path.join(saliency_dir, sal_name), size=(W, H))

        rgb_path = os.path.join(root, "data", dataset, "test", "RGB", stem + ".jpg")
        if os.path.exists(rgb_path):
            rgb = load_image(rgb_path, size=(W, H))
        else:
            rgb = np.zeros((H, W, 3), np.uint8)

        det_list = scale_boxes(detections.get(stem, []),
                               (orig_shape[1], orig_shape[0]), (W, H))

        frame = overlay_saliency(rgb, sal)
        frame = draw_detections(frame, det_list)
        if car_states is not None and idx < len(car_states):
            frame = draw_car_state(frame, car_states[idx], goal or (W * 0.9, H * 0.15))

        writer.write(frame)
        if save_frames:
            cv2.imwrite(os.path.join(results_dir, f"{dataset}_frame_{idx:03d}.png"), frame)

    writer.release()
    log.info("Video saved to: %s", out_path)
    return out_path
