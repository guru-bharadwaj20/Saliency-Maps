"""Potential-field steering: goal attraction, saliency and detection repulsion."""

import cv2
import numpy as np

SAL_THRESHOLD = 0.35
AVOID_RADIUS = 60
GOAL_WEIGHT = 1.0
SAL_WEIGHT = 2.2
DET_WEIGHT = 3.5


def compute_saliency_gradient(saliency_map, pos, radius=AVOID_RADIUS):
    """Unit repulsion vector from the local saliency gradient around pos."""
    H, W = saliency_map.shape
    x, y = int(pos[0]), int(pos[1])
    x0, x1 = max(0, x - radius), min(W, x + radius)
    y0, y1 = max(0, y - radius), min(H, y + radius)
    patch = saliency_map[y0:y1, x0:x1]
    if patch.size == 0:
        return np.zeros(2, dtype=np.float32)

    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
    gx /= grad_mag
    gy /= grad_mag

    weights = patch * (patch > SAL_THRESHOLD)
    vec = np.array([-np.sum(gx * weights), -np.sum(gy * weights)], dtype=np.float32)
    norm = np.linalg.norm(vec) + 1e-9
    if norm > 0:
        vec /= norm
    return vec


def compute_detection_repulsion(detections, pos, radius=AVOID_RADIUS):
    """Unit repulsion vector from nearby detection-box centers."""
    total_vec = np.zeros(2, dtype=np.float32)
    for det in detections:
        x1, y1, x2, y2 = det[:4]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        diff = pos - np.array([cx, cy])
        dist = np.linalg.norm(diff) + 1e-6
        if 1 < dist < radius:
            total_vec += (diff / dist) * (1.0 / dist)
    norm = np.linalg.norm(total_vec) + 1e-9
    if norm > 0:
        total_vec /= norm
    return total_vec


def compute_goal_vector(pos, goal):
    vec = np.array(goal) - np.array(pos)
    return vec / (np.linalg.norm(vec) + 1e-9)


def obstacle_avoidance_control(pos, goal, saliency_map, detections=None,
                               goal_weight=GOAL_WEIGHT, sal_weight=SAL_WEIGHT,
                               det_weight=DET_WEIGHT, radius=AVOID_RADIUS):
    """Combine goal attraction and repulsion terms into a unit steering vector."""
    goal_vec = compute_goal_vector(pos, goal)
    sal_vec = compute_saliency_gradient(saliency_map, pos, radius=radius)
    det_vec = compute_detection_repulsion(detections or [], pos, radius=radius)
    steer = goal_weight * goal_vec + sal_weight * sal_vec + det_weight * det_vec
    return steer / (np.linalg.norm(steer) + 1e-9)


def run_step(pos, goal, saliency_map, detections=None, step_size=5):
    steer = obstacle_avoidance_control(pos, goal, saliency_map, detections)
    new_pos = np.array(pos) + steer * step_size
    H, W = saliency_map.shape
    new_pos[0] = np.clip(new_pos[0], 0, W - 1)
    new_pos[1] = np.clip(new_pos[1], 0, H - 1)
    return new_pos, steer
