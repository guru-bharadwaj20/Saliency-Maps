"""
simulation/obstacle_avoidance.py

Implements obstacle avoidance logic using saliency maps and detection boxes.
Used by car_simulator.py or any autonomous navigation agent in this project.

Author: Guru R Bharadwaj
Project: Saliency-Aware Autonomous Car
"""

import numpy as np
import cv2

# Constants — tweak for sensitivity
SAL_THRESHOLD = 0.35    # pixels above this saliency considered risky
AVOID_RADIUS = 60       # influence radius for obstacle repulsion
GOAL_WEIGHT = 1.0       # weight for goal attraction
SAL_WEIGHT = 2.2        # weight for saliency avoidance
DET_WEIGHT = 3.5        # weight for detection avoidance


def compute_saliency_gradient(saliency_map, pos, radius=AVOID_RADIUS):
    """
    Compute a 2D avoidance vector based on local saliency gradient around the car position.
    The higher the local saliency, the stronger the repulsion.
    """
    H, W = saliency_map.shape
    x, y = int(pos[0]), int(pos[1])
    x0, x1 = max(0, x - radius), min(W, x + radius)
    y0, y1 = max(0, y - radius), min(H, y + radius)
    patch = saliency_map[y0:y1, x0:x1]
    if patch.size == 0:
        return np.zeros(2, dtype=np.float32)

    # Compute gradients (using Sobel filters)
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-6

    # Normalize gradients
    gx /= grad_mag
    gy /= grad_mag

    # Weighted by saliency intensity
    weights = patch * (patch > SAL_THRESHOLD)
    vx = -np.sum(gx * weights)
    vy = -np.sum(gy * weights)
    vec = np.array([vx, vy], dtype=np.float32)

    norm = np.linalg.norm(vec) + 1e-9
    if norm > 0:
        vec /= norm
    return vec


def compute_detection_repulsion(detections, pos, radius=AVOID_RADIUS):
    """
    Computes repulsive force from detection boxes near the current position.
    detections: list of [x1, y1, x2, y2, score, cls]
    """
    total_vec = np.zeros(2, dtype=np.float32)
    for det in detections:
        x1, y1, x2, y2 = det[:4]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        diff = pos - np.array([cx, cy])
        dist = np.linalg.norm(diff) + 1e-6
        if dist < radius and dist > 1:
            # stronger repulsion when closer
            repulse = (diff / dist) * (1.0 / dist)
            total_vec += repulse
    norm = np.linalg.norm(total_vec) + 1e-9
    if norm > 0:
        total_vec /= norm
    return total_vec


def compute_goal_vector(pos, goal):
    """Compute normalized vector from current pos → goal."""
    vec = np.array(goal) - np.array(pos)
    norm = np.linalg.norm(vec) + 1e-9
    return vec / norm


def obstacle_avoidance_control(pos, goal, saliency_map, detections=None):
    """
    Main control function for obstacle avoidance.

    Args:
        pos: current car position (x, y)
        goal: goal position (x, y)
        saliency_map: 2D np.ndarray normalized [0, 1]
        detections: list of detection boxes [x1, y1, x2, y2, score, cls]

    Returns:
        steering_vector: 2D unit vector indicating movement direction
    """
    # Attraction toward goal
    goal_vec = compute_goal_vector(pos, goal)

    # Repulsion from saliency
    sal_vec = compute_saliency_gradient(saliency_map, pos, radius=AVOID_RADIUS)

    # Repulsion from detection boxes
    det_vec = compute_detection_repulsion(detections or [], pos, radius=AVOID_RADIUS)

    # Combine forces
    steer = GOAL_WEIGHT * goal_vec + SAL_WEIGHT * sal_vec + DET_WEIGHT * det_vec

    # Normalize
    norm = np.linalg.norm(steer) + 1e-9
    steer /= norm

    return steer


def run_step(pos, goal, saliency_map, detections=None, step_size=5):
    """
    Computes the next position based on obstacle avoidance control.
    """
    steer = obstacle_avoidance_control(pos, goal, saliency_map, detections)
    new_pos = np.array(pos) + steer * step_size
    H, W = saliency_map.shape
    new_pos[0] = np.clip(new_pos[0], 0, W - 1)
    new_pos[1] = np.clip(new_pos[1], 0, H - 1)
    return new_pos, steer
