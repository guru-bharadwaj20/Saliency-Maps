import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2
import numpy as np
import glob
import json
from pathlib import Path
from typing import List, Tuple
from simulation.visualize import visualize_sequence

WINDOW_W = 800
WINDOW_H = 600
CAR_RADIUS = 8
STEP_SIZE = 5.0
GOAL_RADIUS = 10
SAL_THRESHOLD = 0.35
AVOID_RADIUS = 60
W_GOAL = 1.0
W_AVOID = 2.5
SMOOTHING = 0.8

def load_saliency_map(path: str, target_size=(WINDOW_W, WINDOW_H)) -> np.ndarray:
    sal = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if sal is None:
        raise FileNotFoundError(f"Saliency map not found: {path}")
    sal = sal.astype(np.float32) / 255.0
    sal = cv2.resize(sal, target_size, interpolation=cv2.INTER_LINEAR)
    return sal

def load_detections_json(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def sample_local_gradient(sal_map: np.ndarray, pos: Tuple[float, float], radius: int = AVOID_RADIUS) -> np.ndarray:
    H, W = sal_map.shape
    x, y = int(pos[0]), int(pos[1])
    x0, x1 = max(0, x - radius), min(W, x + radius)
    y0, y1 = max(0, y - radius), min(H, y + radius)
    patch = sal_map[y0:y1, x0:x1]
    if patch.size == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    h, w = patch.shape
    xs = (np.arange(w) + x0) - x
    ys = (np.arange(h) + y0) - y
    xv, yv = np.meshgrid(xs, ys)
    grad_x = gx * -1.0
    grad_y = gy * -1.0
    mag = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-8
    dist = np.sqrt(xv ** 2 + yv ** 2) + 1e-8
    radial = np.clip(1.0 - (dist / radius), 0.0, 1.0)
    vx = np.sum(grad_x * mag * radial)
    vy = np.sum(grad_y * mag * radial)
    vec = np.array([vx, vy], dtype=np.float32)
    norm = np.linalg.norm(vec) + 1e-9
    if norm > 0:
        vec = vec / norm
    return vec

def bbox_to_center(box):
    x1, y1, x2, y2 = box[:4]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

class CarSimulator:
    def __init__(self, sal_map: np.ndarray, detections: List[List] = None, start: Tuple[int, int] = None, goal: Tuple[int, int] = None):
        self.sal = sal_map
        self.H, self.W = sal_map.shape
        self.detections = detections or []
        self.start = start or (int(self.W * 0.1), int(self.H * 0.9))
        self.goal = goal or (int(self.W * 0.9), int(self.H * 0.1))
        self.pos = np.array(self.start, dtype=np.float32)
        self.heading = np.array([1.0, 0.0], dtype=np.float32)
        self.path = [tuple(self.pos)]

    def step(self):
        vec_goal = np.array(self.goal) - self.pos
        gnorm = np.linalg.norm(vec_goal) + 1e-9
        vec_goal = vec_goal / gnorm
        avoid_vec = sample_local_gradient(self.sal, (self.pos[0], self.pos[1]), radius=AVOID_RADIUS)
        for d in self.detections:
            cx, cy = bbox_to_center(d)
            diff = self.pos - np.array([cx, cy])
            dist = np.linalg.norm(diff) + 1e-9
            if dist < AVOID_RADIUS and dist > 1e-6:
                avoid_vec += (diff / dist) * (1.0 / dist) * 5.0
        steer = W_GOAL * vec_goal + W_AVOID * avoid_vec
        if np.linalg.norm(steer) > 1e-6:
            steer = steer / (np.linalg.norm(steer) + 1e-9)
        self.heading = (SMOOTHING * self.heading + (1 - SMOOTHING) * steer)
        if np.linalg.norm(self.heading) > 1e-6:
            self.heading = self.heading / (np.linalg.norm(self.heading) + 1e-9)
        self.pos += self.heading * STEP_SIZE
        self.pos[0] = np.clip(self.pos[0], 0, self.W - 1)
        self.pos[1] = np.clip(self.pos[1], 0, self.H - 1)
        self.path.append(tuple(self.pos))

    def is_goal_reached(self):
        return np.linalg.norm(self.pos - np.array(self.goal)) <= GOAL_RADIUS

def run_car_simulation(dataset="NJUD"):
    print(f"\n[CarSimulator] 🚗 Running headless simulation for dataset: {dataset}")
    saliency_dir = f"./results/saliency_maps/{dataset}"
    detections_json = f"./results/detection_outputs/{dataset}_detections.json"
    results_dir = f"./results/simulation_outputs/{dataset}"
    os.makedirs(results_dir, exist_ok=True)
    if not os.path.exists(saliency_dir):
        raise FileNotFoundError(f"Missing directory: {saliency_dir}")
    dets = load_detections_json(detections_json)
    sal_files = sorted(glob.glob(os.path.join(saliency_dir, "*.png")))
    if len(sal_files) == 0:
        raise RuntimeError(f"No saliency maps found in {saliency_dir}")
    path_trace = [(100 + i * 10, 300 - i * 2) for i in range(50)]
    goal = (550, 100)
    car_heading = np.array([1.0, -0.3])
    visualize_sequence(
        dataset=dataset,
        saliency_dir=saliency_dir,
        detections_json=detections_json,
        results_dir=results_dir,
        path_trace=path_trace,
        goal=goal,
        car_heading=car_heading
    )
    frame_files = glob.glob(os.path.join(results_dir, "*.png"))
    for f in frame_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"[CarSimulator] ⚠️ Error deleting {f}: {e}")
    print(f"[CarSimulator] ✅ Simulation complete for {dataset}")
    print(f"[CarSimulator] 🎬 Video saved at: {results_dir}/{dataset}_simulation.mp4")

if __name__ == "__main__":
    run_car_simulation("NJUD")
