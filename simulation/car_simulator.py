"""Saliency-driven car simulation.

Treats the test set's saliency maps as a stream of scene observations: for
each frame the car takes several control steps steered by
obstacle_avoidance (goal attraction + saliency-gradient repulsion +
detection-box repulsion), and the resulting state is rendered by
simulation.visualize.
"""

import sys, os
import glob
import json
import cv2
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from simulation.obstacle_avoidance import obstacle_avoidance_control
from simulation.visualize import visualize_sequence, scale_boxes, WINDOW_W, WINDOW_H

STEP_SIZE = 5.0
STEPS_PER_FRAME = 15
GOAL_RADIUS = 12
SMOOTHING = 0.8


class CarSimulator:
    """Point-mass car steered by obstacle_avoidance_control in window space."""

    def __init__(self, start=None, goal=None, window=(WINDOW_W, WINDOW_H)):
        self.W, self.H = window
        self.start = start or (self.W * 0.08, self.H * 0.9)
        self.goal = goal or (self.W * 0.9, self.H * 0.12)
        self.pos = np.array(self.start, dtype=np.float32)
        self.heading = np.array([1.0, 0.0], dtype=np.float32)
        self.path = [tuple(self.pos)]

    def goal_reached(self):
        return np.linalg.norm(self.pos - np.array(self.goal)) <= GOAL_RADIUS

    def update(self, sal_map, detections, steps=STEPS_PER_FRAME):
        """Advance the car through one observed frame."""
        for _ in range(steps):
            if self.goal_reached():
                break
            steer = obstacle_avoidance_control(self.pos, self.goal, sal_map, detections)
            self.heading = SMOOTHING * self.heading + (1 - SMOOTHING) * steer
            norm = np.linalg.norm(self.heading)
            if norm > 1e-6:
                self.heading = self.heading / norm
            self.pos = self.pos + self.heading * STEP_SIZE
            self.pos[0] = float(np.clip(self.pos[0], 0, self.W - 1))
            self.pos[1] = float(np.clip(self.pos[1], 0, self.H - 1))
            self.path.append((float(self.pos[0]), float(self.pos[1])))

    def state(self):
        return {
            "pos": (float(self.pos[0]), float(self.pos[1])),
            "heading": (float(self.heading[0]), float(self.heading[1])),
            "path": list(self.path),
            "reached": bool(self.goal_reached()),
        }


def run_car_simulation(dataset="NJUD", detections_json=None, goal=None):
    """Run the simulation over every saliency frame and render the video."""
    print(f"\n[CarSimulator] Running simulation for dataset: {dataset}")
    saliency_dir = os.path.join(ROOT_DIR, "results", "saliency_maps", dataset)
    detections_json = detections_json or os.path.join(
        ROOT_DIR, "results", "detection_outputs", f"{dataset}_detections.json")
    results_dir = os.path.join(ROOT_DIR, "results", "simulation_outputs", dataset)
    os.makedirs(results_dir, exist_ok=True)

    sal_files = sorted(glob.glob(os.path.join(saliency_dir, "*.png")))
    if not sal_files:
        raise RuntimeError(
            f"No saliency maps found in {saliency_dir}. Run `python main.py --saliency` first.")

    detections = {}
    if os.path.exists(detections_json):
        with open(detections_json, "r") as f:
            detections = json.load(f)
    else:
        print(f"[CarSimulator] No detections at {detections_json} — steering on saliency only.")

    sim = CarSimulator(goal=goal)
    car_states = []
    for sal_path in sal_files:
        stem = os.path.splitext(os.path.basename(sal_path))[0]
        sal = cv2.imread(sal_path, cv2.IMREAD_GRAYSCALE)
        if sal is None:
            continue
        orig_h, orig_w = sal.shape
        sal_win = cv2.resize(sal.astype(np.float32) / 255.0, (WINDOW_W, WINDOW_H))
        dets_win = scale_boxes(detections.get(stem, []),
                               (orig_w, orig_h), (WINDOW_W, WINDOW_H))
        sim.update(sal_win, dets_win)
        car_states.append(sim.state())

    reached = car_states[-1]["reached"] if car_states else False
    print(f"[CarSimulator] {len(car_states)} frames simulated | "
          f"path length: {len(sim.path)} steps | goal reached: {reached}")

    out_path = visualize_sequence(
        dataset=dataset,
        saliency_dir=saliency_dir,
        detections_json=detections_json,
        results_dir=results_dir,
        car_states=car_states,
        goal=sim.goal,
    )
    print(f"[CarSimulator] Simulation video: {out_path}")
    return car_states


if __name__ == "__main__":
    run_car_simulation("NJUD")
