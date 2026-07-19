"""Saliency-driven car simulation over a stream of saliency frames."""

import glob
import json
import os

import cv2
import numpy as np

from saliency_driving.config import load_config, resolve_path, set_seed
from saliency_driving.log import get_logger
from saliency_driving.simulation.avoidance import obstacle_avoidance_control
from saliency_driving.simulation.visualize import scale_boxes, visualize_sequence

log = get_logger(__name__)


class CarSimulator:
    """Point-mass car steered by obstacle_avoidance_control in window space."""

    def __init__(self, start=None, goal=None, cfg=None):
        scfg = (cfg or load_config())["simulation"]
        self.W, self.H = scfg["window"]
        self.step_size = scfg["step_size"]
        self.steps_per_frame = scfg["steps_per_frame"]
        self.goal_radius = scfg["goal_radius"]
        self.smoothing = scfg["smoothing"]
        self.weights = {"goal_weight": scfg["goal_weight"],
                        "sal_weight": scfg["saliency_weight"],
                        "det_weight": scfg["detection_weight"]}
        self.start = start or (self.W * 0.08, self.H * 0.9)
        self.goal = goal or (self.W * 0.9, self.H * 0.12)
        self.pos = np.array(self.start, dtype=np.float32)
        self.heading = np.array([1.0, 0.0], dtype=np.float32)
        self.path = [tuple(self.pos)]

    def goal_reached(self):
        return np.linalg.norm(self.pos - np.array(self.goal)) <= self.goal_radius

    def update(self, sal_map, detections, steps=None):
        for _ in range(steps or self.steps_per_frame):
            if self.goal_reached():
                break
            steer = obstacle_avoidance_control(self.pos, self.goal, sal_map,
                                               detections, **self.weights)
            self.heading = self.smoothing * self.heading + (1 - self.smoothing) * steer
            norm = np.linalg.norm(self.heading)
            if norm > 1e-6:
                self.heading = self.heading / norm
            self.pos = self.pos + self.heading * self.step_size
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
    """Simulate over every saliency frame and render the sequence video."""
    cfg = load_config()
    set_seed(cfg["seed"])
    window = tuple(cfg["simulation"]["window"])
    log.info("Running simulation for dataset: %s", dataset)
    saliency_dir = resolve_path("saliency_maps", cfg) / dataset
    detections_json = detections_json or str(
        resolve_path("detection_outputs", cfg) / f"{dataset}_detections.json")
    results_dir = resolve_path("simulation_outputs", cfg) / dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    sal_files = sorted(glob.glob(os.path.join(str(saliency_dir), "*.png")))
    if not sal_files:
        raise RuntimeError(
            f"No saliency maps found in {saliency_dir}. Run `python main.py --saliency` first.")

    detections = {}
    if os.path.exists(detections_json):
        with open(detections_json) as f:
            detections = json.load(f)
    else:
        log.warning("No detections at %s — steering on saliency only.", detections_json)

    sim = CarSimulator(goal=goal, cfg=cfg)
    car_states = []
    for sal_path in sal_files:
        stem = os.path.splitext(os.path.basename(sal_path))[0]
        sal = cv2.imread(sal_path, cv2.IMREAD_GRAYSCALE)
        if sal is None:
            continue
        orig_h, orig_w = sal.shape
        sal_win = cv2.resize(sal.astype(np.float32) / 255.0, window)
        dets_win = scale_boxes(detections.get(stem, []), (orig_w, orig_h), window)
        sim.update(sal_win, dets_win)
        car_states.append(sim.state())

    reached = car_states[-1]["reached"] if car_states else False
    log.info("%d frames simulated | path length: %d steps | goal reached: %s",
             len(car_states), len(sim.path), reached)

    out_path = visualize_sequence(
        dataset=dataset,
        saliency_dir=str(saliency_dir),
        detections_json=detections_json,
        results_dir=str(results_dir),
        car_states=car_states,
        goal=sim.goal,
    )
    log.info("Simulation video: %s", out_path)
    return car_states


if __name__ == "__main__":
    run_car_simulation("NJUD")
