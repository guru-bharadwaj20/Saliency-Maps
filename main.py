#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import argparse
import subprocess
import traceback
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SAL_DIR = RESULTS_DIR / "saliency_maps"
DET_RESULTS = RESULTS_DIR / "detection_outputs"
SIM_DIR = RESULTS_DIR / "simulation_outputs"

def safe_import(module_path: str, attr: str = "main"):
    try:
        mod = __import__(module_path, fromlist=[attr])
        fn = getattr(mod, attr)
        return fn
    except Exception:
        return None

def run_saliency(dataset=None):
    fn = safe_import("saliency_generation.generate_saliency", "main")
    if fn is None:
        script = ROOT / "saliency_generation" / "generate_saliency.py"
        if script.exists():
            cmd = [sys.executable, str(script)]
            if dataset:
                os.environ["DATASETS"] = dataset
            subprocess.check_call(cmd)
            return
        raise RuntimeError("generate_saliency.main not found.")
    fn()

def run_train_detector():
    fn = safe_import("detection.train_detector", "train")
    if fn is None:
        script = ROOT / "detection" / "train_detector.py"
        if script.exists():
            subprocess.check_call([sys.executable, str(script)])
            return
        raise RuntimeError("train_detector.train not found.")
    fn()

def run_test_detector():
    fn = safe_import("detection.test_detector", "run_inference")
    if fn is None:
        script = ROOT / "detection" / "test_detector.py"
        if script.exists():
            subprocess.check_call([sys.executable, str(script)])
            return
        raise RuntimeError("test_detector.run_inference not found.")
    return fn()

def run_visualize_as_simulation(dataset, saliency_dir=None, detections_json=None, results_dir=None, path_trace=None, goal=None, car_heading=None):
    fn = safe_import("simulation.visualize", "visualize_sequence")
    if fn is None:
        script = ROOT / "simulation" / "visualize.py"
        if script.exists():
            subprocess.check_call([sys.executable, str(script)])
            return
        raise RuntimeError("visualize_sequence not found.")
    sal_dir = saliency_dir or (SAL_DIR / dataset)
    res_dir = results_dir or SIM_DIR
    fn(dataset=dataset,
       saliency_dir=str(sal_dir),
       detections_json=str(detections_json) if detections_json else "",
       results_dir=str(res_dir),
       path_trace=path_trace or [],
       goal=goal or (50, 50),
       car_heading=car_heading or (1.0, 0.0))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--saliency", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--dataset", default="NJUD")
    p.add_argument("--salmap", default=None)
    p.add_argument("--detections_json", default=None)
    p.add_argument("--save_video", default=None)
    p.add_argument("--auto_sim", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    try:
        if args.all:
            run_saliency(dataset=args.dataset)
            run_train_detector()
            _ = run_test_detector()
            detections_json = args.detections_json or (DET_RESULTS / "preds.json")
            run_visualize_as_simulation(dataset=args.dataset,
                                        saliency_dir=SAL_DIR / args.dataset,
                                        detections_json=str(detections_json),
                                        results_dir=SIM_DIR,
                                        path_trace=[], goal=(550, 80), car_heading=(1.0, 0.0))
            return

        if args.saliency:
            run_saliency(dataset=args.dataset)

        if args.train:
            run_train_detector()

        if args.test:
            _ = run_test_detector()

        if args.simulate or args.visualize:
            detections_json = args.detections_json or (DET_RESULTS / "preds.json")
            run_visualize_as_simulation(dataset=args.dataset,
                                        saliency_dir=SAL_DIR / args.dataset,
                                        detections_json=str(detections_json),
                                        results_dir=SIM_DIR,
                                        path_trace=[], goal=(550, 80), car_heading=(1.0, 0.0))

        if not (args.saliency or args.train or args.test or args.simulate or args.visualize or args.all):
            print("No action requested.")

    except Exception:
        print("[main] ERROR during pipeline execution:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
