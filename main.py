#!/usr/bin/env python3
"""Pipeline orchestrator.

Stages:
    --saliency   Generate saliency maps with cmSalGAN (train + test splits)
    --train      Train Faster R-CNN + SaliencyFusion on GT-derived boxes
    --test       Run inference, save detections JSON, print real metrics
    --simulate   Run the saliency-driven car simulation and render video
    --visualize  Render saliency + detection overlays only (no car)
    --all        Run every stage in order

Configuration lives in config.yaml; --dataset overrides the configured one.
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import get_logger, load_config, resolve_path, set_seed

log = get_logger(__name__)


def detections_path(dataset, override=None):
    if override:
        return str(override)
    return str(resolve_path("detection_outputs") / f"{dataset}_detections.json")


def run_saliency(dataset):
    from saliency_generation.generate_saliency import main as saliency_main
    saliency_main(dataset=dataset)


def run_train_detector(dataset):
    os.environ["DATASETS"] = dataset
    from detection.train_detector import train
    train()


def run_test_detector(dataset):
    os.environ["DATASETS"] = dataset
    from detection.test_detector import evaluate, run_inference
    preds, info, timing = run_inference()
    log.info("Best checkpoint: epoch %s | train loss %.4f", info["epoch"], info["avg_loss"])
    print(evaluate(preds, timing))
    return preds


def run_simulation(dataset, detections_json=None):
    from simulation.car_simulator import run_car_simulation
    run_car_simulation(dataset=dataset,
                       detections_json=detections_path(dataset, detections_json))


def run_visualize(dataset, detections_json=None):
    from simulation.visualize import visualize_sequence
    visualize_sequence(dataset=dataset,
                       saliency_dir=str(resolve_path("saliency_maps") / dataset),
                       detections_json=detections_path(dataset, detections_json),
                       results_dir=str(resolve_path("simulation_outputs") / dataset))


def parse_args():
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--saliency", action="store_true", help="Generate saliency maps")
    p.add_argument("--train", action="store_true", help="Train the detector + fusion")
    p.add_argument("--test", action="store_true", help="Test detector and print metrics")
    p.add_argument("--simulate", action="store_true", help="Run car simulation + video")
    p.add_argument("--visualize", action="store_true", help="Render overlays without car")
    p.add_argument("--all", action="store_true", help="Run all stages in order")
    p.add_argument("--dataset", default=cfg["dataset"],
                   help=f"Dataset name (default from config.yaml: {cfg['dataset']})")
    p.add_argument("--detections_json", default=None,
                   help="Override path to detections JSON for simulate/visualize")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed()
    ran_any = False
    try:
        if args.all or args.saliency:
            run_saliency(args.dataset)
            ran_any = True
        if args.all or args.train:
            run_train_detector(args.dataset)
            ran_any = True
        if args.all or args.test:
            run_test_detector(args.dataset)
            ran_any = True
        if args.all or args.simulate:
            run_simulation(args.dataset, args.detections_json)
            ran_any = True
        if args.visualize and not (args.all or args.simulate):
            run_visualize(args.dataset, args.detections_json)
            ran_any = True

        if not ran_any:
            print("No action requested. Use --saliency / --train / --test / "
                  "--simulate / --visualize / --all (see --help).")
    except Exception:
        log.error("ERROR during pipeline execution:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
