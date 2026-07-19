"""Ablation study: measure what saliency fusion actually buys the detector.

Trains one detector per variant, evaluates all of them on the test split
with real metrics, and writes a markdown + JSON report to results/ablation/.
The variant with the best AP@0.5 is copied to models/detector_best.pt so the
normal pipeline (--test / --simulate) uses it afterwards.

Variants (each trains the same Faster R-CNN, seeded identically):
    baseline_no_fusion  — raw RGB, no fusion module (control)
    fusion_plain        — fusion without attention, oracle (GT) saliency
    fusion_attention    — full fusion with attention, oracle (GT) saliency

"Oracle" = the GT mask is used as the saliency signal. This isolates the
value of the fusion architecture from cmSalGAN's prediction quality. When
models/cmSalGAN.ckpt is available, rerun with --saliency-source precomputed
for the realistic end-to-end numbers.

Usage:
    python scripts/run_ablation.py
    python scripts/run_ablation.py --epochs 3 --saliency-source precomputed
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import dataset_name, get_logger, load_config, resolve_path

log = get_logger(__name__)


def variants_for(saliency_source):
    return [
        {"name": "baseline_no_fusion", "fusion_mode": "none", "saliency_source": "zero"},
        {"name": "fusion_plain", "fusion_mode": "plain",
         "saliency_source": saliency_source},
        {"name": "fusion_attention", "fusion_mode": "attention",
         "saliency_source": saliency_source},
    ]


def run(epochs=None, saliency_source="gt", extra_precomputed_eval=True):
    from detection.annotations import load_gt_boxes_for_split
    from detection.metrics import evaluate_detections
    from detection.test_detector import run_inference
    from detection.train_detector import train

    cfg = load_config()
    ds_name = dataset_name(cfg)
    out_dir = resolve_path("results", cfg) / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = resolve_path("checkpoints", cfg)
    gt_boxes = load_gt_boxes_for_split(str(resolve_path("data", cfg) / ds_name / "test"))
    if not gt_boxes:
        raise RuntimeError("No GT boxes available for evaluation — check the test split.")

    rows = []
    start = time.time()
    for variant in variants_for(saliency_source):
        name = variant["name"]
        log.info("=== Variant: %s (fusion=%s, saliency=%s) ===",
                 name, variant["fusion_mode"], variant["saliency_source"])
        ckpt_path = ckpt_dir / f"ablation_{name}.pt"
        summary = train(fusion_mode=variant["fusion_mode"],
                        saliency_source=variant["saliency_source"],
                        epochs=epochs, out_path=str(ckpt_path))
        preds, info, timing = run_inference(model_path=str(ckpt_path), save_images=False)
        with open(out_dir / f"{name}_detections.json", "w") as f:
            json.dump(preds, f)
        det = evaluate_detections(preds, gt_boxes,
                                  iou_threshold=cfg["detection"]["iou_threshold"],
                                  score_threshold=cfg["detection"]["score_threshold"])
        rows.append({"variant": name, **variant, "train_loss": summary["best_loss"],
                     "train_time_s": summary["train_time_s"],
                     "AP@0.5": det["AP@0.5"], "precision": det["precision"],
                     "recall": det["recall"], "F1": det["F1"],
                     "inference_ms": timing["avg_inference_ms"],
                     "checkpoint": str(ckpt_path)})
        log.info("%s: AP@0.5=%.4f F1=%.4f", name, det["AP@0.5"], det["F1"])

    # Extra row: the attention model trained on oracle saliency, evaluated on
    # the realistic precomputed cmSalGAN maps (distribution shift check).
    if extra_precomputed_eval and saliency_source == "gt":
        ckpt_path = ckpt_dir / "ablation_fusion_attention.pt"
        preds, _, timing = run_inference(model_path=str(ckpt_path),
                                         saliency_source="precomputed", save_images=False)
        det = evaluate_detections(preds, gt_boxes,
                                  iou_threshold=cfg["detection"]["iou_threshold"],
                                  score_threshold=cfg["detection"]["score_threshold"])
        rows.append({"variant": "fusion_attention (test w/ cmSalGAN maps)",
                     "fusion_mode": "attention", "saliency_source": "gt→precomputed",
                     "train_loss": rows[-1]["train_loss"],
                     "train_time_s": rows[-1]["train_time_s"],
                     "AP@0.5": det["AP@0.5"], "precision": det["precision"],
                     "recall": det["recall"], "F1": det["F1"],
                     "inference_ms": timing["avg_inference_ms"],
                     "checkpoint": str(ckpt_path)})
        log.info("attention w/ cmSalGAN maps: AP@0.5=%.4f F1=%.4f", det["AP@0.5"], det["F1"])

    # Promote the best trainable variant to the standard best-model path.
    trained_rows = [r for r in rows if "→" not in str(r["saliency_source"])]
    best = max(trained_rows, key=lambda r: r["AP@0.5"])
    shutil.copy(best["checkpoint"], resolve_path("detector_best", cfg))
    log.info("Best variant '%s' (AP@0.5=%.4f) copied to %s",
             best["variant"], best["AP@0.5"], resolve_path("detector_best", cfg))

    report = {"dataset": ds_name, "epochs": epochs or cfg["training"]["epochs"],
              "seed": cfg["seed"], "device": rows and "cpu/gpu per config",
              "total_time_s": time.time() - start, "rows": rows, "best": best["variant"]}
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    md = ["# Ablation results", "",
          f"Dataset: {ds_name} (sample), epochs: {report['epochs']}, seed: {report['seed']}", "",
          "| Variant | Fusion | Saliency | Train loss | AP@0.5 | Precision | Recall "
          "| F1 | Inference (ms) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['variant']} | {r['fusion_mode']} | {r['saliency_source']} "
                  f"| {r['train_loss']:.4f} | {r['AP@0.5']:.4f} | {r['precision']:.4f} "
                  f"| {r['recall']:.4f} | {r['F1']:.4f} | {r['inference_ms']:.0f} |")
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("Report written to %s", out_dir / "report.md")
    print("\n".join(md))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--saliency-source", choices=["gt", "precomputed"], default="gt",
                        help="Saliency signal for the fusion variants (default: gt oracle)")
    args = parser.parse_args()
    run(epochs=args.epochs, saliency_source=args.saliency_source)
