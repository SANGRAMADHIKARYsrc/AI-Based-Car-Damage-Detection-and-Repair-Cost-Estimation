"""Phase 4 — Baseline Detector Evaluation Script

Evaluates trained YOLOv8 model on validation and test sets.
Generates: per-class metrics, confusion matrix, sample predictions,
speed benchmarks, and comprehensive evaluation report.

Usage:
    python src/detection/evaluate_baseline.py
    python src/detection/evaluate_baseline.py --weights models/detection/phase4_yolov8n_baseline/weights/best.pt
    python src/detection/evaluate_baseline.py --weights best.pt --split test
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


CLASS_NAMES = ['dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat']


def evaluate(args: argparse.Namespace) -> dict:
    """Run full evaluation suite on trained model."""
    from ultralytics import YOLO

    # Paths
    data_yaml = PROJECT_ROOT / "data" / "Datasets" / "data.yaml"
    eval_dir = PROJECT_ROOT / "evaluation" / "detection"
    eval_dir.mkdir(parents=True, exist_ok=True)

    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path

    if not weights_path.exists():
        # Try default location
        default = PROJECT_ROOT / "models" / "detection" / "phase4_yolov8n_baseline" / "weights" / "best.pt"
        if default.exists():
            weights_path = default
            print(f"Using default weights: {weights_path}")
        else:
            raise FileNotFoundError(f"Weights not found: {weights_path}")

    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    # === Validation Metrics ===
    print(f"\n=== Evaluating on {args.split} set ===")
    val_results = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        save_json=True,
        plots=True,
        verbose=True,
        project=str(eval_dir),
        name=f"phase4_{args.split}_eval",
        exist_ok=True,
    )

    # Extract overall metrics
    overall = {
        "precision": float(val_results.results_dict.get("metrics/precision(B)", 0)),
        "recall": float(val_results.results_dict.get("metrics/recall(B)", 0)),
        "mAP50": float(val_results.results_dict.get("metrics/mAP50(B)", 0)),
        "mAP50_95": float(val_results.results_dict.get("metrics/mAP50-95(B)", 0)),
    }
    overall["f1"] = round(
        2 * overall["precision"] * overall["recall"]
        / (overall["precision"] + overall["recall"] + 1e-8), 4
    )

    # Per-class metrics
    per_class = {}
    if hasattr(val_results, "box"):
        box = val_results.box
        for i, cls_name in enumerate(CLASS_NAMES):
            if i < len(box.p):
                p = float(box.p[i])
                r = float(box.r[i])
                f1 = 2 * p * r / (p + r + 1e-8)
                per_class[cls_name] = {
                    "precision": round(p, 4),
                    "recall": round(r, 4),
                    "f1": round(f1, 4),
                    "ap50": round(float(box.ap50[i]), 4) if i < len(box.ap50) else None,
                    "ap": round(float(box.ap[i]), 4) if i < len(box.ap) else None,
                }

    # === Inference Speed Benchmark ===
    print("\n=== Speed Benchmark ===")
    speed_results = benchmark_speed(model, args.imgsz, n_runs=50)

    # === Compile Report ===
    report = {
        "experiment": "phase4_baseline_detector",
        "model": str(weights_path.name),
        "split": args.split,
        "date": datetime.now().isoformat(),
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "overall_metrics": overall,
        "per_class_metrics": per_class,
        "speed": speed_results,
    }

    # Save JSON report
    report_path = eval_dir / f"phase4_{args.split}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")

    # Print summary table
    print_summary(overall, per_class, speed_results)

    return report


def benchmark_speed(model, imgsz: int, n_runs: int = 50) -> dict:
    """Benchmark inference speed on dummy input."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        model.predict(dummy, imgsz=imgsz, verbose=False, device=device)

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, imgsz=imgsz, verbose=False, device=device)
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "device": device,
        "imgsz": imgsz,
        "n_runs": n_runs,
        "mean_ms": round(np.mean(times), 2),
        "std_ms": round(np.std(times), 2),
        "median_ms": round(np.median(times), 2),
        "p95_ms": round(np.percentile(times, 95), 2),
        "fps": round(1000 / np.mean(times), 1),
    }


def print_summary(overall: dict, per_class: dict, speed: dict):
    """Print formatted evaluation summary."""
    print("\n" + "=" * 65)
    print("  PHASE 4 — BASELINE DETECTOR EVALUATION SUMMARY")
    print("=" * 65)

    print(f"\n  {'Metric':<20} {'Value':>10}")
    print(f"  {'-'*20} {'-'*10}")
    for k, v in overall.items():
        print(f"  {k:<20} {v:>10.4f}")

    if per_class:
        print(f"\n  {'Class':<18} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AP50':>7} {'AP':>7}")
        print(f"  {'-'*18} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for cls, m in per_class.items():
            ap50 = f"{m['ap50']:.4f}" if m.get('ap50') is not None else "  N/A"
            ap = f"{m['ap']:.4f}" if m.get('ap') is not None else "  N/A"
            print(f"  {cls:<18} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} {ap50:>7} {ap:>7}")

    print(f"\n  Speed: {speed['mean_ms']:.1f}ms ± {speed['std_ms']:.1f}ms ({speed['fps']} FPS) on {speed['device']}")
    print("=" * 65)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4: Evaluate Baseline Detector")
    parser.add_argument("--weights", type=str,
                        default="models/detection/phase4_yolov8n_baseline/weights/best.pt",
                        help="Path to trained weights")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
