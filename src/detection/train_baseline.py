"""Phase 4 — Baseline YOLOv8 Damage Detector Training Script

Trains YOLOv8n on CarDD 6-class damage detection dataset.
Handles: experiment config loading, class-weighted training, metric logging,
model checkpointing, and reproducibility metadata.

Usage:
    python src/detection/train_baseline.py
    python src/detection/train_baseline.py --config experiments/phase4_baseline_detector.yaml
    python src/detection/train_baseline.py --epochs 50 --batch 8 --imgsz 640
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def get_system_info() -> dict:
    """Collect hardware and environment metadata for reproducibility."""
    info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "os": platform.platform(),
        "timestamp": datetime.now().isoformat(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        info["cuda_version"] = torch.version.cuda
    return info


def load_experiment_config(config_path: Path) -> dict:
    """Load experiment configuration YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_dataset(data_yaml: Path) -> dict:
    """Validate dataset structure before training."""
    with open(data_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)

    base_path = Path(data_cfg["path"])
    stats = {}

    for split in ["train", "val", "test"]:
        if split in data_cfg:
            img_dir = base_path / data_cfg[split]
            lbl_dir = img_dir.parent / "labels"
            n_images = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0
            n_labels = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
            stats[split] = {"images": n_images, "labels": n_labels}

            if n_images == 0:
                raise FileNotFoundError(f"No images found in {img_dir}")
            if n_images != n_labels:
                print(f"WARNING: {split} image/label mismatch: {n_images} images, {n_labels} labels")

    print(f"Dataset validated: {stats}")
    return stats


def train(args: argparse.Namespace) -> dict:
    """Execute YOLOv8 baseline training."""
    from ultralytics import YOLO

    # Paths
    data_yaml = PROJECT_ROOT / "data" / "Datasets" / "data.yaml"
    weights_yaml = PROJECT_ROOT / "data" / "processed" / "class_weights.yaml"
    output_dir = PROJECT_ROOT / "models" / "detection"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load class weights from Phase 3
    with open(weights_yaml, "r") as f:
        weights_cfg = yaml.safe_load(f)
    class_weights = weights_cfg["recommended_loss_weights_tensor"]
    print(f"Class weights (smoothed sqrt): {class_weights}")

    # Validate dataset
    dataset_stats = validate_dataset(data_yaml)

    # Load experiment config if provided
    exp_config = {}
    if args.config:
        exp_config = load_experiment_config(Path(args.config))
        print(f"Loaded experiment config: {Path(args.config).name}")

    # Training hyperparameters — CLI args override config
    aug_cfg = exp_config.get("training", {}).get("augmentation", {})

    train_args = {
        # Core
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "seed": args.seed,

        # Optimizer
        "optimizer": "auto",
        "lr0": 0.01,
        "lrf": 0.01,        # final LR = lr0 * lrf
        "warmup_epochs": 3,
        "cos_lr": True,

        # Performance — critical for 8GB RAM / 4GB VRAM
        "amp": True,
        "workers": args.workers,
        "cache": False,       # don't cache images in RAM

        # Mosaic schedule
        "close_mosaic": 10,

        # Augmentations from Phase 3
        "hsv_h": aug_cfg.get("hsv_h", 0.015),
        "hsv_s": aug_cfg.get("hsv_s", 0.7),
        "hsv_v": aug_cfg.get("hsv_v", 0.4),
        "degrees": aug_cfg.get("degrees", 10.0),
        "translate": aug_cfg.get("translate", 0.1),
        "scale": aug_cfg.get("scale", 0.5),
        "shear": aug_cfg.get("shear", 2.0),
        "perspective": aug_cfg.get("perspective", 0.0005),
        "fliplr": aug_cfg.get("fliplr", 0.5),
        "flipud": aug_cfg.get("flipud", 0.0),
        "mosaic": aug_cfg.get("mosaic", 1.0),
        "mixup": aug_cfg.get("mixup", 0.15),
        "copy_paste": aug_cfg.get("copy_paste", 0.3),
        "erasing": aug_cfg.get("erasing", 0.2),

        # Output
        "project": str(output_dir),
        "name": args.name,
        "exist_ok": True,
        "save": True,
        "save_period": -1,  # save only best + last
        "plots": False,     # Disable epoch curve plotting to prevent RAM exhaustion on 8GB RAM
        "verbose": True,
    }

    # Log system info
    sys_info = get_system_info()
    print("\n=== System Info ===")
    for k, v in sys_info.items():
        print(f"  {k}: {v}")

    print(f"\n=== Training Config ===")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch: {args.batch}")
    print(f"  Image Size: {args.imgsz}")
    print(f"  Patience: {args.patience}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  AMP: {train_args['amp']}")

    # Initialize model
    model = YOLO(args.model)
    print(f"\nModel loaded: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

    # Train
    start_time = time.time()
    results = model.train(**train_args)
    training_time = time.time() - start_time

    # Extract metrics
    metrics = {
        "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
        "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
        "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
        "mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
        "training_time_hours": round(training_time / 3600, 2),
        "best_epoch": int(results.results_dict.get("epoch", -1)),
    }
    metrics["f1"] = round(
        2 * metrics["precision"] * metrics["recall"]
        / (metrics["precision"] + metrics["recall"] + 1e-8), 4
    )

    print(f"\n=== Training Complete ===")
    print(f"  Time: {metrics['training_time_hours']} hours")
    print(f"  mAP50: {metrics['mAP50']:.4f}")
    print(f"  mAP50-95: {metrics['mAP50_95']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")

    # Save experiment record
    experiment_record = {
        "experiment": {
            "name": args.name,
            "phase": 4,
            "date": datetime.now().isoformat(),
        },
        "system_info": sys_info,
        "dataset_stats": dataset_stats,
        "training_config": {k: str(v) if isinstance(v, Path) else v for k, v in train_args.items()},
        "class_weights": class_weights,
        "results": metrics,
        "model_path": str(output_dir / args.name / "weights" / "best.pt"),
    }

    record_path = output_dir / args.name / "experiment_record.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with open(record_path, "w") as f:
        json.dump(experiment_record, f, indent=2)
    print(f"\nExperiment record saved: {record_path}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4: Baseline YOLOv8 Damage Detector")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Model weights (yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (16 for nano on 4GB, 8 for small)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--name", type=str, default="phase4_yolov8n_baseline",
                        help="Experiment run name")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to experiment YAML config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
