"""Phase 5 — Damage Classifier Evaluation Script.

Evaluates trained damage classifier on held-out test set crops:
- Computes overall accuracy, macro/weighted Precision, Recall, and F1
- Computes per-class metrics (support, precision, recall, f1)
- Generates normalized confusion matrix heatmap
- Exports evaluation/classification/phase5_test_report.json
- Compares against Phase 4 YOLO direct classification baseline
"""

import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.dataset import (
    CLASS_NAMES,
    DamageCropDataset,
    get_transforms,
)
from src.classification.model import DamageClassifier


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: Path):
    """Plot raw and normalized confusion matrices side-by-side."""
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        cbar=True,
    )
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=13, pad=10)
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Label", fontsize=11)
    axes[0].tick_params(axis="x", rotation=30)

    # Normalized percentages
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
        cbar=True,
    )
    axes[1].set_title("Normalized Confusion Matrix (Recall)", fontsize=13, pad=10)
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_ylabel("True Label", fontsize=11)
    axes[1].tick_params(axis="x", rotation=30)

    plt.suptitle("Phase 5 Damage Classification — Test Set Performance", fontsize=15, y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def evaluate(args):
    """Execute evaluation on test set."""
    model_path = Path(args.weights)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[Phase 5] Evaluating on device: {device}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    backbone_name = cfg.get("model", {}).get("backbone", "efficientnet_b0")
    img_size = cfg.get("training", {}).get("image_size", 224)

    # Rebuild model
    model = DamageClassifier(
        backbone_name=backbone_name,
        num_classes=len(CLASS_NAMES),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test dataset
    crop_dir = PROJECT_ROOT / cfg.get("dataset", {}).get("crop_dir", "data/processed/damage_crops")
    meta_path = crop_dir / "crops_metadata.csv"

    test_transform = get_transforms(img_size=img_size, is_train=False)
    test_ds = DamageCropDataset(
        metadata=meta_path,
        split=args.split,
        transform=test_transform,
        base_dir=crop_dir,
    )
    print(f"[Phase 5] Loaded {len(test_ds)} crops from split='{args.split}'")

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_preds, all_targets, all_probs = [], [], []
    latencies = []

    print("[Phase 5] Running inference benchmark...")
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            t0 = time.perf_counter()
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                logits = model(images)
                probs = torch.softmax(logits, dim=-1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) / images.size(0))

            preds = torch.argmax(probs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    avg_latency_ms = float(np.mean(latencies) * 1000)

    # Compute metrics
    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))
    macro_p = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
    macro_r = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(CLASS_NAMES))))
    report_dict = classification_report(
        all_targets,
        all_preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    # Compare with Phase 4 Baseline Detector results if report exists
    phase4_report_path = PROJECT_ROOT / "evaluation" / "detection" / "phase4_test_report.json"
    phase4_comparison = {}
    if phase4_report_path.exists():
        with open(phase4_report_path, "r") as f:
            p4_data = json.load(f)
            p4_classes = p4_data.get("per_class_test_metrics", {})
            for c_name in CLASS_NAMES:
                if c_name in p4_classes:
                    phase4_comparison[c_name] = {
                        "phase4_detector_f1": p4_classes[c_name].get("f1", 0.0),
                        "phase5_classifier_f1": report_dict[c_name]["f1-score"],
                        "f1_delta": round(report_dict[c_name]["f1-score"] - p4_classes[c_name].get("f1", 0.0), 4),
                    }

    out_dir = PROJECT_ROOT / "evaluation" / "classification"
    out_dir.mkdir(parents=True, exist_ok=True)

    cm_plot_path = out_dir / f"phase5_confusion_matrix_{args.split}.png"
    plot_confusion_matrix(cm, CLASS_NAMES, cm_plot_path)

    results_payload = {
        "evaluation_split": args.split,
        "sample_count": len(test_ds),
        "model_architecture": backbone_name,
        "latency_ms_per_crop": round(avg_latency_ms, 2),
        "overall_metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
        },
        "per_class_metrics": {
            name: {
                "precision": round(report_dict[name]["precision"], 4),
                "recall": round(report_dict[name]["recall"], 4),
                "f1_score": round(report_dict[name]["f1-score"], 4),
                "support": int(report_dict[name]["support"]),
            }
            for name in CLASS_NAMES
        },
        "phase4_vs_phase5_f1_comparison": phase4_comparison,
        "confusion_matrix": cm.tolist(),
    }

    report_json_path = out_dir / f"phase5_{args.split}_report.json"
    with open(report_json_path, "w") as f:
        json.dump(results_payload, f, indent=2)

    print("\n" + "=" * 65)
    print(f" Phase 5 Damage Classification — {args.split.upper()} Evaluation Results")
    print("=" * 65)
    print(f" Overall Accuracy : {acc * 100:.2f}%")
    print(f" Macro F1 Score   : {macro_f1:.4f}")
    print(f" Weighted F1      : {weighted_f1:.4f}")
    print(f" Avg Crop Latency : {avg_latency_ms:.2f} ms")
    print("-" * 65)
    print(f"{'Class':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 65)
    for name in CLASS_NAMES:
        m = report_dict[name]
        print(f"{name:<16} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1-score']:<10.4f} | {m['support']:<8}")
    print("=" * 65)
    print(f"Report saved to: {report_json_path}")
    print(f"Confusion Matrix saved to: {cm_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 Classifier Evaluation")
    parser.add_argument("--weights", type=str, default="models/classification/phase5_damage_classifier/best_classifier.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    evaluate(args)
