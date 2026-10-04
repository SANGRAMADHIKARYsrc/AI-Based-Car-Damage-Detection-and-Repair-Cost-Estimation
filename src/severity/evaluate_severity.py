"""Phase 7 — Damage Severity Evaluation Benchmark.

Evaluates trained severity model on held-out test split (N=245):
- Computes overall accuracy, macro F1, per-class metrics
- Plots normalized confusion matrix heatmap
- Exports evaluation/severity/phase7_severity_report.json
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.severity.dataset import (
    SEVERITY_CLASSES,
    SeverityDataset,
    get_severity_transforms,
)
from src.severity.model import DamageSeverityClassifier


def plot_cm(cm: np.ndarray, save_path: Path):
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=SEVERITY_CLASSES, yticklabels=SEVERITY_CLASSES, ax=axes[0])
    axes[0].set_title("Severity Confusion Matrix (Counts)", fontsize=12)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=SEVERITY_CLASSES, yticklabels=SEVERITY_CLASSES, ax=axes[1])
    axes[1].set_title("Normalized Confusion Matrix (Recall)", fontsize=12)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def evaluate(args):
    model_path = Path(args.weights)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[Phase 7] Evaluating on device: {device}", flush=True)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    backbone = cfg.get("model", {}).get("backbone", "resnet18")
    img_size = cfg.get("training", {}).get("image_size", 224)

    model = DamageSeverityClassifier(backbone_name=backbone, num_classes=3, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    strat_meta_path = PROJECT_ROOT / "data" / "processed" / "severity_metadata_stratified.csv"
    strat_df = pd.read_csv(strat_meta_path)

    test_tf = get_severity_transforms(img_size=img_size, is_train=False)
    test_ds = SeverityDataset(strat_df, split=args.split, transform=test_tf, project_root=PROJECT_ROOT)
    print(f"[Phase 7] Loaded {len(test_ds)} images from split='{args.split}'", flush=True)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_preds, all_targets = [], []
    latencies = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            t0 = time.perf_counter()
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                logits = model(images)
                probs = torch.softmax(logits, dim=-1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) / images.size(0))

            preds = torch.argmax(probs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    avg_latency = float(np.mean(latencies) * 1000)

    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro"))
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])
    rep = classification_report(all_targets, all_preds, target_names=SEVERITY_CLASSES, output_dict=True)

    out_dir = PROJECT_ROOT / "evaluation" / "severity"
    out_dir.mkdir(parents=True, exist_ok=True)

    cm_path = out_dir / f"phase7_confusion_matrix_{args.split}.png"
    plot_cm(cm, cm_path)

    report_payload = {
        "evaluation_split": args.split,
        "sample_count": len(test_ds),
        "latency_ms_per_image": round(avg_latency, 2),
        "overall_metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(rep["weighted avg"]["f1-score"], 4),
        },
        "per_class_metrics": {
            c: {
                "precision": round(rep[c]["precision"], 4),
                "recall": round(rep[c]["recall"], 4),
                "f1_score": round(rep[c]["f1-score"], 4),
                "support": int(rep[c]["support"]),
            }
            for c in SEVERITY_CLASSES
        },
        "confusion_matrix": cm.tolist(),
    }

    rep_path = out_dir / f"phase7_severity_report_{args.split}.json"
    with open(rep_path, "w") as f:
        json.dump(report_payload, f, indent=2)

    print("\n" + "=" * 60, flush=True)
    print(f" Phase 7 Damage Severity Assessment — {args.split.upper()} Results", flush=True)
    print("=" * 60, flush=True)
    print(f" Accuracy         : {acc * 100:.2f}%", flush=True)
    print(f" Macro F1 Score   : {macro_f1:.4f}", flush=True)
    print(f" Inference Latency: {avg_latency:.2f} ms/image", flush=True)
    print("-" * 60, flush=True)
    print(f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}", flush=True)
    print("-" * 60, flush=True)
    for c in SEVERITY_CLASSES:
        m = rep[c]
        print(f"{c:<12} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1-score']:<10.4f} | {m['support']:<8}", flush=True)
    print("=" * 60, flush=True)
    print(f"Report saved to: {rep_path}", flush=True)
    print(f"Confusion Matrix saved to: {cm_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="models/severity/phase7_severity_model/best_severity_model.pt")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    evaluate(args)
