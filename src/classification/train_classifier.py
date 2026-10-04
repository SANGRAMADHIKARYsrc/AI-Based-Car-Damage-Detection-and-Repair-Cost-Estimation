"""Phase 5 — Damage Classifier Training Script.

Trains fine-grained damage classifier on localized CarDD crops.
Supports:
- Automatic crop extraction with context padding
- Pretrained backbones (EfficientNet-B0, MobileNetV3, ResNet)
- Class-weighted loss and label smoothing to combat class imbalance
- Mixed precision training (AMP) for RTX 2050 4GB VRAM
- Full reproducibility logging, metrics tracking, and early stopping
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.dataset import (
    CLASS_NAMES,
    DamageCropDataset,
    extract_damage_crops,
    get_transforms,
)
from src.classification.model import DamageClassifier, FocalLoss


def get_system_info() -> dict:
    """Collect hardware and environment metadata."""
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


def set_seed(seed: int = 42):
    """Ensure deterministic training behavior."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Tuple[float, float]:
    """Execute one training epoch with mixed precision."""
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_p = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_r = recall_score(all_targets, all_preds, average="macro", zero_division=0)

    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)

    results = {
        "loss": float(val_loss),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
    }
    for i, cls_name in enumerate(CLASS_NAMES):
        results[f"f1_{cls_name}"] = float(per_class_f1[i]) if i < len(per_class_f1) else 0.0

    return results


def plot_learning_curves(history: pd.DataFrame, save_path: Path):
    """Plot and save training and validation metrics over epochs."""
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss", color="royalblue", lw=2)
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss", color="crimson", lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Damage Classifier — Training & Validation Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Accuracy / Macro F1 plot
    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["train_acc"], label="Train Acc", color="royalblue", lw=2)
    plt.plot(history["epoch"], history["val_acc"], label="Val Acc", color="seagreen", lw=2)
    plt.plot(history["epoch"], history["val_macro_f1"], label="Val Macro F1", color="darkorange", lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Damage Classifier — Accuracy & Macro F1")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def train(args):
    """Main training routine."""
    config_path = Path(args.config)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Override config with CLI arguments if provided
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = args.lr or cfg["training"]["learning_rate"]
    backbone_name = args.backbone or cfg["model"]["backbone"]
    seed = cfg["training"].get("random_seed", 42)
    patience = cfg["training"].get("patience", 8)
    img_size = cfg["training"].get("image_size", 224)

    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    )
    print(f"[Phase 5] Training Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    crop_dir = PROJECT_ROOT / cfg["dataset"]["crop_dir"]
    meta_path = crop_dir / "crops_metadata.csv"

    # Step 1: Ensure crops exist
    if not meta_path.exists() or args.extract_crops:
        raw_dataset_root = PROJECT_ROOT / cfg["dataset"]["source"]
        print("[Phase 5] Extracting CarDD damage crops...")
        extract_damage_crops(
            dataset_root=raw_dataset_root,
            output_dir=crop_dir,
            padding_ratio=cfg["dataset"].get("padding_ratio", 0.15),
        )

    # Step 2: Datasets & Dataloaders
    train_transform = get_transforms(img_size=img_size, is_train=True)
    val_transform = get_transforms(img_size=img_size, is_train=False)

    train_ds = DamageCropDataset(metadata=meta_path, split="train", transform=train_transform, base_dir=crop_dir)
    val_ds = DamageCropDataset(metadata=meta_path, split="val", transform=val_transform, base_dir=crop_dir)

    print(f"[Phase 5] Dataset: Train={len(train_ds)} crops, Val={len(val_ds)} crops")
    print(f"  Train class distribution: {train_ds.get_class_counts()}")

    # On Windows, num_workers=0 avoids multiprocessing deadlocks
    num_workers = 0 if os.name == "nt" else cfg["training"].get("workers", 2)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Step 3: Model Architecture
    model = DamageClassifier(
        backbone_name=backbone_name,
        num_classes=len(CLASS_NAMES),
        pretrained=cfg["model"].get("pretrained", True),
        dropout_rate=cfg["model"].get("dropout_rate", 0.3),
        hidden_dim=cfg["model"].get("hidden_dim", 256),
    ).to(device)

    # Step 4: Class Weights & Loss Function
    raw_weights = cfg["dataset"].get("class_weights")
    if raw_weights:
        class_weights_t = torch.tensor(raw_weights, dtype=torch.float32).to(device)
    else:
        class_weights_t = train_ds.get_class_weights().to(device)

    loss_type = cfg["training"].get("loss_function", "weighted_cross_entropy")
    if loss_type == "focal_loss":
        criterion = FocalLoss(alpha=class_weights_t, gamma=2.0)
    else:
        label_smooth = cfg["training"].get("label_smoothing", 0.05)
        criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=label_smooth)

    # Step 5: Optimizer & LR Scheduler
    weight_decay = cfg["training"].get("weight_decay", 1e-4)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")

    # Output directory
    output_dir = PROJECT_ROOT / "models" / "classification" / "phase5_damage_classifier"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_classifier.pt"
    last_model_path = output_dir / "last_classifier.pt"

    # Step 6: Training Loop
    print(f"\n[Phase 5] Starting training: {epochs} epochs, batch_size={batch_size}, lr={lr}, backbone={backbone_name}", flush=True)
    print("-" * 75, flush=True)

    best_val_macro_f1 = 0.0
    patience_counter = 0
    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        ep_duration = time.time() - ep_start

        current_f1 = val_metrics["macro_f1"]
        current_acc = val_metrics["accuracy"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": current_acc,
            "val_macro_f1": current_f1,
            "val_precision": val_metrics["macro_precision"],
            "val_recall": val_metrics["macro_recall"],
            "lr": optimizer.param_groups[0]["lr"],
            "duration_sec": ep_duration,
        }
        for cls_name in CLASS_NAMES:
            row[f"val_f1_{cls_name}"] = val_metrics.get(f"f1_{cls_name}", 0.0)
        history.append(row)

        is_best = current_f1 > best_val_macro_f1
        if is_best:
            best_val_macro_f1 = current_f1
            patience_counter = 0
            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": current_f1,
                "val_accuracy": current_acc,
                "config": cfg,
                "classes": CLASS_NAMES,
            }
            torch.save(checkpoint, best_model_path)
            saved_flag = "[BEST]"
        else:
            patience_counter += 1
            saved_flag = f"(patience: {patience_counter}/{patience})"

        # Save last checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": current_f1,
                "classes": CLASS_NAMES,
            },
            last_model_path,
        )

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {current_acc:.3f} Macro-F1: {current_f1:.4f} | "
            f"Crack-F1: {val_metrics.get('f1_crack', 0.0):.3f} | "
            f"{ep_duration:.1f}s {saved_flag}",
            flush=True,
        )

        if patience_counter >= patience:
            print(f"\n[Phase 5] Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs).")
            break

    total_time = time.time() - start_time
    history_df = pd.DataFrame(history)
    history_csv = output_dir / "results.csv"
    history_df.to_csv(history_csv, index=False)

    plot_path = output_dir / "results.png"
    plot_learning_curves(history_df, plot_path)

    # Save Experiment Record
    record = {
        "experiment": cfg["experiment"],
        "dataset": cfg["dataset"],
        "model": cfg["model"],
        "training": {
            **cfg["training"],
            "total_epochs_trained": len(history_df),
            "training_time_sec": round(total_time, 2),
        },
        "system_info": get_system_info(),
        "best_metrics": {
            "val_macro_f1": float(best_val_macro_f1),
            "best_epoch": int(history_df.loc[history_df["val_macro_f1"].idxmax()]["epoch"]),
            "best_val_acc": float(history_df.loc[history_df["val_macro_f1"].idxmax()]["val_acc"]),
        },
    }
    with open(output_dir / "experiment_record.json", "w") as f:
        json.dump(record, f, indent=2)

    print("-" * 75)
    print(f"[Phase 5] Training complete in {total_time/60:.2f} mins.")
    print(f"  Best Val Macro-F1: {best_val_macro_f1:.4f}")
    print(f"  Best weights saved to: {best_model_path}")
    print(f"  Curves saved to: {plot_path}")
    print(f"  Results saved to: {history_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 Damage Classifier Training")
    parser.add_argument("--config", type=str, default="experiments/phase5_damage_classifier.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--extract-crops", action="store_true", help="Force re-extraction of crops")
    args = parser.parse_args()
    train(args)
