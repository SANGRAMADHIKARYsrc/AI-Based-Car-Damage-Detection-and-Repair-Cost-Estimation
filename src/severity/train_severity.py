"""Phase 7 — Vehicle Damage Severity Classifier Training Script.

Trains 3-class severity classifier (minor, moderate, severe) on data3a.
- Stratified 70/15/15 split
- Pretrained ResNet18 transfer learning
- PyTorch AMP mixed precision on RTX 2050 GPU
- CosineAnnealingLR & early stopping
- Saves checkpoints and training curves to models/severity/phase7_severity_model/
"""

import argparse
import json
import os
import platform
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.severity.dataset import (
    SEVERITY_CLASSES,
    SeverityDataset,
    get_severity_transforms,
    prepare_stratified_severity_splits,
)
from src.severity.model import DamageSeverityClassifier


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
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

    return running_loss / len(loader.dataset), accuracy_score(all_targets, all_preds)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
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
    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)

    res = {
        "loss": float(val_loss),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "f1_minor": float(per_class_f1[0]),
        "f1_moderate": float(per_class_f1[1]),
        "f1_severe": float(per_class_f1[2]),
    }
    return res


def plot_curves(history: pd.DataFrame, save_path: Path):
    plt.figure(figsize=(11, 4.5))

    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss", color="royalblue", lw=2)
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss", color="crimson", lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Severity Classifier — Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["train_acc"], label="Train Acc", color="royalblue", lw=2)
    plt.plot(history["epoch"], history["val_acc"], label="Val Acc", color="seagreen", lw=2)
    plt.plot(history["epoch"], history["val_macro_f1"], label="Val Macro F1", color="darkorange", lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Severity Classifier — Accuracy & F1")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def train(args):
    config_path = Path(args.config)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = args.lr or cfg["training"]["learning_rate"]
    backbone_name = cfg["model"].get("backbone", "resnet18")
    seed = cfg["training"].get("random_seed", 42)
    patience = cfg["training"].get("patience", 8)
    img_size = cfg["training"].get("image_size", 224)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[Phase 7] Severity Training Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)

    # 1. Dataset splits
    meta_path = PROJECT_ROOT / cfg["dataset"]["source"]
    stratified_df = prepare_stratified_severity_splits(meta_path, seed=seed)

    # Save stratified metadata
    strat_meta_path = PROJECT_ROOT / "data" / "processed" / "severity_metadata_stratified.csv"
    stratified_df.to_csv(strat_meta_path, index=False)

    train_tf = get_severity_transforms(img_size=img_size, is_train=True)
    val_tf = get_severity_transforms(img_size=img_size, is_train=False)

    train_ds = SeverityDataset(stratified_df, split="train", transform=train_tf, project_root=PROJECT_ROOT)
    val_ds = SeverityDataset(stratified_df, split="val", transform=val_tf, project_root=PROJECT_ROOT)

    print(f"[Phase 7] Split: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(stratified_df[stratified_df['split_stratified']=='test'])}", flush=True)
    print(f"  Train classes: {train_ds.get_class_counts()}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True if device.type == "cuda" else False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True if device.type == "cuda" else False)

    # 2. Model & Optimization
    model = DamageSeverityClassifier(
        backbone_name=backbone_name,
        num_classes=3,
        pretrained=cfg["model"].get("pretrained", True),
        dropout_rate=cfg["model"].get("dropout_rate", 0.3),
        hidden_dim=cfg["model"].get("hidden_dim", 128),
    ).to(device)

    label_smooth = cfg["training"].get("label_smoothing", 0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=cfg["training"].get("weight_decay", 1e-4))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")

    out_dir = PROJECT_ROOT / "models" / "severity" / "phase7_severity_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = out_dir / "best_severity_model.pt"

    print(f"\n[Phase 7] Starting training: {epochs} epochs, batch_size={batch_size}, lr={lr}, backbone={backbone_name}", flush=True)
    print("-" * 75, flush=True)

    best_val_f1 = 0.0
    patience_cnt = 0
    history = []
    start_t = time.time()

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        v_res = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        ep_sec = time.time() - ep_t0

        current_f1 = v_res["macro_f1"]
        current_acc = v_res["accuracy"]

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": v_res["loss"],
            "val_acc": current_acc,
            "val_macro_f1": current_f1,
            "f1_minor": v_res["f1_minor"],
            "f1_moderate": v_res["f1_moderate"],
            "f1_severe": v_res["f1_severe"],
            "lr": optimizer.param_groups[0]["lr"],
            "duration_sec": ep_sec,
        })

        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            patience_cnt = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": current_f1,
                "val_accuracy": current_acc,
                "config": cfg,
                "classes": SEVERITY_CLASSES,
            }, best_weights_path)
            flag = "[BEST]"
        else:
            patience_cnt += 1
            flag = f"(patience: {patience_cnt}/{patience})"

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | "
            f"Val Loss: {v_res['loss']:.4f} Acc: {current_acc:.3f} Macro-F1: {current_f1:.4f} | "
            f"Min/Mod/Sev F1: [{v_res['f1_minor']:.2f}/{v_res['f1_moderate']:.2f}/{v_res['f1_severe']:.2f}] | "
            f"{ep_sec:.1f}s {flag}",
            flush=True,
        )

        if patience_cnt >= patience:
            print(f"\n[Phase 7] Early stopping at epoch {epoch}.", flush=True)
            break

    total_time = time.time() - start_t
    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "results.csv", index=False)
    plot_curves(history_df, out_dir / "results.png")

    record = {
        "experiment": cfg["experiment"],
        "dataset": cfg["dataset"],
        "model": cfg["model"],
        "training": {**cfg["training"], "total_epochs": len(history_df), "time_sec": round(total_time, 2)},
        "best_metrics": {"val_macro_f1": float(best_val_f1), "best_val_acc": float(history_df["val_acc"].max())},
    }
    with open(out_dir / "experiment_record.json", "w") as f:
        json.dump(record, f, indent=2)

    print("-" * 75, flush=True)
    print(f"[Phase 7] Complete in {total_time:.1f}s. Best Val Macro-F1: {best_val_f1:.4f}", flush=True)
    print(f"  Weights saved to: {best_weights_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/phase7_severity_assessment.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    train(args)
