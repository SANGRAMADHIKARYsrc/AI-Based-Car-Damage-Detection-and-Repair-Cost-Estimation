"""Phase 5 — Damage Crop Dataset and Extraction Utilities.

Extracts localized damage regions from CarDD YOLO annotations, applies
context-preserving padding, and provides PyTorch Datasets with domain-specific
data augmentations for fine-grained damage classification.
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

CLASS_NAMES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}


def get_transforms(
    img_size: int = 224,
    is_train: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """Return torchvision transform pipeline for training or validation."""
    if is_train:
        return T.Compose([
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomCrop((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


def extract_damage_crops(
    dataset_root: Union[str, Path],
    output_dir: Union[str, Path],
    padding_ratio: float = 0.15,
    min_size_px: int = 16,
    splits: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Extract damage bounding box crops from CarDD YOLO dataset with context padding.

    Args:
        dataset_root: Directory containing train/val/test with images/ and labels/.
        output_dir: Directory to save cropped image hierarchy and metadata CSV.
        padding_ratio: Context margin around bbox relative to box dimensions.
        min_size_px: Minimum width and height for valid crops.
        splits: List of splits to process (default: ['train', 'val', 'test']).

    Returns:
        pd.DataFrame with full metadata of all extracted crops.
    """
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    splits = splits or ["train", "val", "test"]

    records = []
    print(f"[Phase 5] Extracting damage crops from {dataset_root} -> {output_dir}")

    for split in splits:
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"  Warning: Split '{split}' missing images or labels directory. Skipping.")
            continue

        split_out_dir = output_dir / split
        for cls_name in CLASS_NAMES:
            (split_out_dir / cls_name.replace(" ", "_")).mkdir(parents=True, exist_ok=True)

        label_files = sorted(list(lbl_dir.glob("*.txt")))
        print(f"  Processing '{split}': {len(label_files)} label files...")

        crop_counter = 0
        for lbl_path in label_files:
            stem = lbl_path.stem
            # Find matching image
            img_candidates = list(img_dir.glob(f"{stem}.*"))
            if not img_candidates:
                continue
            img_path = img_candidates[0]

            # Read image
            cv_img = cv2.imread(str(img_path))
            if cv_img is None:
                continue
            img_h, img_w = cv_img.shape[:2]

            # Parse YOLO labels: class_id x_center y_center width height
            with open(lbl_path, "r") as f:
                lines = [line.strip().split() for line in f if line.strip()]

            for box_idx, parts in enumerate(lines):
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(float(parts[0]))
                    xc, yc, bw, bh = map(float, parts[1:5])
                except ValueError:
                    continue

                if cls_id < 0 or cls_id >= len(CLASS_NAMES):
                    continue

                cls_name = CLASS_NAMES[cls_id]
                cls_folder = cls_name.replace(" ", "_")

                # Box coordinates in pixels
                box_w_px = bw * img_w
                box_h_px = bh * img_h
                pad_w = box_w_px * padding_ratio
                pad_h = box_h_px * padding_ratio

                x1 = max(0, int((xc * img_w) - (box_w_px / 2.0) - pad_w))
                y1 = max(0, int((yc * img_h) - (box_h_px / 2.0) - pad_h))
                x2 = min(img_w, int((xc * img_w) + (box_w_px / 2.0) + pad_w))
                y2 = min(img_h, int((yc * img_h) + (box_h_px / 2.0) + pad_h))

                crop_w = x2 - x1
                crop_h = y2 - y1

                if crop_w < min_size_px or crop_h < min_size_px:
                    continue

                crop = cv_img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_filename = f"{stem}_crop{box_idx:02d}_{cls_folder}.jpg"
                crop_rel_path = f"{split}/{cls_folder}/{crop_filename}"
                crop_full_path = output_dir / crop_rel_path

                cv2.imwrite(str(crop_full_path), crop)
                crop_counter += 1

                records.append({
                    "crop_id": f"{split}_{stem}_{box_idx}",
                    "split": split,
                    "crop_rel_path": crop_rel_path,
                    "crop_full_path": str(crop_full_path),
                    "original_image": str(img_path),
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "box_x_center": xc,
                    "box_y_center": yc,
                    "box_width": bw,
                    "box_height": bh,
                    "crop_width_px": crop_w,
                    "crop_height_px": crop_h,
                })

        print(f"  Split '{split}' done: extracted {crop_counter} valid crops.")

    df = pd.DataFrame(records)
    meta_path = output_dir / "crops_metadata.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(meta_path, index=False)
    print(f"[Phase 5] Crop metadata saved to {meta_path} (Total crops: {len(df)})")
    return df


class DamageCropDataset(Dataset):
    """PyTorch Dataset for Car Damage Crops."""

    def __init__(
        self,
        metadata: Union[str, Path, pd.DataFrame],
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        base_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize dataset.

        Args:
            metadata: Path to crops_metadata.csv or pre-loaded DataFrame.
            split: Filter to specific split ('train', 'val', 'test').
            transform: PyTorch image transform callable.
            base_dir: Base directory if crop_rel_path is relative.
        """
        if isinstance(metadata, (str, Path)):
            self.df = pd.read_csv(metadata)
        else:
            self.df = metadata.copy()

        if split:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.transform = transform
        self.base_dir = Path(base_dir) if base_dir else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]

        if "crop_full_path" in row and os.path.exists(str(row["crop_full_path"])):
            img_path = str(row["crop_full_path"])
        elif self.base_dir:
            img_path = str(self.base_dir / row["crop_rel_path"])
        else:
            img_path = str(row["crop_rel_path"])

        image = Image.open(img_path).convert("RGB")
        target = int(row["class_id"])

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def get_class_counts(self) -> Dict[str, int]:
        """Return counts of crops per damage class."""
        return self.df["class_name"].value_counts().to_dict()

    def get_class_weights(self) -> torch.Tensor:
        """Compute smoothed inverse weights for the current split."""
        counts = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        for cls_id in range(len(CLASS_NAMES)):
            counts[cls_id] = (self.df["class_id"] == cls_id).sum()

        counts = np.maximum(counts, 1.0)
        max_c = np.max(counts)
        weights = np.sqrt(max_c / counts)
        weights = weights / np.mean(weights)
        return torch.tensor(weights, dtype=torch.float32)
