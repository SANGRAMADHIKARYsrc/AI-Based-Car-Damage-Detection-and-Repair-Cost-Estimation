"""Phase 7 — Vehicle Damage Severity Dataset & Transforms.

Loads 3-class severity images (minor, moderate, severe) from data3a.
Supports stratified train/val/test splits, domain augmentations, and PyTorch DataLoaders.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

SEVERITY_CLASSES = ["minor", "moderate", "severe"]
SEVERITY_TO_IDX = {name: idx for idx, name in enumerate(SEVERITY_CLASSES)}
IDX_TO_SEVERITY = {idx: name for idx, name in enumerate(SEVERITY_CLASSES)}


def get_severity_transforms(
    img_size: int = 224,
    is_train: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """Return torchvision transform pipeline for severity classification."""
    if is_train:
        return T.Compose([
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomCrop((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


def prepare_stratified_severity_splits(
    meta_path: Union[str, Path],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> pd.DataFrame:
    """Stratify 1,631 images into balanced train/val/test splits."""
    df = pd.read_csv(meta_path)

    # Stratified split: train vs temp (val + test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["severity_class_id"],
        random_state=seed,
    )

    # Split temp into val and test
    val_test_ratio = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_test_ratio,
        stratify=temp_df["severity_class_id"],
        random_state=seed,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split_stratified"] = "train"
    val_df["split_stratified"] = "val"
    test_df["split_stratified"] = "test"

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return combined


class SeverityDataset(Dataset):
    """PyTorch Dataset for Car Damage Severity Assessment."""

    def __init__(
        self,
        metadata: Union[str, Path, pd.DataFrame],
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        project_root: Optional[Path] = None,
        split_column: str = "split_stratified",
    ):
        if isinstance(metadata, (str, Path)):
            self.df = pd.read_csv(metadata)
        else:
            self.df = metadata.copy()

        if split and split_column in self.df.columns:
            self.df = self.df[self.df[split_column] == split].reset_index(drop=True)

        self.transform = transform
        self.project_root = project_root

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        rel_path = str(row["relative_path"])
        full_path = (self.project_root / rel_path) if self.project_root else Path(rel_path)

        image = Image.open(full_path).convert("RGB")
        target = int(row["severity_class_id"])

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def get_class_counts(self) -> Dict[str, int]:
        """Return sample counts per severity grade."""
        return self.df["severity_label"].value_counts().to_dict()
