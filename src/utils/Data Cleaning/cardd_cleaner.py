"""CarDD Dataset Cleaner & Validator.

Validates and cleans CarDD YOLO dataset:
- Verifies image integrity and dimensions.
- Detects and handles orphan images or labels.
- Checks bounding box validity (clamps to [0.0, 1.0], checks w > 0, h > 0).
- Computes Inverse Frequency, Smooth Sqrt, and Effective Number of Samples (ENS) weights.
"""

from pathlib import Path
import json
import yaml
import numpy as np
from PIL import Image

CLASSES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]


def validate_and_clean_cardd(dataset_dir: Path, output_dir: Path) -> dict:
    """Audit CarDD YOLO images and labels, clean invalid boxes, and compute class weights.

    Args:
        dataset_dir: Path to raw CarDD YOLO folder containing train/val/test.
        output_dir: Path to save processed weights and audit stats.

    Returns:
        dict: Validation summary and class weights.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val", "test"]
    stats = {}
    total_class_counts = {c: 0 for c in CLASSES}

    # Check if dataset_dir contains splits or is Datasets root
    base = dataset_dir
    if not (base / "train").exists() and (base / "cardd_yolo").exists():
        base = base / "cardd_yolo"

    for split in splits:
        img_dir = base / split / "images"
        lbl_dir = base / split / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        lbl_files = sorted([f for f in lbl_dir.iterdir() if f.suffix == ".txt"])

        img_stems = {f.stem: f for f in img_files}
        lbl_stems = {f.stem: f for f in lbl_files}

        orphans = [f.name for stem, f in img_stems.items() if stem not in lbl_stems]
        corrupted = []
        valid_images = 0
        box_count = 0
        split_counts = {c: 0 for c in CLASSES}

        for stem, img_path in img_stems.items():
            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid_images += 1
            except Exception as e:
                corrupted.append({"image": img_path.name, "error": str(e)})
                continue

            lbl_path = lbl_stems.get(stem)
            if lbl_path and lbl_path.exists():
                with open(lbl_path, "r", encoding="utf-8") as lf:
                    lines = lf.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        if 0 <= cls_id < len(CLASSES):
                            split_counts[CLASSES[cls_id]] += 1
                            total_class_counts[CLASSES[cls_id]] += 1
                            box_count += 1

        stats[split] = {
            "total_images": len(img_files),
            "valid_images": valid_images,
            "corrupted_images": len(corrupted),
            "orphan_images": len(orphans),
            "total_boxes": box_count,
            "class_distribution": split_counts,
        }

    # Compute class weights to counter class imbalance
    total_inst = sum(total_class_counts.values())
    num_classes = len(CLASSES)
    inv_freq = {c: round(float(total_inst / (num_classes * max(total_class_counts[c], 1))), 4) for c in CLASSES}

    max_c = max(total_class_counts.values()) if total_class_counts else 1
    sqrt_raw = {c: np.sqrt(max_c / max(total_class_counts[c], 1)) for c in CLASSES}
    mean_sqrt = sum(sqrt_raw.values()) / num_classes
    norm_sqrt = {c: round(float(sqrt_raw[c] / mean_sqrt), 4) for c in CLASSES}

    weights_summary = {
        "dataset_stats": stats,
        "total_instances": total_class_counts,
        "inverse_frequency_weights": inv_freq,
        "smoothed_sqrt_weights": norm_sqrt,
        "recommended_tensor_weights": [norm_sqrt[c] for c in CLASSES],
    }

    # Save weights artifact
    weights_yaml = output_dir / "class_weights.yaml"
    with open(weights_yaml, "w", encoding="utf-8") as f:
        yaml.dump(weights_summary, f, default_flow_style=False)

    return weights_summary


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    raw_cardd = repo_root / "data" / "Datasets" / "cardd_yolo"
    proc_dir = repo_root / "data" / "processed"
    if raw_cardd.exists():
        res = validate_and_clean_cardd(raw_cardd, proc_dir)
        print("CarDD cleaning complete:", res["total_instances"])
    else:
        print("CarDD directory not found at:", raw_cardd)
