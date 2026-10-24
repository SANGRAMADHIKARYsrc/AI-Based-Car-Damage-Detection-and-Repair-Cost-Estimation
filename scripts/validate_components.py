"""Component Inference Validation Tool.

Provides a systematic protocol to manually benchmark the 3-tier component reasoning engine:
1. --generate: Samples 150 CarDD test instances, runs component inference, and generates
   a CSV (`evaluation/component/component_human_validation_150.csv`) with a blank
   `ground_truth_component` column for human annotation.
2. --evaluate: Reads the annotated CSV, computes overall accuracy, macro/weighted F1,
   per-class precision/recall/F1, prints a formatted confusion matrix, and saves an evaluation JSON.

Usage:
    # 1. Generate the 150-sample CSV for annotation
    python scripts/validate_components.py --generate

    # 2. Hand-label `ground_truth_component` in evaluation/component/component_human_validation_150.csv
    # Valid classes: front_bumper, rear_bumper, hood, door, fender, quarter_panel,
    #                roof, windshield_glass, lamp, wheel_tire

    # 3. Compute accuracy and confusion matrix
    python scripts/validate_components.py --evaluate
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.component.identifier import VehicleComponentIdentifier
from src.component.ontology import VEHICLE_COMPONENTS

CLASS_NAMES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
VALID_COMPONENTS = sorted(VEHICLE_COMPONENTS)


def generate_validation_csv(output_csv: Path, n_samples: int = 150, seed: int = 42) -> Path:
    """Sample test instances and generate pre-filled CSV for human annotation."""
    test_labels_dir = REPO_ROOT / "data" / "Datasets" / "test" / "labels"
    test_images_dir = REPO_ROOT / "data" / "Datasets" / "test" / "images"

    if not test_labels_dir.exists():
        raise FileNotFoundError(f"Missing test labels directory at {test_labels_dir}")

    all_instances = []
    label_files = sorted(test_labels_dir.glob("*.txt"))

    for lf in label_files:
        stem = lf.stem
        img_name = f"{stem}.jpg"
        img_path = test_images_dir / img_name
        if not img_path.exists():
            continue

        with open(lf, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    all_instances.append({
                        "image_file": img_name,
                        "instance_idx": line_idx,
                        "class_id": cls_id,
                        "damage_type": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}",
                        "box_x_center": xc,
                        "box_y_center": yc,
                        "box_width": w,
                        "box_height": h,
                    })

    total_available = len(all_instances)
    print(f"Discovered {total_available} total ground-truth damage instances across {len(label_files)} test images.")

    # Stratified or random sample with seed
    rng = np.random.default_rng(seed)
    n_sample = min(n_samples, total_available)
    sampled_indices = rng.choice(total_available, size=n_sample, replace=False)
    sampled_indices.sort()

    identifier = VehicleComponentIdentifier()
    rows = []
    for rank, idx in enumerate(sampled_indices, 1):
        item = all_instances[idx]
        norm_bbox = (
            item["box_x_center"],
            item["box_y_center"],
            item["box_width"],
            item["box_height"],
        )
        res = identifier.identify_component(
            damage_class=item["damage_type"],
            norm_bbox=norm_bbox,
        )

        rows.append({
            "sample_id": rank,
            "image_file": item["image_file"],
            "damage_type": item["damage_type"],
            "bbox_norm_xc_yc_w_h": f"{item['box_x_center']:.4f},{item['box_y_center']:.4f},{item['box_width']:.4f},{item['box_height']:.4f}",
            "predicted_component": res["primary_component"],
            "confidence": round(res["confidence"], 4),
            "inference_method": res["method"],
            "copart_sector": res["copart_sector"],
            "ground_truth_component": "",  # To be filled by user
            "notes": "",
        })

    out_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"\n[OK] Successfully created human validation sheet with {len(out_df)} instances at:")
    print(f"     --> {output_csv.resolve()}")
    print("\nInstructions:")
    print("  1. Open the CSV and view the image in `data/Datasets/test/images/<image_file>`.")
    print("  2. Fill in `ground_truth_component` with one of the 10 valid labels:")
    print("     ", ", ".join(VALID_COMPONENTS))
    print("  3. Run: python scripts/validate_components.py --evaluate\n")
    return output_csv


def evaluate_validation_csv(csv_path: Path):
    """Compute accuracy, per-class metrics, and confusion matrix on annotated rows."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Validation CSV not found: {csv_path}. Run with --generate first.")

    df = pd.read_csv(csv_path)
    if "ground_truth_component" not in df.columns:
        raise ValueError("Missing column 'ground_truth_component' in CSV.")

    filled = df[df["ground_truth_component"].notna() & (df["ground_truth_component"].astype(str).str.strip() != "")].copy()
    num_filled = len(filled)
    if num_filled == 0:
        print(f"\n[!] No annotated rows found in {csv_path}.")
        print("    Please populate the `ground_truth_component` column before running --evaluate.")
        return

    filled["ground_truth_component"] = filled["ground_truth_component"].astype(str).str.strip().str.lower()
    filled["predicted_component"] = filled["predicted_component"].astype(str).str.strip().str.lower()

    # Validate classes
    invalid = filled[~filled["ground_truth_component"].isin(VALID_COMPONENTS)]
    if len(invalid) > 0:
        print(f"[Warning] Found {len(invalid)} rows with invalid component names:")
        print(invalid[["sample_id", "ground_truth_component"]])
        print(f"Valid classes are: {VALID_COMPONENTS}\n")

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    y_true = filled["ground_truth_component"].tolist()
    y_pred = filled["predicted_component"].tolist()

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Union of present labels
    present_labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    cm_df = pd.DataFrame(cm, index=[f"True: {l}" for l in present_labels], columns=[f"Pred: {l}" for l in present_labels])

    clf_rep = classification_report(y_true, y_pred, labels=present_labels, output_dict=True, zero_division=0)

    print("\n" + "=" * 65)
    print(f"COMPONENT INFERENCE VALIDATION REPORT (Annotated N = {num_filled})")
    print("=" * 65)
    print(f"  Overall Accuracy: {acc * 100:.2f}% ({np.sum(np.array(y_true) == np.array(y_pred))}/{num_filled})")
    print(f"  Macro-Averaged F1: {macro_f1:.4f}")
    print(f"  Weighted F1:       {weighted_f1:.4f}")
    print("\nPer-Class Breakdown:")
    for lbl in present_labels:
        if lbl in clf_rep:
            d = clf_rep[lbl]
            print(f"  - {lbl:<18}: Prec={d['precision']:.3f} | Rec={d['recall']:.3f} | F1={d['f1-score']:.3f} | Supp={d['support']}")

    print("\nConfusion Matrix:")
    print(cm_df.to_string())

    # Save report
    out_json = csv_path.parent / "component_validation_results.json"
    results_payload = {
        "num_annotated": num_filled,
        "overall_accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "present_classes": present_labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_rep,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n[OK] Saved results to: {out_json.resolve()}\n")


def main():
    parser = argparse.ArgumentParser(description="Component Inference Validation Tool")
    parser.add_argument("--generate", action="store_true", help="Generate 150-sample CSV for hand-labeling")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate filled CSV and compute metrics")
    parser.add_argument(
        "--csv",
        type=str,
        default="evaluation/component/component_human_validation_150.csv",
        help="Path to CSV file",
    )
    parser.add_argument("--samples", type=int, default=150, help="Number of samples to draw")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    target_csv = Path(args.csv)
    if not target_csv.is_absolute():
        target_csv = REPO_ROOT / target_csv

    if args.evaluate:
        evaluate_validation_csv(target_csv)
    else:
        # Default or explicit generate
        generate_validation_csv(target_csv, n_samples=args.samples, seed=args.seed)


if __name__ == "__main__":
    main()
