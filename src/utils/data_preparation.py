"""Phase 3 Data Preparation Pipeline for Car Damage AI
1. Computes class weights for CarDD damage detection (handles severe class imbalance).
2. Converts VehiDE VIA polygon annotations to standard COCO JSON format.
3. Prepares severity classification metadata dataset from data3a.
4. Cleans, filters, and engineers features for Copart repair cost dataset.
"""

import os
import glob
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import yaml

ROOT = Path(r"c:\Users\ASUS\OneDrive\Desktop\Antigravity\Capstone Project\car-damage-ai")
DATA_DIR = ROOT / "data"
DATASETS_DIR = DATA_DIR / "Datasets"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)


def compute_class_weights():
    """Computes standard and smoothed inverse frequency class weights for CarDD."""
    print(">>> 1. Computing class weights for CarDD...")
    # Train counts from Phase 2 audit
    counts = {
        "dent": 1806,
        "scratch": 2560,
        "crack": 651,
        "glass shatter": 475,
        "lamp broken": 494,
        "tire flat": 225
    }
    class_names = list(counts.keys())
    total_instances = sum(counts.values())
    num_classes = len(class_names)
    
    # 1. Standard inverse frequency: N / (K * N_c)
    inv_freq = {c: total_instances / (num_classes * counts[c]) for c in class_names}
    
    # 2. Square-root smoothed weights: sqrt(max(N) / N_c)
    max_count = max(counts.values())
    sqrt_weights = {c: np.sqrt(max_count / counts[c]) for c in class_names}
    
    # Normalize sqrt weights to average 1.0
    mean_w = sum(sqrt_weights.values()) / num_classes
    norm_sqrt = {c: round(float(sqrt_weights[c] / mean_w), 4) for c in class_names}
    
    # Effective Number of Samples (ENS) weights: beta = (N-1)/N
    beta = 0.999
    ens_weights = {}
    for c in class_names:
        n = counts[c]
        ens = (1.0 - beta) / (1.0 - (beta ** n))
        ens_weights[c] = ens
    ens_mean = sum(ens_weights.values()) / num_classes
    norm_ens = {c: round(float(ens_weights[c] / ens_mean), 4) for c in class_names}
    
    weights_data = {
        "train_instance_counts": counts,
        "inverse_frequency_weights": {c: round(float(inv_freq[c]), 4) for c in class_names},
        "smoothed_sqrt_weights": norm_sqrt,
        "effective_number_samples_weights": norm_ens,
        "recommended_loss_weights_tensor": [float(norm_sqrt[c]) for c in class_names]
    }
    
    out_yaml = PROCESSED_DIR / "class_weights.yaml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.dump(weights_data, f, default_flow_style=False)
        
    out_json = PROCESSED_DIR / "class_weights.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(weights_data, f, indent=2)
        
    print(f"Class weights written to {out_yaml}")
    return weights_data


def polygon_area(x, y):
    """Calculates polygon area using Shoelace formula."""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def convert_via_to_coco():
    """Converts VehiDE VIA polygon annotations to standard COCO JSON format."""
    print(">>> 2. Converting VehiDE VIA annotations to COCO format...")
    
    with open(DATA_DIR / "class_mapping.yaml", "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)
    
    # Build COCO category dictionary
    categories = []
    cat_to_id = {}
    for i, (k, v) in enumerate(mapping["vehide_classes"].items(), start=1):
        cat_to_id[k] = i
        categories.append({
            "id": i,
            "name": k,
            "supercategory": "vehicle_damage",
            "english_name": v["english"],
            "cardd_mapped_name": v["cardd_name"]
        })
        
    splits = [
        ("0Train_via_annos.json", "instances_vehide_train.json", "train"),
        ("0Val_via_annos.json", "instances_vehide_val.json", "val")
    ]
    
    summary = {}
    for via_file, coco_file, split_name in splits:
        p = DATASETS_DIR / via_file
        if not p.exists():
            continue
            
        print(f"Reading {via_file}...")
        with open(p, "r", encoding="utf-8") as f:
            via_data = json.load(f)
            
        coco_images = []
        coco_annotations = []
        anno_id = 1
        img_id = 1
        
        for file_key, item in via_data.items():
            img_name = item.get("name", file_key)
            regions = item.get("regions", [])
            
            # Record image entry
            coco_images.append({
                "id": img_id,
                "file_name": img_name,
                "width": 1024, # default nominal if unprobed
                "height": 768
            })
            
            for r in regions:
                all_x = r.get("all_x", [])
                all_y = r.get("all_y", [])
                c = r.get("class", "unknown")
                
                if len(all_x) < 3 or len(all_y) < 3 or len(all_x) != len(all_y):
                    continue
                    
                x_arr = np.array(all_x, dtype=float)
                y_arr = np.array(all_y, dtype=float)
                
                x_min = float(np.min(x_arr))
                y_min = float(np.min(y_arr))
                w = float(np.max(x_arr) - x_min)
                h = float(np.max(y_arr) - y_min)
                area = float(polygon_area(x_arr, y_arr))
                
                # Flatten polygon coordinates [x1, y1, x2, y2, ...]
                poly = []
                for x, y in zip(all_x, all_y):
                    poly.extend([float(x), float(y)])
                    
                coco_annotations.append({
                    "id": anno_id,
                    "image_id": img_id,
                    "category_id": cat_to_id.get(c, 0),
                    "segmentation": [poly],
                    "area": area,
                    "bbox": [x_min, y_min, w, h],
                    "iscrowd": 0
                })
                anno_id += 1
                
            img_id += 1
            
        coco_dict = {
            "info": {
                "description": f"VehiDE Dataset in COCO Format ({split_name})",
                "version": "1.0",
                "year": 2026
            },
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": categories
        }
        
        out_path = ANNOTATIONS_DIR / coco_file
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(coco_dict, f)
            
        summary[split_name] = {
            "images": len(coco_images),
            "annotations": len(coco_annotations),
            "output": str(out_path)
        }
        print(f"Saved {len(coco_images)} images and {len(coco_annotations)} annotations to {out_path}")
        
    return summary


def prepare_severity_metadata():
    """Generates clean tabular split index for data3a severity dataset."""
    print(">>> 3. Preparing severity classification dataset index...")
    base_dir = DATASETS_DIR / "data3a"
    classes = {"01-minor": 0, "02-moderate": 1, "03-severe": 2}
    
    rows = []
    for split_dir, split_name in [("training", "train"), ("validation", "val")]:
        d = base_dir / split_dir
        if not d.exists():
            continue
        for class_folder, class_id in classes.items():
            cdir = d / class_folder
            if not cdir.exists():
                continue
            for img_file in cdir.iterdir():
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    rows.append({
                        "image_id": img_file.stem,
                        "file_name": img_file.name,
                        "relative_path": str(img_file.relative_to(ROOT)).replace("\\", "/"),
                        "split": split_name,
                        "severity_label": class_folder.split("-")[1],
                        "severity_class_id": class_id
                    })
                
    df = pd.DataFrame(rows)
    out_csv = PROCESSED_DIR / "severity_metadata.csv"
    df.to_csv(out_csv, index=False)
    print(f"Severity metadata written to {out_csv} ({len(df)} rows)")
    return len(df)


def prepare_clean_copart_dataset():
    """Cleans, filters, and prepares Copart repair cost dataset for cost model training."""
    print(">>> 4. Cleaning and preparing Copart repair cost dataset...")
    csv_dir = DATASETS_DIR / "copart-dataset-main" / "auction-listings" / "data"
    csv_files = sorted(glob.glob(str(csv_dir / "*.csv")))
    
    dfs = []
    for f in csv_files:
        dfs.append(pd.read_csv(f, low_memory=False))
        
    combined = pd.concat(dfs, ignore_index=True)
    unique = combined.drop_duplicates(subset=["lotId"]).copy()
    
    # Cast and filter repairCost
    unique["repairCost"] = pd.to_numeric(unique["repairCost"], errors="coerce")
    unique["year"] = pd.to_numeric(unique["year"], errors="coerce")
    unique["mileage"] = pd.to_numeric(unique["mileage"], errors="coerce")
    
    # Filter valid repair costs: positive, reasonable realistic bounds
    clean = unique[
        (unique["repairCost"] >= 100.0) & 
        (unique["repairCost"] <= 150000.0) &
        (unique["year"] >= 1990) &
        (unique["year"] <= 2026) &
        (unique["damageDescription"].notna())
    ].copy()
    
    # Feature engineering
    clean["vehicle_age"] = 2026 - clean["year"]
    clean["log_repair_cost"] = np.log1p(clean["repairCost"])
    
    # Keep key features
    keep_cols = [
        "lotId", "year", "vehicle_age", "make", "modelGroup", "bodyStyle",
        "damageDescription", "secondaryDamage", "drivetrain", "transmission",
        "fuelType", "runsDrives", "repairCost", "log_repair_cost"
    ]
    existing_cols = [c for c in keep_cols if c in clean.columns]
    df_final = clean[existing_cols].copy()
    
    # Stratified train/val/test split (70% train, 15% val, 15% test)
    np.random.seed(42)
    shuffled = df_final.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    
    shuffled["split"] = "train"
    shuffled.loc[n_train:n_train+n_val, "split"] = "val"
    shuffled.loc[n_train+n_val:, "split"] = "test"
    
    out_csv = PROCESSED_DIR / "copart_repair_cost_clean.csv"
    shuffled.to_csv(out_csv, index=False)
    print(f"Clean Copart dataset written to {out_csv} ({len(shuffled)} rows, 70/15/15 split)")
    return len(shuffled)


def main():
    print("=== STARTING PHASE 3: DATA PREPARATION ===")
    w = compute_class_weights()
    coco = convert_via_to_coco()
    sev_count = prepare_severity_metadata()
    copart_count = prepare_clean_copart_dataset()
    print("=== PHASE 3 DATA PREPARATION COMPLETE ===")


if __name__ == "__main__":
    main()
