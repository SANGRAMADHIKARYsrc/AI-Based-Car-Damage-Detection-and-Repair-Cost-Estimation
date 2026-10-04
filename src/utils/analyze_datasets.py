"""Dataset Analysis and Audit Script for Car Damage AI (Phase 2)
Analyzes:
1. CarDD YOLO dataset (images, labels, boxes, aspect ratios, areas)
2. Severity dataset (data3a: 3 classes - minor, moderate, severe)
3. VehiDE VIA JSONs (polygons, 7 classes)
4. Copart auction repair cost dataset (tabular repair costs, vehicle features)
5. Comprehensive data leakage and cross-split duplicate checks
"""

import os
import glob
import json
import hashlib
from collections import defaultdict
from PIL import Image
import pandas as pd
import numpy as np

def get_image_info(path):
    try:
        with Image.open(path) as img:
            return {
                "valid": True,
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode
            }
    except Exception as e:
        return {"valid": False, "error": str(e)}

def compute_file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def analyze_cardd_yolo(base_dir):
    print(">>> Analyzing CarDD YOLO Dataset...")
    splits = ["train", "val", "test"]
    class_names = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
    stats = {}
    hashes_by_split = defaultdict(dict)
    
    for split in splits:
        img_dir = os.path.join(base_dir, split, "images")
        lbl_dir = os.path.join(base_dir, split, "labels")
        
        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        lbl_files = sorted([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])
        
        img_stems = {os.path.splitext(f)[0]: f for f in img_files}
        lbl_stems = {os.path.splitext(f)[0]: f for f in lbl_files}
        
        missing_labels = [f for stem, f in img_stems.items() if stem not in lbl_stems]
        missing_images = [f for stem, f in lbl_stems.items() if stem not in img_stems]
        
        resolutions = []
        corrupted = []
        for img_name in img_files:
            p = os.path.join(img_dir, img_name)
            info = get_image_info(p)
            if info["valid"]:
                resolutions.append((info["width"], info["height"]))
                file_hash = compute_file_hash(p)
                hashes_by_split[split][img_name] = file_hash
            else:
                corrupted.append((img_name, info["error"]))
                
        class_counts = defaultdict(int)
        instances_per_image = []
        bbox_areas = []
        bbox_aspect_ratios = []
        invalid_bboxes = 0
        total_boxes = 0
        
        for lbl_name in lbl_files:
            p = os.path.join(lbl_dir, lbl_name)
            with open(p, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            
            instances_per_image.append(len(lines))
            total_boxes += len(lines)
            
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    class_counts[cls_id] += 1
                    
                    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        invalid_bboxes += 1
                    
                    bbox_areas.append(w * h)
                    if h > 0:
                        bbox_aspect_ratios.append(w / h)
                        
        stats[split] = {
            "image_count": len(img_files),
            "label_count": len(lbl_files),
            "missing_labels": missing_labels,
            "missing_images": missing_images,
            "corrupted_images": corrupted,
            "total_instances": total_boxes,
            "mean_instances_per_img": round(total_boxes / max(len(lbl_files), 1), 2),
            "class_distribution": {class_names[k]: class_counts[k] for k in range(len(class_names))},
            "invalid_bboxes": invalid_bboxes,
            "resolutions_sample": list(set(resolutions))[:5],
            "common_resolution": max(set(resolutions), key=resolutions.count) if resolutions else None,
            "mean_bbox_area": round(sum(bbox_areas) / len(bbox_areas), 4) if bbox_areas else 0,
            "median_bbox_area": round(sorted(bbox_areas)[len(bbox_areas)//2], 4) if bbox_areas else 0,
            "min_bbox_area": round(min(bbox_areas), 6) if bbox_areas else 0,
            "max_bbox_area": round(max(bbox_areas), 4) if bbox_areas else 0
        }
        
    return stats, hashes_by_split

def analyze_severity_data3a(base_dir):
    print(">>> Analyzing Severity Dataset (data3a)...")
    splits = ["training", "validation"]
    classes = ["01-minor", "02-moderate", "03-severe"]
    stats = {}
    hashes_by_split = defaultdict(dict)
    
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue
        stats[split] = {}
        total = 0
        resolutions = []
        corrupted = []
        
        for c in classes:
            c_dir = os.path.join(split_dir, c)
            if not os.path.exists(c_dir):
                stats[split][c] = 0
                continue
            files = [f for f in os.listdir(c_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            stats[split][c] = len(files)
            total += len(files)
            
            for f in files:
                p = os.path.join(c_dir, f)
                info = get_image_info(p)
                if info["valid"]:
                    resolutions.append((info["width"], info["height"]))
                    hashes_by_split[split][f] = compute_file_hash(p)
                else:
                    corrupted.append((f, info["error"]))
                    
        stats[split]["total"] = total
        stats[split]["corrupted"] = corrupted
        stats[split]["common_res"] = max(set(resolutions), key=resolutions.count) if resolutions else None
        
    return stats, hashes_by_split

def analyze_via_json(base_dir):
    print(">>> Analyzing VehiDE VIA JSON Annotations...")
    via_files = ["0Train_via_annos.json", "0Val_via_annos.json"]
    stats = {}
    
    # Vietnamese to English translation mapping for VehiDE classes
    trans_map = {
        "tray_son": "scratch (tray_son)",
        "mop_lom": "dent (mop_lom)",
        "rach": "tear/crack (rach)",
        "be_den": "broken lamp (be_den)",
        "vo_kinh": "shattered glass (vo_kinh)",
        "thung": "puncture/hole (thung)",
        "mat_bo_phan": "missing part (mat_bo_phan)"
    }
    
    for vf in via_files:
        p = os.path.join(base_dir, vf)
        if not os.path.exists(p):
            continue
        print(f"Reading {vf}...")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        num_images = len(data)
        num_regions = 0
        cat_counts = defaultdict(int)
        
        for k, v in data.items():
            regions = v.get("regions", [])
            num_regions += len(regions)
            for r in regions:
                c = r.get("class") or "unknown"
                label = trans_map.get(c, c)
                cat_counts[label] += 1
                
        stats[vf] = {
            "image_count": num_images,
            "polygon_region_count": num_regions,
            "mean_regions_per_img": round(num_regions / max(num_images, 1), 2),
            "damage_classes": dict(cat_counts)
        }
    return stats

def analyze_copart_dataset(base_dir):
    print(">>> Analyzing Copart Auction Dataset...")
    csv_dir = os.path.join(base_dir, "copart-dataset-main", "auction-listings", "data")
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    if not csv_files:
        return {"found": False, "message": "No CSV files found"}
        
    print(f"Loading {len(csv_files)} daily snapshot CSV files...")
    dfs = []
    for f in csv_files:
        dfs.append(pd.read_csv(f, low_memory=False))
        
    combined = pd.concat(dfs, ignore_index=True)
    total_raw_rows = len(combined)
    unique_lots = combined.drop_duplicates(subset=["lotId"]).copy()
    
    repair_costs = pd.to_numeric(unique_lots["repairCost"], errors="coerce").dropna()
    valid_costs = repair_costs[repair_costs > 0]
    
    damage_desc_counts = unique_lots["damageDescription"].value_counts().head(10).to_dict()
    make_counts = unique_lots["make"].value_counts().head(10).to_dict()
    year_stats = {
        "min_year": int(unique_lots["year"].min()) if not unique_lots["year"].isna().all() else None,
        "max_year": int(unique_lots["year"].max()) if not unique_lots["year"].isna().all() else None,
        "median_year": int(unique_lots["year"].median()) if not unique_lots["year"].isna().all() else None
    }
    
    stats = {
        "total_snapshot_records": total_raw_rows,
        "unique_lots": len(unique_lots),
        "total_with_repair_cost": len(repair_costs),
        "total_nonzero_repair_cost": len(valid_costs),
        "repair_cost_stats": {
            "mean": round(float(valid_costs.mean()), 2) if len(valid_costs) > 0 else 0,
            "median": round(float(valid_costs.median()), 2) if len(valid_costs) > 0 else 0,
            "std": round(float(valid_costs.std()), 2) if len(valid_costs) > 0 else 0,
            "min": round(float(valid_costs.min()), 2) if len(valid_costs) > 0 else 0,
            "q25": round(float(valid_costs.quantile(0.25)), 2) if len(valid_costs) > 0 else 0,
            "q75": round(float(valid_costs.quantile(0.75)), 2) if len(valid_costs) > 0 else 0,
            "max": round(float(valid_costs.max()), 2) if len(valid_costs) > 0 else 0,
        },
        "top_damage_descriptions": damage_desc_counts,
        "top_makes": make_counts,
        "year_distribution": year_stats
    }
    return stats

def check_leakage(hashes_cardd, hashes_data3a):
    print(">>> Checking for data leakage and duplicates...")
    leakage = {}
    
    train_hashes = set(hashes_cardd.get("train", {}).values())
    val_hashes = set(hashes_cardd.get("val", {}).values())
    test_hashes = set(hashes_cardd.get("test", {}).values())
    
    leakage["cardd_train_val_overlap"] = len(train_hashes.intersection(val_hashes))
    leakage["cardd_train_test_overlap"] = len(train_hashes.intersection(test_hashes))
    leakage["cardd_val_test_overlap"] = len(val_hashes.intersection(test_hashes))
    
    for split in ["train", "val", "test"]:
        h_dict = hashes_cardd.get(split, {})
        unique_hashes = set(h_dict.values())
        leakage[f"cardd_{split}_exact_duplicates"] = len(h_dict) - len(unique_hashes)
        
    d3_train_hashes = set(hashes_data3a.get("training", {}).values())
    d3_val_hashes = set(hashes_data3a.get("validation", {}).values())
    leakage["data3a_train_val_overlap"] = len(d3_train_hashes.intersection(d3_val_hashes))
    
    for split in ["training", "validation"]:
        h_dict = hashes_data3a.get(split, {})
        unique_hashes = set(h_dict.values())
        leakage[f"data3a_{split}_exact_duplicates"] = len(h_dict) - len(unique_hashes)
        
    cardd_all = train_hashes | val_hashes | test_hashes
    d3_all = d3_train_hashes | d3_val_hashes
    leakage["cardd_data3a_overlap"] = len(cardd_all.intersection(d3_all))
    
    return leakage

def fix_data_yaml(base_dir):
    yaml_path = os.path.join(base_dir, "data.yaml")
    abs_base = os.path.abspath(base_dir).replace("\\", "/")
    # Standardize data.yaml to use relative paths from repo root or absolute local paths
    content = f"""# CarDD YOLO Dataset Configuration
# Standardized for local training in car-damage-ai project

path: {abs_base}
train: train/images
val: val/images
test: test/images

nc: 6
names: ['dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat']
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f">>> Updated data.yaml at {yaml_path}")

def main():
    root = r"c:\Users\ASUS\OneDrive\Desktop\Antigravity\Capstone Project\car-damage-ai"
    ds_dir = os.path.join(root, "data", "Datasets")
    
    fix_data_yaml(ds_dir)
    cardd_stats, cardd_hashes = analyze_cardd_yolo(ds_dir)
    data3a_stats, data3a_hashes = analyze_severity_data3a(os.path.join(ds_dir, "data3a"))
    via_stats = analyze_via_json(ds_dir)
    copart_stats = analyze_copart_dataset(ds_dir)
    leakage_stats = check_leakage(cardd_hashes, data3a_hashes)
    
    full_report = {
        "cardd_yolo": cardd_stats,
        "severity_data3a": data3a_stats,
        "via_polygons_vehide": via_stats,
        "copart_repair_cost": copart_stats,
        "leakage_and_duplicates": leakage_stats
    }
    
    out_dir = os.path.join(root, "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dataset_analysis_stats.json")
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
        
    print(f"\n[DONE] Full stats written to {out_path}")

if __name__ == "__main__":
    main()
