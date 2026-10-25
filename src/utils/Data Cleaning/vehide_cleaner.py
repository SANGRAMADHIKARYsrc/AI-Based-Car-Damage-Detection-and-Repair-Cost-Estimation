"""VehiDE VIA Annotation Cleaner & COCO Converter.

Cleans and converts VehiDE VIA polygon annotations:
- Filters out degenerate polygons (< 3 points or mismatched x, y lengths).
- Calculates polygon area via the Shoelace formula.
- Computes tight bounding boxes [x_min, y_min, width, height].
- Maps Vietnamese label strings to English CarDD canonical taxonomy.
- Emits standardized COCO JSON formatted dataset annotations.
"""

from pathlib import Path
import json
import yaml
import numpy as np


def polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates polygon area using the Shoelace formula (Gauss's area formula)."""
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def clean_and_convert_vehide(datasets_dir: Path, mapping_file: Path, output_dir: Path) -> dict:
    """Clean VehiDE annotations and convert to COCO format.

    Args:
        datasets_dir: Path where 0Train_via_annos.json and 0Val_via_annos.json reside.
        mapping_file: Path to class_mapping.yaml.
        output_dir: Path where instances_vehide_train.json and val.json will be saved.

    Returns:
        dict: Summary of cleaned images and annotations per split.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping_data = yaml.safe_load(f)

    vehide_classes = mapping_data.get("vehide_classes", {})
    categories = []
    cat_to_id = {}

    for idx, (cat_name, meta) in enumerate(vehide_classes.items(), start=1):
        cat_to_id[cat_name] = idx
        categories.append({
            "id": idx,
            "name": cat_name,
            "supercategory": "vehicle_damage",
            "english_name": meta.get("english", cat_name),
            "cardd_mapped_name": meta.get("cardd_name", "unknown"),
        })

    splits = [
        ("0Train_via_annos.json", "instances_vehide_train.json", "train"),
        ("0Val_via_annos.json", "instances_vehide_val.json", "val"),
    ]

    summary = {}

    for via_name, coco_name, split_name in splits:
        via_path = datasets_dir / via_name
        if not via_path.exists():
            continue

        with open(via_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        coco_images = []
        coco_annotations = []
        anno_id = 1
        img_id = 1
        degenerate_polys = 0

        for file_key, record in raw_data.items():
            img_filename = record.get("name", file_key)
            regions = record.get("regions", [])

            coco_images.append({
                "id": img_id,
                "file_name": img_filename,
                "width": 1024,
                "height": 768,
            })

            for r in regions:
                all_x = r.get("all_x", [])
                all_y = r.get("all_y", [])
                cls_str = r.get("class", "unknown")

                # Filter out degenerate polygons
                if len(all_x) < 3 or len(all_y) < 3 or len(all_x) != len(all_y):
                    degenerate_polys += 1
                    continue

                x_arr = np.array(all_x, dtype=float)
                y_arr = np.array(all_y, dtype=float)

                x_min = float(np.min(x_arr))
                y_min = float(np.min(y_arr))
                w = float(np.max(x_arr) - x_min)
                h = float(np.max(y_arr) - y_min)

                if w <= 0 or h <= 0:
                    degenerate_polys += 1
                    continue

                area = polygon_area(x_arr, y_arr)

                # Flatten coordinates for COCO: [x1, y1, x2, y2, ...]
                poly = []
                for px, py in zip(all_x, all_y):
                    poly.extend([float(px), float(py)])

                coco_annotations.append({
                    "id": anno_id,
                    "image_id": img_id,
                    "category_id": cat_to_id.get(cls_str, 0),
                    "segmentation": [poly],
                    "area": area,
                    "bbox": [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)],
                    "iscrowd": 0,
                })
                anno_id += 1

            img_id += 1

        coco_payload = {
            "info": {
                "description": f"VehiDE Cleaned Dataset in COCO Format ({split_name})",
                "version": "1.0",
                "year": 2026,
            },
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": categories,
        }

        out_file = output_dir / coco_name
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(coco_payload, f)

        summary[split_name] = {
            "valid_images": len(coco_images),
            "valid_annotations": len(coco_annotations),
            "filtered_degenerate_polygons": degenerate_polys,
            "output_file": str(out_file),
        }

    return summary


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    d_dir = repo_root / "data" / "Datasets"
    map_f = repo_root / "data" / "class_mapping.yaml"
    out_d = repo_root / "data" / "annotations"
    if d_dir.exists() and map_f.exists():
        res = clean_and_convert_vehide(d_dir, map_f, out_d)
        print("VehiDE cleaning complete:", res)
    else:
        print("Required VehiDE paths not found.")
