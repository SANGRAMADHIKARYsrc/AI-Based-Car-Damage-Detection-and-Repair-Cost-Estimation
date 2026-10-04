"""Phase 6 — Multi-Layer Damage & Component Visualizer.

Demonstrates the 3-stage visual pipeline:
1. YOLOv8n damage localization
2. EfficientNet-B0 fine-grained damage classification
3. Spatial-affinity vehicle component identification
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.dataset import CLASS_NAMES, get_transforms
from src.classification.model import DamageClassifier
from src.component import VehicleComponentIdentifier


def run_pipeline_visualization(num_samples: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Detector
    det_weights = PROJECT_ROOT / "models" / "detection" / "phase4_yolov8n_baseline" / "weights" / "best.pt"
    detector = YOLO(str(det_weights))

    # 2. Classifier
    clf_weights = PROJECT_ROOT / "models" / "classification" / "phase5_damage_classifier" / "best_classifier.pt"
    ckpt = torch.load(clf_weights, map_location=device, weights_only=False)
    classifier = DamageClassifier(backbone_name="efficientnet_b0", num_classes=6, pretrained=False).to(device)
    classifier.load_state_dict(ckpt["model_state_dict"])
    classifier.eval()
    transform = get_transforms(img_size=224, is_train=False)

    # 3. Component Identifier
    component_id = VehicleComponentIdentifier()

    test_imgs = sorted(list((PROJECT_ROOT / "data" / "Datasets" / "test" / "images").glob("*.jpg")))[:num_samples]
    out_dir = PROJECT_ROOT / "evaluation" / "component" / "annotated_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    print(f"\n[Phase 6 Visualizer] Running 3-stage inference on {len(test_imgs)} test images...")

    for img_path in test_imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]
        annotated = img.copy()

        # Step 1: Detect
        results = detector.predict(source=img, conf=0.25, verbose=False)[0]
        image_records = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            det_conf = float(box.conf[0].cpu().numpy())
            det_cls_id = int(box.cls[0].cpu().numpy())
            det_name = results.names.get(det_cls_id, str(det_cls_id))

            # Step 2: Crop & Classify
            pad_w = (x2 - x1) * 0.15
            pad_h = (y2 - y1) * 0.15
            cx1 = max(0, int(x1 - pad_w))
            cy1 = max(0, int(y1 - pad_h))
            cx2 = min(orig_w, int(x2 + pad_w))
            cy2 = min(orig_h, int(y2 + pad_h))

            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
                continue

            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            crop_t = transform(crop_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = classifier.predict_proba(crop_t)[0].cpu().numpy()

            clf_id = int(np.argmax(probs))
            final_damage = CLASS_NAMES[clf_id]
            damage_conf = float(probs[clf_id])

            # Step 3: Identify Component
            norm_bbox = (
                ((x1 + x2) / 2.0) / orig_w,
                ((y1 + y2) / 2.0) / orig_h,
                (x2 - x1) / orig_w,
                (y2 - y1) / orig_h,
            )
            comp_res = component_id.identify_component(
                damage_class=final_damage,
                norm_bbox=norm_bbox,
                detector_conf=damage_conf,
            )

            primary_comp = comp_res["primary_component"]
            comp_conf = comp_res["confidence"]
            copart_sec = comp_res["copart_sector"]

            rec = {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "damage_type": final_damage,
                "damage_conf": round(damage_conf, 3),
                "component": primary_comp,
                "component_conf": round(comp_conf, 3),
                "copart_sector": copart_sec,
                "cost_multiplier": comp_res["cost_multiplier"],
            }
            image_records.append(rec)

            # Draw 3-layer annotation box
            color = (36, 179, 83) if comp_res["method"] == "deterministic_semantic" else (235, 140, 52)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            label_top = f"[{primary_comp}] {final_damage} ({damage_conf:.2f})"
            label_bot = f"Sector: {copart_sec} (x{comp_res['cost_multiplier']})"

            # Top label
            (tw1, th1), _ = cv2.getTextSize(label_top, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(annotated, (int(x1), max(0, int(y1) - 34)), (int(x1) + max(tw1, 160) + 4, max(0, int(y1))), (25, 25, 25), -1)
            cv2.putText(annotated, label_top, (int(x1) + 2, max(14, int(y1) - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated, label_bot, (int(x1) + 2, max(26, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1, cv2.LINE_AA)

        out_img_path = out_dir / f"component_{img_path.name}"
        cv2.imwrite(str(out_img_path), annotated)
        summary.append({"image": img_path.name, "damages": image_records})
        print(f"  {img_path.name}: {len(image_records)} damage/component instances labeled.")

    sum_path = out_dir / "component_visualization_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Phase 6 Visualizer] Finished. Annotated samples saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10)
    args = parser.parse_args()
    run_pipeline_visualization(args.num_samples)
