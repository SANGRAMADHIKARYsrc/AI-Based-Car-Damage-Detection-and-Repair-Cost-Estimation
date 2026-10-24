import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ultralytics import YOLO

from src.component.identifier import VehicleComponentIdentifier
from src.severity.dsi import compute_instance_dsi

REPO_ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = REPO_ROOT / "data" / "Datasets" / "data3a" / "validation"
MODEL_PATH = REPO_ROOT / "models" / "detection" / "phase4_yolov8n_baseline" / "weights" / "best.pt"


def main():
    if not MODEL_PATH.exists():
        print(f"Detector not found at {MODEL_PATH}")
        return

    print("Loading YOLO detector...")
    model = YOLO(str(MODEL_PATH))
    identifier = VehicleComponentIdentifier()

    classes = [("01-minor", "minor"), ("02-moderate", "moderate"), ("03-severe", "severe")]

    records = []
    print("Running detector & DSI over data3a validation set...")
    for folder_name, true_sev in classes:
        img_paths = list((VAL_DIR / folder_name).glob("*.*"))
        for p in img_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Run YOLO
            res = model.predict(img, conf=0.25, verbose=False)[0]
            boxes = res.boxes

            if len(boxes) == 0:
                pred_sev = "minor"
                max_dsi = 20.0
                num_det = 0
            else:
                num_det = len(boxes)
                dsi_scores = []
                for box in boxes:
                    cls_id = int(box.cls[0])
                    dmg_type = model.names[cls_id]
                    b = box.xywhn[0].cpu().numpy()
                    xc, yc, bw, bh = b[0], b[1], b[2], b[3]
                    area = float(bw * bh)
                    comp_res = identifier.identify_component(
                        damage_class=dmg_type,
                        norm_bbox=(xc, yc, bw, bh),
                        image=img,
                    )
                    dsi_res = compute_instance_dsi(
                        damage_type=dmg_type,
                        component=comp_res["primary_component"],
                        norm_area=area,
                        confidence=float(box.conf[0]),
                    )
                    dsi_scores.append(dsi_res["dsi_score"])

                max_dsi = max(dsi_scores)
                if max_dsi < 38.0:
                    pred_sev = "minor"
                elif max_dsi < 72.0:
                    pred_sev = "moderate"
                else:
                    pred_sev = "severe"

            records.append({
                "path": str(p.name),
                "true_severity": true_sev,
                "num_detections": num_det,
                "max_dsi": round(max_dsi, 1),
                "predicted_severity": pred_sev,
            })

    df = pd.DataFrame(records)
    print(f"\nEvaluated {len(df)} images.")
    print(f"Images with 0 detections: {(df['num_detections'] == 0).sum()} ({(df['num_detections'] == 0).mean()*100:.1f}%)")

    y_true = df["true_severity"].tolist()
    y_pred = df["predicted_severity"].tolist()

    acc = accuracy_score(y_true, y_pred)
    labels = ["minor", "moderate", "severe"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"True: {l}" for l in labels], columns=[f"Pred: {l}" for l in labels])

    print(f"\nOverall Agreement (Accuracy): {acc*100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm_df)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))


if __name__ == "__main__":
    main()
