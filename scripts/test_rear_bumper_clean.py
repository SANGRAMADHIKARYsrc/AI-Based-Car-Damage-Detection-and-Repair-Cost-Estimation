import cv2
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))
df = pd.read_csv(REPO_ROOT / "evaluation" / "component" / "component_human_validation_150.csv")
img_dir = REPO_ROOT / "data" / "Datasets" / "test" / "images"

from src.component.identifier import VehicleComponentIdentifier
identifier = VehicleComponentIdentifier()

preds = []
for idx, r in df.iterrows():
    dmg = r["damage_type"]
    coords = [float(x) for x in r["bbox_norm_xc_yc_w_h"].split(",")]
    norm_bbox = (coords[0], coords[1], coords[2], coords[3])
    
    # Original prediction
    res = identifier.identify_component(dmg, norm_bbox)
    comp = res["primary_component"]
    
    # Targeted bumper fix:
    if comp in ["front_bumper", "rear_bumper"]:
        img_path = img_dir / r["image_file"]
        img = cv2.imread(str(img_path))
        h, w, _ = img.shape
        yc = coords[1]
        
        # Check red taillight presence in upper area above bumper
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
        red_mask = red1 | red2
        upper_h = max(1, int(yc * h))
        upper_red_pct = np.sum(red_mask[:upper_h, :]) / (upper_h * w) * 100
        
        # If upper area has red taillight pixels, it's a rear bumper!
        if upper_red_pct > 2.0:
            comp = "rear_bumper"
            
    preds.append(comp)

y_true = df["ground_truth_component"].tolist()
acc = accuracy_score(y_true, preds)
macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
weighted_f1 = f1_score(y_true, preds, average="weighted", zero_division=0)

print(f"Targeted Fix Accuracy: {acc*100:.2f}% (was 84.67%)")
print(f"Targeted Fix Macro F1: {macro_f1:.4f} (was 0.7943)")

labels = sorted(list(set(y_true + preds)))
cm = confusion_matrix(y_true, preds, labels=labels)
cm_df = pd.DataFrame(cm, index=["True: " + l for l in labels], columns=["Pred: " + l for l in labels])
print("\nConfusion Matrix:")
print(cm_df)
print("\nClassification Report:")
print(classification_report(y_true, preds, labels=labels, zero_division=0))
