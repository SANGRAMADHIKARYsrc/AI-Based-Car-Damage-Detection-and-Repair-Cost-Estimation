import cv2
import os
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(REPO_ROOT / "evaluation" / "component" / "component_human_validation_150.csv")
img_dir = REPO_ROOT / "data" / "Datasets" / "test" / "images"

print("--- Front Bumper Samples: True Red % ---")
fb_samples = [28, 35, 42, 59, 83, 107, 148, 149]
for sid in fb_samples:
    r = df[df["sample_id"] == sid].iloc[0]
    img = cv2.imread(str(img_dir / r["image_file"]))
    h, w, _ = img.shape
    coords = [float(x) for x in r["bbox_norm_xc_yc_w_h"].split(",")]
    yc = coords[1]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 100, 70), (8, 255, 255))
    m2 = cv2.inRange(hsv, (172, 100, 70), (180, 255, 255))
    red_mask = m1 | m2
    upper_h = max(1, int(yc * h))
    true_pct = np.mean(red_mask[:upper_h, :] > 0) * 100
    print(f"Sample {sid:3d} ({r['image_file']}): true_red_pct = {true_pct:.3f}%")

print("\n--- Rear Bumper Samples: True Red % ---")
rb_samples = [17, 26, 27, 53, 54, 79, 90, 111, 112, 117, 132]
for sid in rb_samples:
    r = df[df["sample_id"] == sid].iloc[0]
    img = cv2.imread(str(img_dir / r["image_file"]))
    h, w, _ = img.shape
    coords = [float(x) for x in r["bbox_norm_xc_yc_w_h"].split(",")]
    yc = coords[1]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 100, 70), (8, 255, 255))
    m2 = cv2.inRange(hsv, (172, 100, 70), (180, 255, 255))
    red_mask = m1 | m2
    upper_h = max(1, int(yc * h))
    true_pct = np.mean(red_mask[:upper_h, :] > 0) * 100
    print(f"Sample {sid:3d} ({r['image_file']}): true_red_pct = {true_pct:.3f}%")
