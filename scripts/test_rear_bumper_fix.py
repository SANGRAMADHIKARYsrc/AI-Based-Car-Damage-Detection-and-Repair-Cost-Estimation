import cv2
import os
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(REPO_ROOT / "evaluation" / "component" / "component_human_validation_150.csv")
img_dir = REPO_ROOT / "data" / "Datasets" / "test" / "images"

rb_cases = df[df["ground_truth_component"] == "rear_bumper"]
print("Rear bumper ground truth cases:")
for idx, r in rb_cases.iterrows():
    p = img_dir / r["image_file"]
    img = cv2.imread(str(p))
    coords = [float(x) for x in r["bbox_norm_xc_yc_w_h"].split(",")]
    xc, yc, bw, bh = coords
    h, w, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    red_mask = red1 | red2
    upper_h = int(yc * h)
    upper_red = np.sum(red_mask[:upper_h, :]) / max(1, upper_h * w)
    print(f"Sample {r['sample_id']:3d} ({r['image_file']}): yc={yc:.2f}, ar={bw/bh:.2f}, upper_red={upper_red*100:.2f}%")

print("\nFront bumper ground truth cases:")
fb_cases = df[df["ground_truth_component"] == "front_bumper"]
for idx, r in fb_cases.iterrows():
    p = img_dir / r["image_file"]
    img = cv2.imread(str(p))
    coords = [float(x) for x in r["bbox_norm_xc_yc_w_h"].split(",")]
    xc, yc, bw, bh = coords
    h, w, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    red_mask = red1 | red2
    upper_h = int(yc * h)
    upper_red = np.sum(red_mask[:upper_h, :]) / max(1, upper_h * w)
    print(f"Sample {r['sample_id']:3d} ({r['image_file']}): yc={yc:.2f}, ar={bw/bh:.2f}, upper_red={upper_red*100:.2f}%")
