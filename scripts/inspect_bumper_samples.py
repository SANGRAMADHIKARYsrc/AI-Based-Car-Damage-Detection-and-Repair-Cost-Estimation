import cv2
import os
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
csv_path = REPO_ROOT / "evaluation" / "component" / "component_human_validation_150.csv"
img_dir = REPO_ROOT / "data" / "Datasets" / "test" / "images"

df = pd.read_csv(csv_path)

print(f"Total validation instances: {len(df)}")

# Analyze each instance
results = []
for idx, r in df.iterrows():
    img_path = img_dir / r['image_file']
    img = cv2.imread(str(img_path))
    h, w, _ = img.shape
    coords = [float(x) for x in r['bbox_norm_xc_yc_w_h'].split(',')]
    xc, yc, bw, bh = coords
    
    # Compute image features
    # Check if there's taillight red at top or sides of bumper
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    red_mask = red1 | red2
    
    # Crop damage region with 20% margin
    x1 = max(0, int((xc - bw * 0.6) * w))
    y1 = max(0, int((yc - bh * 0.6) * h))
    x2 = min(w, int((xc + bw * 0.6) * w))
    y2 = min(h, int((yc + bh * 0.6) * h))
    crop = img[y1:y2, x1:x2]
    
    # Upper region red (taillight area above rear bumper)
    upper_y = int(yc * h)
    upper_red = np.sum(red_mask[:upper_y, :]) / max(1, upper_y * w)
    
    results.append({
        "sample_id": r["sample_id"],
        "image": r["image_file"],
        "damage": r["damage_type"],
        "pred": r["predicted_component"],
        "xc": xc,
        "yc": yc,
        "bw": bw,
        "bh": bh,
        "upper_red_pct": round(upper_red * 100, 2),
    })

res_df = pd.DataFrame(results)
print("\n--- All 26 Front Bumper Predictions ---")
fb = res_df[res_df["pred"] == "front_bumper"]
for _, row in fb.iterrows():
    print(f"Sample {row['sample_id']:3d}: {row['image']} | {row['damage']} | xc={row['xc']:.2f}, yc={row['yc']:.2f}, bw={row['bw']:.2f}, bh={row['bh']:.2f} | upper_red={row['upper_red_pct']}%")
