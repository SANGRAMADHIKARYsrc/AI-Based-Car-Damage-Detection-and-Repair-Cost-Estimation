"""Phase 4 — Prediction Visualization and Sample Inference

Runs inference on sample images and generates annotated visualizations
for qualitative assessment of the baseline detector.

Usage:
    python src/detection/predict_samples.py
    python src/detection/predict_samples.py --weights best.pt --source data/Datasets/test/images --n 20
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# BGR colors for each damage class (visually distinct)
CLASS_COLORS = {
    0: (0, 165, 255),    # dent — orange
    1: (0, 255, 0),      # scratch — green
    2: (0, 0, 255),      # crack — red
    3: (255, 0, 0),      # glass shatter — blue
    4: (255, 0, 255),    # lamp broken — magenta
    5: (0, 255, 255),    # tire flat — yellow
}

CLASS_NAMES = ['dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat']


def predict_and_visualize(args: argparse.Namespace):
    """Run inference on sample images, save annotated visualizations."""
    from ultralytics import YOLO

    output_dir = PROJECT_ROOT / "evaluation" / "detection" / "phase4_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path
    if not weights_path.exists():
        default = PROJECT_ROOT / "models" / "detection" / "phase4_yolov8n_baseline" / "weights" / "best.pt"
        if default.exists():
            weights_path = default

    model = YOLO(str(weights_path))

    # Collect source images
    source_dir = Path(args.source)
    if not source_dir.is_absolute():
        source_dir = PROJECT_ROOT / source_dir

    image_paths = sorted(source_dir.glob("*.*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

    if args.n > 0 and len(image_paths) > args.n:
        random.seed(42)
        image_paths = random.sample(image_paths, args.n)

    print(f"Running inference on {len(image_paths)} images...")

    for img_path in image_paths:
        results = model.predict(
            str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=0.7,
            verbose=False,
        )

        # Draw custom annotations
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        result = results[0]
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"

            # Bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Save
        out_path = output_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(out_path), img)

    print(f"Predictions saved to: {output_dir}")
    print(f"Total images processed: {len(image_paths)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4: Sample Predictions")
    parser.add_argument("--weights", type=str,
                        default="models/detection/phase4_yolov8n_baseline/weights/best.pt")
    parser.add_argument("--source", type=str,
                        default="data/Datasets/test/images",
                        help="Image directory for inference")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of sample images (0=all)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_and_visualize(args)
