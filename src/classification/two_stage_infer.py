"""Phase 5 — Two-Stage Hierarchical Inference Pipeline.

Connects Phase 4 YOLOv8n detector with Phase 5 EfficientNet-B0 classifier:
1. Full vehicle image -> YOLO detector localizes damage ROIs
2. Bounding box crops extracted with context margin padding
3. High-resolution crop classifier refines damage type and probability
4. Output annotated images showing one-stage vs two-stage predictions
"""

import argparse
from pathlib import Path
import sys
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.dataset import CLASS_NAMES, get_transforms
from src.classification.model import DamageClassifier


class TwoStageDamageDetector:
    """Two-stage damage pipeline: YOLO Localization -> CNN Fine-grained Classification."""

    def __init__(
        self,
        detector_weights: Path,
        classifier_weights: Path,
        device: str = "cuda",
        conf_threshold: float = 0.25,
        padding_ratio: float = 0.15,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.padding_ratio = padding_ratio
        self.conf_threshold = conf_threshold

        # 1. Load Detector
        print(f"[Two-Stage Pipeline] Loading YOLO detector: {detector_weights}")
        self.detector = YOLO(str(detector_weights))

        # 2. Load Classifier
        print(f"[Two-Stage Pipeline] Loading Crop Classifier: {classifier_weights}")
        ckpt = torch.load(classifier_weights, map_location=self.device, weights_only=False)
        backbone = ckpt.get("config", {}).get("model", {}).get("backbone", "efficientnet_b0")
        img_size = ckpt.get("config", {}).get("training", {}).get("image_size", 224)

        self.classifier = DamageClassifier(
            backbone_name=backbone,
            num_classes=len(CLASS_NAMES),
            pretrained=False,
        ).to(self.device)
        self.classifier.load_state_dict(ckpt["model_state_dict"])
        self.classifier.eval()

        self.transform = get_transforms(img_size=img_size, is_train=False)

    def predict(self, image_input: Union[Path, str, np.ndarray]) -> Tuple[np.ndarray, List[dict]]:
        """Run two-stage detection and classification on single image."""
        if isinstance(image_input, np.ndarray):
            img = image_input.copy()
        else:
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Could not load image: {image_input}")

        orig_h, orig_w = img.shape[:2]
        annotated = img.copy()

        # Step 1: YOLO Detection
        results = self.detector.predict(
            source=img,
            conf=self.conf_threshold,
            device=0 if self.device.type == "cuda" else "cpu",
            verbose=False,
        )[0]

        detections = []
        boxes = results.boxes

        for i in range(len(boxes)):
            box = boxes[i]
            xyxy = box.xyxy[0].cpu().numpy()
            det_conf = float(box.conf[0].cpu().numpy())
            det_cls_id = int(box.cls[0].cpu().numpy())
            det_cls_name = results.names.get(det_cls_id, str(det_cls_id))

            x1, y1, x2, y2 = xyxy
            bw = x2 - x1
            bh = y2 - y1
            pad_w = bw * self.padding_ratio
            pad_h = bh * self.padding_ratio

            cx1 = max(0, int(x1 - pad_w))
            cy1 = max(0, int(y1 - pad_h))
            cx2 = min(orig_w, int(x2 + pad_w))
            cy2 = min(orig_h, int(y2 + pad_h))

            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
                continue

            # Step 2: Classifier Refinement
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            crop_t = self.transform(crop_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu"):
                    probs = self.classifier.predict_proba(crop_t)[0].float().cpu().numpy()

            clf_cls_id = int(np.argmax(probs))
            clf_conf = float(probs[clf_cls_id])
            clf_cls_name = CLASS_NAMES[clf_cls_id]

            record = {
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "detector_class": det_cls_name,
                "detector_conf": round(det_conf, 3),
                "classifier_class": clf_cls_name,
                "classifier_conf": round(clf_conf, 3),
                "agreement": (det_cls_name == clf_cls_name),
            }
            detections.append(record)

            # Draw visualization
            color = (0, 200, 0) if record["agreement"] else (0, 165, 255)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            label_str = f"Det: {det_cls_name} ({det_conf:.2f}) | Refined: {clf_cls_name} ({clf_conf:.2f})"
            (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(
                annotated,
                (int(x1), max(0, int(y1) - 22)),
                (int(x1) + tw + 4, max(0, int(y1))),
                (30, 30, 30),
                -1,
            )
            cv2.putText(
                annotated,
                label_str,
                (int(x1) + 2, max(14, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated, detections


def main():
    parser = argparse.ArgumentParser(description="Two-Stage Damage Detection + Classification")
    parser.add_argument("--detector", type=str, default="models/detection/phase4_yolov8n_baseline/weights/best.pt")
    parser.add_argument("--classifier", type=str, default="models/classification/phase5_damage_classifier/best_classifier.pt")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    detector_path = PROJECT_ROOT / args.detector
    classifier_path = PROJECT_ROOT / args.classifier

    pipeline = TwoStageDamageDetector(
        detector_weights=detector_path,
        classifier_weights=classifier_path,
        conf_threshold=args.conf,
    )

    test_img_dir = PROJECT_ROOT / "data" / "Datasets" / "test" / "images"
    test_imgs = sorted(list(test_img_dir.glob("*.jpg")))[: args.num_samples]

    out_dir = PROJECT_ROOT / "evaluation" / "classification" / "two_stage_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Two-Stage Pipeline] Processing {len(test_imgs)} test images -> {out_dir}")
    summary = []
    for img_path in test_imgs:
        annotated, dets = pipeline.predict(img_path)
        out_path = out_dir / f"two_stage_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)
        print(f"  {img_path.name}: {len(dets)} damages detected.")
        for d in dets:
            status = "MATCH" if d["agreement"] else "REFINED"
            print(f"    [{status}] Det={d['detector_class']} ({d['detector_conf']}) -> Clf={d['classifier_class']} ({d['classifier_conf']})")
        summary.append({"image": img_path.name, "detections": dets})

    summary_path = out_dir / "two_stage_summary.json"
    import json
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Two-Stage Pipeline] Complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
