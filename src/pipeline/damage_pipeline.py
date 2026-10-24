"""Phase 13 — Complete End-to-End Car Damage AI Pipeline.

Unifies all 5 specialized modules:
1. Stage 1: YOLOv8n Damage Localization
2. Stage 2: EfficientNet-B0 Damage Fine-Grained Classification
3. Stage 3: Automotive Component Ontology Identification
4. Stage 4: Damage Severity Assessment (ResNet18 & DSI)
5. Stage 5: Repair Cost & Uncertainty Estimation (HistGradientBoosting & Quantiles)

Produces comprehensive structured inspection reports with annotated visual overlays.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
from PIL import Image
import torch

from src.classification.two_stage_infer import TwoStageDamageDetector
from src.component.identifier import VehicleComponentIdentifier
from src.cost.cost_estimator import RepairCostEstimator
from src.severity.assessor import VehicleSeverityAssessor


class CarDamageAIPipeline:
    """Production End-to-End Car Damage Detection & Cost Estimation Pipeline."""

    def __init__(
        self,
        detector_weights: str = "models/detection/phase4_yolov8n_baseline/weights/best.pt",
        classifier_weights: str = "models/classification/phase5_damage_classifier/best_classifier.pt",
        severity_weights: str = "models/severity/phase7_severity_model/best_severity_model.pt",
        cost_model_path: str = "models/cost/best_cost_model.joblib",
        quantile_path: str = "models/cost/quantile_models.joblib",
        preprocessor_path: str = "data/processed/cost_preprocessor.joblib",
        device: str = "cuda",
        conf_threshold: float = 0.25,
    ):
        self.device = "cuda" if torch.cuda.is_available() and device != "cpu" else "cpu"
        print(f"[Pipeline] Initializing Complete Car Damage AI Pipeline on {self.device}...")

        # 1 & 2. Detector + Classifier
        self.two_stage_detector = TwoStageDamageDetector(
            detector_weights=Path(detector_weights),
            classifier_weights=Path(classifier_weights),
            device=self.device,
            conf_threshold=conf_threshold,
        )

        # 3. Component Identifier
        print("[Pipeline] Initializing Component Identification Engine...")
        self.component_identifier = VehicleComponentIdentifier()

        # 4. Severity Assessor
        print("[Pipeline] Initializing Severity Assessor Engine...")
        self.severity_assessor = VehicleSeverityAssessor(
            model_weights=Path(severity_weights),
            device=self.device,
        )

        # 5. Cost Estimator
        print("[Pipeline] Initializing Repair Cost & Uncertainty Estimator...")
        self.cost_estimator = RepairCostEstimator(
            model_path=cost_model_path,
            quantile_path=quantile_path,
            preprocessor_path=preprocessor_path,
        )

        print("[Pipeline] All 5 Pipeline Stages Successfully Loaded and Ready.")

    def run_inspection(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        vehicle_metadata: Optional[Dict[str, Any]] = None,
        viewpoint_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute complete 5-stage inspection workflow.

        Parameters:
        -----------
        image_input : file path, numpy BGR array, or PIL Image.
        vehicle_metadata : Dict with 'year', 'make', 'bodyStyle', 'drivetrain', 'runsDrives'.
        viewpoint_hint : optional str ('front', 'side', 'rear') for component spatial prior.

        Returns:
        --------
        Dict with comprehensive inspection summary, damage items, cost estimate, and annotated image.
        """
        # Default metadata if not provided
        if vehicle_metadata is None:
            vehicle_metadata = {
                "year": 2021,
                "make": "TOYOTA",
                "bodyStyle": "SEDAN",
                "drivetrain": "FWD",
                "runsDrives": "Run & Drive Verified",
            }

        # 1. Load image to BGR numpy
        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
            if img_bgr is None:
                raise FileNotFoundError(f"Failed to read image at: {image_input}")
            img_path_str = str(image_input)
        elif isinstance(image_input, Image.Image):
            img_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
            img_path_str = None
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input.copy()
            img_path_str = None
        else:
            raise TypeError(f"Unsupported image_input type: {type(image_input)}")

        orig_h, orig_w = img_bgr.shape[:2]
        total_pixels = float(orig_h * orig_w)

        # 2. Global Severity Assessment (ResNet18)
        if img_path_str and Path(img_path_str).exists():
            global_sev = self.severity_assessor.assess_global_image(img_path_str)
        else:
            # Fallback to in-memory PIL
            pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            temp_path = REPO_ROOT / "evaluation" / "temp_eval.jpg"
            pil_img.save(temp_path)
            global_sev = self.severity_assessor.assess_global_image(temp_path)
            if temp_path.exists():
                temp_path.unlink()

        # 3. Two-Stage Detection & Fine-Grained Classification
        _, raw_detections = self.two_stage_detector.predict(img_bgr)

        # 4. Component Identification & Instance DSI
        processed_damages: List[Dict[str, Any]] = []
        cost_bridge_detections: List[Dict[str, Any]] = []

        for det in raw_detections:
            # det keys: 'bbox_xyxy', 'detector_class', 'classifier_class', 'classifier_conf'
            x1, y1, x2, y2 = det["bbox_xyxy"]
            box_area = (x2 - x1) * (y2 - y1)
            norm_area = float(np.clip(box_area / total_pixels, 0.001, 1.0))

            # Calculate normalized xywh center coordinates
            bw_norm = (x2 - x1) / float(orig_w)
            bh_norm = (y2 - y1) / float(orig_h)
            xc_norm = (x1 + x2) / (2.0 * orig_w)
            yc_norm = (y1 + y2) / (2.0 * orig_h)

            damage_type = det["classifier_class"]
            conf = float(det["classifier_conf"])

            # Identify vehicle component
            comp_info = self.component_identifier.identify_component(
                damage_class=damage_type,
                norm_bbox=(xc_norm, yc_norm, bw_norm, bh_norm),
                detector_conf=conf,
                image=img_bgr,
            )

            comp_name = comp_info["primary_component"]

            # Instance Severity (DSI)
            from src.severity.dsi import compute_instance_dsi
            dsi_res = compute_instance_dsi(
                damage_type=damage_type,
                component=comp_name,
                norm_area=norm_area,
                confidence=conf,
            )

            damage_item = {
                "damage_type": damage_type,
                "confidence": round(conf, 4),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "normalized_area_pct": round(norm_area * 100, 2),
                "component": comp_name,
                "component_confidence": round(comp_info["confidence"], 4),
                "copart_sector": comp_info["copart_sector"],
                "labor_parts_multiplier": comp_info["cost_multiplier"],
                "dsi_score": dsi_res["dsi_score"],
                "severity_grade": dsi_res["severity_grade"],
            }
            processed_damages.append(damage_item)

            cost_bridge_detections.append({
                "damage_type": damage_type,
                "component": comp_name,
                "norm_area": norm_area,
                "confidence": conf,
            })

        # 5. Actuarial Repair Cost & Uncertainty Estimation
        cost_result = self.cost_estimator.predict(
            image_detections=cost_bridge_detections,
            vehicle_metadata=vehicle_metadata,
        )

        # 6. Generate Annotated Overlay Image
        annotated_img = self._render_annotated_image(img_bgr, processed_damages, cost_result)

        # Compile final inspection report
        report = {
            "vehicle_profile": {
                "make": vehicle_metadata.get("make", "TOYOTA"),
                "year": vehicle_metadata.get("year", 2021),
                "vehicle_age": cost_result["vehicle_summary"]["age_years"],
                "brand_tier": cost_result["vehicle_summary"]["brand_tier"],
                "body_style": cost_result["vehicle_summary"]["body_style"],
                "operable": cost_result["vehicle_summary"]["operable"],
            },
            "inspection_summary": {
                "total_damage_instances": len(processed_damages),
                "total_damaged_area_pct": cost_result["visual_summary"]["damaged_area_ratio_pct"],
                "global_severity_classifier": global_sev["severity_grade"],
                "max_dsi_score": cost_result["visual_summary"]["dsi_score"],
                "composite_severity": cost_result["visual_summary"]["severity_grade"],
                "primary_damaged_component": cost_result["visual_summary"]["primary_component"],
                "primary_impact_sector": cost_result["visual_summary"]["primary_damage_sector"],
            },
            "repair_cost_estimate": {
                "estimated_cost_usd": cost_result["estimated_cost_usd"],
                "confidence_interval_90_usd": cost_result["confidence_interval_90_usd"],
                "primary_cost_driver": cost_result["primary_cost_driver"],
            },
            "detected_damages": processed_damages,
            "pipeline_stages_verified": [
                "Stage 1: YOLOv8n Damage Localization",
                "Stage 2: EfficientNet-B0 Crop Classification",
                "Stage 3: 10-Component Ontology & Spatial Geometry",
                "Stage 4: Damage Severity Index (DSI) Formulation",
                "Stage 5: HistGradientBoosting & Quantile Cost Regressors",
            ],
        }

        return {
            "report": report,
            "annotated_image": annotated_img,
        }

    def _render_annotated_image(
        self,
        img: np.ndarray,
        damages: List[Dict[str, Any]],
        cost_res: Dict[str, Any],
    ) -> np.ndarray:
        """Render high-resolution visual inspection overlay with banner."""
        canvas = img.copy()
        h, w = canvas.shape[:2]

        # Draw damage boxes
        colors = {
            "scratch": (0, 215, 255),       # Gold
            "dent": (255, 120, 0),          # Blue
            "crack": (0, 0, 255),           # Red
            "glass shatter": (255, 0, 255), # Magenta
            "lamp broken": (0, 165, 255),   # Orange
            "tire flat": (128, 0, 128),     # Purple
        }

        for d in damages:
            x1, y1, x2, y2 = d["bbox"]
            color = colors.get(d["damage_type"], (0, 255, 0))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)

            # Label text
            label = f"{d['component'].upper()} | {d['damage_type']} (DSI: {d['dsi_score']})"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(canvas, (x1, max(0, y1 - 25)), (x1 + lw + 8, y1), color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 4, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
            )

        # Header Watermark Banner
        banner_h = 75
        banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
        # Deep blue slate banner
        banner[:] = (40, 30, 20)

        cost_val = cost_res["estimated_cost_usd"]
        ci_low = cost_res["confidence_interval_90_usd"]["lower_bound"]
        ci_high = cost_res["confidence_interval_90_usd"]["upper_bound"]

        title_text = "CAR DAMAGE AI INSPECTION & COST ESTIMATE"
        cv2.putText(banner, title_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        stats_text = (
            f"Damages: {len(damages)} | DSI: {cost_res['visual_summary']['dsi_score']} ({cost_res['visual_summary']['severity_grade'].upper()}) | "
            f"EST. REPAIR COST: ${cost_val:,.2f} [90% PI: ${ci_low:,.0f} - ${ci_high:,.0f}]"
        )
        cv2.putText(banner, stats_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 230, 255), 2)

        # Stack banner on top of image
        annotated_full = np.vstack([banner, canvas])
        return annotated_full
