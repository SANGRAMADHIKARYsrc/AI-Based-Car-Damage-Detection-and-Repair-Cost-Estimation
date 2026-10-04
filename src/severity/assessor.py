"""Phase 7 — Vehicle Severity Assessor Engine.

Fuses:
1. Deep visual global severity classification (minor, moderate, severe)
2. Localized instance-level Damage Severity Index (DSI)
Provides holistic damage severity assessment for repair cost estimation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from PIL import Image
import torch

from .dataset import SEVERITY_CLASSES, get_severity_transforms
from .dsi import compute_instance_dsi
from .model import DamageSeverityClassifier


class VehicleSeverityAssessor:
    """Holistic vehicle damage severity assessor."""

    def __init__(
        self,
        model_weights: Optional[Union[str, Path]] = None,
        device: str = "cuda",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.model = None
        self.transform = get_severity_transforms(img_size=224, is_train=False)

        if model_weights and Path(model_weights).exists():
            ckpt = torch.load(model_weights, map_location=self.device, weights_only=False)
            backbone = ckpt.get("config", {}).get("model", {}).get("backbone", "resnet18")
            self.model = DamageSeverityClassifier(backbone_name=backbone, num_classes=3, pretrained=False).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()

    def assess_global_image(self, image_path: Union[str, Path]) -> Dict[str, Union[str, float, List[float]]]:
        """Classify global damage severity of a full vehicle image."""
        if self.model is None:
            return {"grade": "moderate", "confidence": 0.5, "probabilities": [0.33, 0.33, 0.34]}

        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.model.predict_proba(tensor)[0].float().cpu().numpy()

        best_idx = int(np.argmax(probs))
        grade = SEVERITY_CLASSES[best_idx]
        confidence = float(probs[best_idx])

        return {
            "severity_grade": grade,
            "confidence": round(confidence, 4),
            "probabilities": {cls_name: round(float(probs[i]), 4) for i, cls_name in enumerate(SEVERITY_CLASSES)},
        }

    def assess_vehicle_damage(
        self,
        detections: List[dict],
        image_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Union[str, float, List[dict]]]:
        """Assess overall vehicle severity from both detections and image."""
        instance_assessments = []
        dsi_scores = []

        for d in detections:
            dmg_type = d.get("damage_type") or d.get("damage_class", "scratch")
            comp = d.get("component", "door")
            norm_area = d.get("norm_area", 0.05)
            conf = d.get("damage_conf") or d.get("confidence", 1.0)

            inst_res = compute_instance_dsi(
                damage_type=dmg_type,
                component=comp,
                norm_area=norm_area,
                confidence=conf,
            )
            instance_assessments.append({**d, **inst_res})
            dsi_scores.append(inst_res["dsi_score"])

        max_dsi = max(dsi_scores) if dsi_scores else 0.0
        avg_dsi = float(np.mean(dsi_scores)) if dsi_scores else 0.0

        # Global visual assessment if image provided
        global_res = None
        if image_path and Path(image_path).exists() and self.model is not None:
            global_res = self.assess_global_image(image_path)

        # Determine composite severity grade
        if global_res:
            g_grade = global_res["severity_grade"]
            if max_dsi >= 75.0 or g_grade == "severe":
                overall_grade = "severe"
            elif max_dsi >= 40.0 or g_grade == "moderate":
                overall_grade = "moderate"
            else:
                overall_grade = "minor"
        else:
            if max_dsi >= 72.0:
                overall_grade = "severe"
            elif max_dsi >= 38.0:
                overall_grade = "moderate"
            else:
                overall_grade = "minor"

        return {
            "overall_severity_grade": overall_grade,
            "max_instance_dsi": round(max_dsi, 1),
            "mean_instance_dsi": round(avg_dsi, 1),
            "total_damages": len(detections),
            "global_assessment": global_res,
            "instances": instance_assessments,
        }
