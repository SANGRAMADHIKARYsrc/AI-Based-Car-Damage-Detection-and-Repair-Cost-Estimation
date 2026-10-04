"""Phase 6 — Vehicle Component Identification Engine.

Fuses:
1. Deterministic damage-type semantics (tire flat -> wheel_tire, lamp broken -> lamp)
2. Spatial geometric topology priors across canonical car viewpoints
3. Substrate material & damage affinity weighting
Outputs standardized vehicle components and Copart loss sector tags.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .ontology import (
    COMPONENT_COST_WEIGHTS,
    COMPONENT_TO_COPART_SECTOR,
    DETERMINISTIC_DAMAGE_MAPPING,
    VEHICLE_COMPONENTS,
)
from .spatial_geometry import SpatialComponentPrior


class VehicleComponentIdentifier:
    """End-to-end Vehicle Component Identification Engine."""

    def __init__(self):
        self.spatial_prior = SpatialComponentPrior()

        # Damage type affinity multipliers on structural components
        self.damage_affinity: Dict[str, Dict[str, float]] = {
            "dent": {
                "front_bumper": 1.25,
                "rear_bumper": 1.20,
                "hood": 1.30,
                "door": 1.40,
                "fender": 1.25,
                "quarter_panel": 1.20,
                "roof": 0.80,
                "windshield_glass": 0.01,
                "lamp": 0.05,
                "wheel_tire": 0.05,
            },
            "scratch": {
                "door": 1.35,
                "front_bumper": 1.30,
                "rear_bumper": 1.25,
                "fender": 1.20,
                "quarter_panel": 1.20,
                "hood": 1.15,
                "roof": 0.70,
                "windshield_glass": 0.05,
                "lamp": 0.10,
                "wheel_tire": 0.05,
            },
            "crack": {
                "front_bumper": 1.55,  # Plastic bumper cracks are common
                "rear_bumper": 1.45,
                "fender": 0.90,
                "hood": 0.70,
                "door": 0.60,
                "quarter_panel": 0.60,
                "roof": 0.40,
                "windshield_glass": 0.30,
                "lamp": 0.20,
                "wheel_tire": 0.05,
            },
        }

    def identify_component(
        self,
        damage_class: str,
        norm_bbox: Union[List[float], Tuple[float, float, float, float]],
        detector_conf: float = 1.0,
    ) -> Dict[str, Union[str, float, List[Dict[str, float]]]]:
        """Identify vehicle component from damage class and normalized bounding box.

        Args:
            damage_class: Classified damage type ('dent', 'scratch', 'crack',
                          'glass shatter', 'lamp broken', 'tire flat').
            norm_bbox: (x_center, y_center, width, height) normalized in [0, 1].
            detector_conf: Confidence of damage detection.

        Returns:
            Dictionary containing:
            - primary_component: Identified component name.
            - confidence: Confidence of component identification [0, 1].
            - copart_sector: Mapped Copart loss sector ('FRONT END', 'SIDE', etc.).
            - cost_multiplier: Relative repair/parts cost weight.
            - candidates: Ranked list of alternative components with probabilities.
        """
        xc, yc, bw, bh = norm_bbox

        # 1. Deterministic semantic shortcut
        if damage_class in DETERMINISTIC_DAMAGE_MAPPING:
            comp = DETERMINISTIC_DAMAGE_MAPPING[damage_class]
            confidence = min(0.99, round(0.95 * detector_conf + 0.04, 3))
            return {
                "primary_component": comp,
                "confidence": confidence,
                "copart_sector": COMPONENT_TO_COPART_SECTOR.get(comp, "SIDE"),
                "cost_multiplier": COMPONENT_COST_WEIGHTS.get(comp, 1.0),
                "method": "deterministic_semantic",
                "candidates": [{"component": comp, "probability": 1.0}],
            }

        # 2. Probabilistic spatial prior
        spatial_probs = self.spatial_prior.compute_prior_distribution(xc, yc, bw, bh)

        # 3. Fuse with damage affinity
        affinity = self.damage_affinity.get(damage_class, {c: 1.0 for c in VEHICLE_COMPONENTS})
        fused_scores = {}
        for comp in VEHICLE_COMPONENTS:
            prior = spatial_probs.get(comp, 1e-4)
            aff = affinity.get(comp, 1.0)
            fused_scores[comp] = prior * aff

        # Normalize probabilities
        total = sum(fused_scores.values()) + 1e-12
        norm_probs = {k: v / total for k, v in fused_scores.items()}

        sorted_candidates = sorted(norm_probs.items(), key=lambda x: x[1], reverse=True)
        primary_comp, top_prob = sorted_candidates[0]

        # Calculate final calibrated confidence
        confidence = float(np.clip(top_prob * 1.15, 0.45, 0.95))

        # Check front vs rear for bumpers and lamps based on vertical/horizontal position
        if primary_comp in ["front_bumper", "rear_bumper"]:
            # If box is in lower middle, default to front_bumper unless specified
            pass

        return {
            "primary_component": primary_comp,
            "confidence": round(confidence, 3),
            "copart_sector": COMPONENT_TO_COPART_SECTOR.get(primary_comp, "FRONT END"),
            "cost_multiplier": COMPONENT_COST_WEIGHTS.get(primary_comp, 1.0),
            "method": "spatial_affinity_fusion",
            "candidates": [
                {"component": c, "probability": round(float(p), 4)}
                for c, p in sorted_candidates[:4]
            ],
        }
