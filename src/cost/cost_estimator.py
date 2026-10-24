"""Phase 9 — Production Cost Estimator Inference Engine.

Provides an end-to-end interface:
Takes raw visual damage detections (bounding boxes, classes, components, severity)
and vehicle metadata, extracts features, queries the champion regression model,
computes 90% prediction intervals, and returns formatted inspection results.
"""

import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Compatibility alias for scikit-learn Cython unpickling
try:
    import sklearn._loss._loss
    sys.modules["_loss"] = sklearn._loss._loss
except ImportError:
    pass

from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

from src.cost.feature_engineering import extract_cost_features_from_visual_pipeline


class RepairCostEstimator:
    """End-to-end inference engine for repair cost estimation."""

    def __init__(
        self,
        model_path: str = "models/cost/best_cost_model.joblib",
        quantile_path: str = "models/cost/quantile_models.joblib",
        preprocessor_path: str = "data/processed/cost_preprocessor.joblib",
    ):
        self.model_path = Path(model_path)
        self.quantile_path = Path(quantile_path)
        self.preprocessor_path = Path(preprocessor_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Champion cost model not found at: {self.model_path}")

        self.model = joblib.load(self.model_path)
        self.quantiles = (
            joblib.load(self.quantile_path) if self.quantile_path.exists() else None
        )
        self.preprocessor = joblib.load(self.preprocessor_path)

    def predict(
        self,
        image_detections: List[Dict[str, Any]],
        vehicle_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Estimate repair cost from visual detections and vehicle attributes.

        Parameters:
        -----------
        image_detections : List of dicts
            Each with keys: 'damage_type', 'component', 'norm_area', 'confidence'
        vehicle_metadata : Dict
            Keys: 'year', 'make', 'bodyStyle', 'drivetrain', 'runsDrives'

        Returns:
        --------
        Dict with predicted_cost, interval_90_lower, interval_90_upper, DSI, etc.
        """
        # 1. Feature extraction
        feature_dict = extract_cost_features_from_visual_pipeline(
            image_detections, vehicle_metadata
        )
        feat_df = pd.DataFrame([feature_dict])

        # 2. Preprocessing
        X = self.preprocessor.transform(feat_df)

        # 3. Model prediction (log target -> dollar)
        pred_log = float(self.model.predict(X)[0])
        pred_dollar = float(np.expm1(pred_log))

        # 4. Uncertainty Intervals
        if self.quantiles:
            q05_log = float(self.quantiles["q_5"].predict(X)[0])
            q95_log = float(self.quantiles["q_95"].predict(X)[0])
            lower = float(min(np.expm1(q05_log), pred_dollar))
            upper = float(max(np.expm1(q95_log), pred_dollar))
        else:
            # Fallback 20% margin
            lower = pred_dollar * 0.80
            upper = pred_dollar * 1.25

        # 5. Determine primary cost driver
        primary_driver = (
            f"{feature_dict['brand_tier'].capitalize()} brand with "
            f"{feature_dict['primary_damage_clean'].lower()} damage on "
            f"{feature_dict['visual_primary_component']}"
        )

        return {
            "estimated_cost_usd": round(pred_dollar, 2),
            "confidence_interval_90_usd": {
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
                "interval_width": round(upper - lower, 2),
            },
            "visual_summary": {
                "damage_count": feature_dict["visual_damage_count"],
                "damaged_area_ratio_pct": round(feature_dict["visual_damaged_area_ratio"] * 100, 2),
                "dsi_score": feature_dict["visual_dsi"],
                "severity_grade": ["minor", "moderate", "severe"][feature_dict["visual_severity_grade"]],
                "primary_component": feature_dict["visual_primary_component"],
                "primary_damage_sector": feature_dict["primary_damage_clean"],
            },
            "vehicle_summary": {
                "year": vehicle_metadata.get("year", 2020),
                "age_years": feature_dict["vehicle_age"],
                "make": vehicle_metadata.get("make", "TOYOTA"),
                "brand_tier": feature_dict["brand_tier"],
                "body_style": feature_dict["body_style_clean"],
                "operable": bool(feature_dict["is_operable"]),
            },
            "primary_cost_driver": primary_driver,
        }
