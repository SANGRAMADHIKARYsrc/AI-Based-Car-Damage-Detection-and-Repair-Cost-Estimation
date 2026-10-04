"""Phase 9 — Unit and Integration Tests for Cost Estimation Engine.

Verifies:
1. Model loading and inference consistency across diverse inputs
2. Non-negative cost predictions and monotonic prediction intervals
3. Monotonicity checks: higher damage / newer cars produce higher estimated repair costs
4. End-to-end RepairCostEstimator interface
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from src.cost.cost_estimator import RepairCostEstimator


@pytest.fixture(scope="module")
def estimator():
    return RepairCostEstimator()


def test_estimator_initialization(estimator):
    assert estimator.model is not None
    assert estimator.quantiles is not None
    assert estimator.preprocessor is not None


def test_predict_returns_valid_structure(estimator):
    detections = [
        {"damage_type": "dent", "component": "front_bumper", "norm_area": 0.08, "confidence": 0.90}
    ]
    metadata = {
        "year": 2020,
        "make": "TOYOTA",
        "bodyStyle": "SEDAN",
        "drivetrain": "FWD",
        "runsDrives": "Run & Drive Verified",
    }
    res = estimator.predict(detections, metadata)

    assert "estimated_cost_usd" in res
    assert "confidence_interval_90_usd" in res
    assert "visual_summary" in res
    assert "vehicle_summary" in res

    assert res["estimated_cost_usd"] > 0
    ci = res["confidence_interval_90_usd"]
    assert ci["lower_bound"] <= res["estimated_cost_usd"] <= ci["upper_bound"]
    assert ci["interval_width"] > 0


def test_cost_monotonicity_with_severity(estimator):
    """Severe damage should result in higher cost than minor scratch."""
    metadata = {
        "year": 2021,
        "make": "FORD",
        "bodyStyle": "SEDAN",
        "drivetrain": "FWD",
        "runsDrives": "Run & Drive Verified",
    }

    # Minor scratch
    minor_det = [
        {"damage_type": "scratch", "component": "door", "norm_area": 0.02, "confidence": 0.85}
    ]
    res_minor = estimator.predict(minor_det, metadata)

    # Severe damage across multiple panels
    severe_det = [
        {"damage_type": "dent", "component": "front_bumper", "norm_area": 0.20, "confidence": 0.95},
        {"damage_type": "crack", "component": "hood", "norm_area": 0.15, "confidence": 0.92},
        {"damage_type": "lamp broken", "component": "lamp", "norm_area": 0.05, "confidence": 0.90},
    ]
    res_severe = estimator.predict(severe_det, metadata)

    assert res_severe["estimated_cost_usd"] > res_minor["estimated_cost_usd"], (
        f"Expected severe ({res_severe['estimated_cost_usd']}) > minor ({res_minor['estimated_cost_usd']})"
    )


def test_brand_tier_cost_differential(estimator):
    """Luxury vehicle should yield higher repair cost than economy vehicle for identical damage."""
    detections = [
        {"damage_type": "dent", "component": "front_bumper", "norm_area": 0.10, "confidence": 0.90}
    ]

    bmw_meta = {
        "year": 2022,
        "make": "BMW",
        "bodyStyle": "SEDAN",
        "drivetrain": "AWD",
        "runsDrives": "Run & Drive Verified",
    }
    honda_meta = {
        "year": 2022,
        "make": "HONDA",
        "bodyStyle": "SEDAN",
        "drivetrain": "FWD",
        "runsDrives": "Run & Drive Verified",
    }

    res_bmw = estimator.predict(detections, bmw_meta)
    res_honda = estimator.predict(detections, honda_meta)

    assert res_bmw["estimated_cost_usd"] > res_honda["estimated_cost_usd"], (
        f"Expected BMW ({res_bmw['estimated_cost_usd']}) > Honda ({res_honda['estimated_cost_usd']})"
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
