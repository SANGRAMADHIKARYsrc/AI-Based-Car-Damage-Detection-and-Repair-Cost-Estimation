"""Phase 8 — Unit and Integration Tests for Repair Cost Dataset Pipeline.

Verifies:
1. Feature engineering consistency and absence of NaN values
2. ColumnTransformer preprocessing repeatability and serializability
3. Absence of target leakage between train, val, and test splits
4. Visual pipeline feature extractor compatibility
"""

import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
import pytest

from src.cost.dataset_builder import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMNS,
    build_preprocessor,
)
from src.cost.feature_engineering import (
    engineer_features,
    extract_cost_features_from_visual_pipeline,
)


def test_feature_engineering_structure():
    dummy_data = pd.DataFrame(
        [
            {
                "lotId": 1001,
                "year": 2021,
                "make": "BMW",
                "modelGroup": "3 SERIES",
                "bodyStyle": "SEDAN 4DR",
                "damageDescription": "FRONT END",
                "secondaryDamage": "MINOR DENT/SCRATCHES",
                "drivetrain": "ALL WHEEL DRIVE",
                "transmission": "AUTOMATIC",
                "fuelType": "GAS",
                "runsDrives": "Run & Drive Verified",
                "repairCost": 12500.0,
                "log_repair_cost": np.log1p(12500.0),
                "split": "train",
            }
        ]
    )

    feat_df = engineer_features(dummy_data, include_synthetic_visual=True, seed=42)

    for col in NUMERICAL_FEATURES:
        assert col in feat_df.columns, f"Missing numerical feature: {col}"
        assert not feat_df[col].isna().any(), f"NaN found in numerical feature: {col}"

    for col in CATEGORICAL_FEATURES:
        assert col in feat_df.columns, f"Missing categorical feature: {col}"
        assert not feat_df[col].isna().any(), f"NaN found in categorical feature: {col}"

    assert feat_df["brand_tier"].iloc[0] == "luxury"
    assert feat_df["body_style_clean"].iloc[0] == "sedan"
    assert feat_df["drivetrain_clean"].iloc[0] == "AWD"
    assert feat_df["is_operable"].iloc[0] == 1


def test_preprocessor_pipeline_serialization(tmp_path):
    train_df = pd.read_csv("data/processed/cost_features_train.csv")
    test_df = pd.read_csv("data/processed/cost_features_test.csv")

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)

    assert X_train.shape[0] == len(train_df)
    assert X_test.shape[0] == len(test_df)
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()

    # Save and reload test
    model_path = tmp_path / "temp_preprocessor.joblib"
    joblib.dump(preprocessor, model_path)
    loaded_preprocessor = joblib.load(model_path)

    X_test_reloaded = loaded_preprocessor.transform(test_df)
    np.testing.assert_allclose(X_test, X_test_reloaded, rtol=1e-5)


def test_no_data_leakage():
    train_df = pd.read_csv("data/processed/cost_features_train.csv")
    val_df = pd.read_csv("data/processed/cost_features_val.csv")
    test_df = pd.read_csv("data/processed/cost_features_test.csv")

    train_lots = set(train_df["lotId"])
    val_lots = set(val_df["lotId"])
    test_lots = set(test_df["lotId"])

    assert len(train_lots.intersection(val_lots)) == 0, "Train-Val lot overlap detected!"
    assert len(train_lots.intersection(test_lots)) == 0, "Train-Test lot overlap detected!"
    assert len(val_lots.intersection(test_lots)) == 0, "Val-Test lot overlap detected!"


def test_visual_pipeline_feature_extractor():
    sample_detections = [
        {
            "damage_type": "dent",
            "component": "front_bumper",
            "norm_area": 0.08,
            "confidence": 0.92,
        },
        {
            "damage_type": "scratch",
            "component": "hood",
            "norm_area": 0.04,
            "confidence": 0.88,
        },
    ]

    vehicle_metadata = {
        "year": 2022,
        "make": "TESLA",
        "bodyStyle": "SEDAN",
        "drivetrain": "AWD",
        "runsDrives": "Run & Drive Verified",
    }

    features = extract_cost_features_from_visual_pipeline(sample_detections, vehicle_metadata)

    assert features["brand_tier"] == "luxury"
    assert features["vehicle_age"] == 4
    assert features["visual_damage_count"] == 2
    assert features["visual_dsi"] > 0
    assert features["primary_damage_clean"] in ["FRONT END", "SIDE", "REAR END", "MINOR DENT/SCRATCHES"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
