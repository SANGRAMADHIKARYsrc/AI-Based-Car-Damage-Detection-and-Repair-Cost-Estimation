"""Phase 8 — Dataset Builder & Preprocessor Pipeline.

Processes raw Copart data, applies feature engineering, fits scikit-learn
preprocessing pipeline strictly on training split (no leakage), and exports
train/val/test splits along with paired multimodal validation benchmarks.
"""

import json
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.cost.feature_engineering import engineer_features

# Feature definitions
NUMERICAL_FEATURES: List[str] = [
    "vehicle_age",
    "vehicle_age_sq",
    "log_vehicle_age",
    "is_operable",
    "starts_only",
    "has_secondary_damage",
    "sector_cost_weight",
    "base_sector_severity",
    "visual_damage_count",
    "visual_damaged_area_ratio",
    "visual_dsi",
    "visual_severity_grade",
]

CATEGORICAL_FEATURES: List[str] = [
    "brand_tier",
    "body_style_clean",
    "drivetrain_clean",
    "primary_damage_clean",
    "visual_primary_component",
]

TARGET_COLUMNS: List[str] = [
    "repairCost",
    "log_repair_cost",
]

META_COLUMNS: List[str] = [
    "lotId",
    "split",
]


def build_preprocessor() -> ColumnTransformer:
    """Instantiate scikit-learn ColumnTransformer for cost estimation features."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def build_and_export_dataset(
    raw_csv_path: str = "data/processed/copart_repair_cost_clean.csv",
    output_dir: str = "data/processed",
) -> Dict[str, any]:
    """Execute complete dataset build, feature engineering, and pipeline export."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw clean dataset from: {raw_csv_path}")
    raw_df = pd.read_csv(raw_csv_path)

    print(f"Engineering features across {len(raw_df)} records...")
    df_feat = engineer_features(raw_df, include_synthetic_visual=True, seed=42)

    # Split data according to pre-established splits
    train_mask = df_feat["split"] == "train"
    val_mask = df_feat["split"] == "val"
    test_mask = df_feat["split"] == "test"

    train_df = df_feat[train_mask].copy()
    val_df = df_feat[val_mask].copy()
    test_df = df_feat[test_mask].copy()

    print(f"Split breakdown: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Save readable CSV feature tables
    train_csv = out_path / "cost_features_train.csv"
    val_csv = out_path / "cost_features_val.csv"
    test_csv = out_path / "cost_features_test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    print(f"Exported readable feature CSVs to {out_path}")

    # Build and fit preprocessor strictly on train
    preprocessor = build_preprocessor()
    print("Fitting preprocessor on training split...")
    X_train_transformed = preprocessor.fit_transform(train_df)
    X_val_transformed = preprocessor.transform(val_df)
    X_test_transformed = preprocessor.transform(test_df)

    # Extract feature names after OneHotEncoding
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feature_names = NUMERICAL_FEATURES + cat_feature_names

    print(f"Total transformed feature dimension: {len(all_feature_names)}")

    # Targets
    y_train_log = train_df["log_repair_cost"].to_numpy()
    y_val_log = val_df["log_repair_cost"].to_numpy()
    y_test_log = test_df["log_repair_cost"].to_numpy()

    y_train_raw = train_df["repairCost"].to_numpy()
    y_val_raw = val_df["repairCost"].to_numpy()
    y_test_raw = test_df["repairCost"].to_numpy()

    # Save arrays for high-speed modeling in Phase 9
    np.save(out_path / "cost_X_train.npy", X_train_transformed)
    np.save(out_path / "cost_X_val.npy", X_val_transformed)
    np.save(out_path / "cost_X_test.npy", X_test_transformed)

    np.save(out_path / "cost_y_train_log.npy", y_train_log)
    np.save(out_path / "cost_y_val_log.npy", y_val_log)
    np.save(out_path / "cost_y_test_log.npy", y_test_log)

    np.save(out_path / "cost_y_train_raw.npy", y_train_raw)
    np.save(out_path / "cost_y_val_raw.npy", y_val_raw)
    np.save(out_path / "cost_y_test_raw.npy", y_test_raw)

    # Save preprocessor and metadata
    preprocessor_path = out_path / "cost_preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    metadata = {
        "num_records_total": len(df_feat),
        "splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "num_features_raw_engineered": len(df_feat.columns),
        "num_features_transformed": len(all_feature_names),
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "transformed_feature_names": all_feature_names,
        "target_variables": TARGET_COLUMNS,
    }

    meta_path = out_path / "cost_features_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Pipeline artifacts successfully written to: {out_path}")
    return metadata


if __name__ == "__main__":
    build_and_export_dataset()
