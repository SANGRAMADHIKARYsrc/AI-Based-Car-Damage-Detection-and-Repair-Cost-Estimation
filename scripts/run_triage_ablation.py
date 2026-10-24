"""Academic Ablation Experiment: Adjuster Triage vs Pure Vehicle Metadata.

Quantifies the exact predictive contribution of human insurance adjuster triage:
- damageDescription (primary damage sector)
- sector_cost_weight & base_sector_severity
- runsDrives (is_operable, starts_only)
- secondaryDamage (has_secondary_damage)
vs pure vehicle metadata (vehicle age, brand tier, body style, drivetrain).

Evaluated on the frozen test split (N = 2,878 lots).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
EVAL_DIR = REPO_ROOT / "evaluation" / "cost"


def compute_metrics(y_true_raw, y_pred_raw, y_true_log, y_pred_log):
    return {
        "mae_dollar": round(float(mean_absolute_error(y_true_raw, y_pred_raw)), 2),
        "median_ae_dollar": round(float(np.median(np.abs(y_true_raw - y_pred_raw))), 2),
        "rmse_dollar": round(float(np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))), 2),
        "mape_pct": round(float(mean_absolute_percentage_error(y_true_raw, y_pred_raw) * 100.0), 2),
        "r2_dollar": round(float(r2_score(y_true_raw, y_pred_raw)), 4),
        "mae_log": round(float(mean_absolute_error(y_true_log, y_pred_log)), 4),
        "rmse_log": round(float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))), 4),
        "r2_log": round(float(r2_score(y_true_log, y_pred_log)), 4),
    }


def main():
    print("Loading preprocessed dataset splits...")
    X_train = np.load(DATA_DIR / "cost_X_train.npy")
    y_train_log = np.load(DATA_DIR / "cost_y_train_log.npy")
    y_train_raw = np.load(DATA_DIR / "cost_y_train_raw.npy")

    X_test = np.load(DATA_DIR / "cost_X_test.npy")
    y_test_log = np.load(DATA_DIR / "cost_y_test_log.npy")
    y_test_raw = np.load(DATA_DIR / "cost_y_test_raw.npy")

    with open(DATA_DIR / "cost_features_metadata.json", "r") as f:
        meta = json.load(f)
    feature_names = meta["transformed_feature_names"]

    print(f"Total features: {len(feature_names)}")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Classify feature indices
    # Pure vehicle metadata: age, brand_tier, body_style, drivetrain
    meta_keywords = ["vehicle_age", "brand_tier", "body_style_clean", "drivetrain_clean"]
    meta_indices = [
        i for i, fn in enumerate(feature_names)
        if any(k in fn for k in meta_keywords)
    ]

    # Adjuster triage: damageDescription, sector_cost_weight, base_sector_severity, runsDrives, secondaryDamage
    triage_keywords = [
        "primary_damage_clean", "sector_cost_weight", "base_sector_severity",
        "is_operable", "starts_only", "has_secondary_damage"
    ]
    triage_indices = [
        i for i, fn in enumerate(feature_names)
        if any(k in fn for k in triage_keywords)
    ]

    # Visual proxy features
    visual_keywords = ["visual_"]
    visual_indices = [
        i for i, fn in enumerate(feature_names)
        if any(k in fn for k in visual_keywords)
    ]

    # Combined sets
    metadata_plus_triage = sorted(list(set(meta_indices + triage_indices)))
    all_indices = list(range(len(feature_names)))

    print(f"Vehicle Metadata features:   {len(meta_indices)}")
    print(f"Adjuster Triage features:    {len(triage_indices)}")
    print(f"Visual Proxy features:       {len(visual_indices)}")
    print(f"Metadata + Triage features:  {len(metadata_plus_triage)}")
    print(f"Full 57-feature Model:       {len(all_indices)}")

    configurations = {
        "1. Dummy Baseline (Median)": None,
        "2. Pure Vehicle Metadata Only (No Triage, No Visual)": meta_indices,
        "3. Adjuster Triage Only (No Vehicle Metadata)": triage_indices,
        "4. Vehicle Metadata + Adjuster Triage (No Visual)": metadata_plus_triage,
        "5. Full Model (Metadata + Triage + Visual Proxy)": all_indices,
    }

    results = {}

    for name, indices in configurations.items():
        print(f"\nEvaluating: {name}...")
        if indices is None:
            # Dummy Regressor
            dummy = DummyRegressor(strategy="median")
            dummy.fit(X_train, y_train_log)
            pred_log = dummy.predict(X_test)
            pred_raw = np.expm1(pred_log)
            m = compute_metrics(y_test_raw, pred_raw, y_test_log, pred_log)
            m["feature_count"] = 0
            results[name] = m
        else:
            X_tr_sub = X_train[:, indices]
            X_te_sub = X_test[:, indices]
            model = HistGradientBoostingRegressor(
                max_iter=150, max_depth=8, learning_rate=0.08, random_state=42
            )
            model.fit(X_tr_sub, y_train_log)
            pred_log = model.predict(X_te_sub)
            pred_raw = np.expm1(pred_log)
            m = compute_metrics(y_test_raw, pred_raw, y_test_log, pred_log)
            m["feature_count"] = len(indices)
            results[name] = m

    # Calculate differences / drops
    full_m = results["5. Full Model (Metadata + Triage + Visual Proxy)"]
    meta_triage_m = results["4. Vehicle Metadata + Adjuster Triage (No Visual)"]
    meta_only_m = results["2. Pure Vehicle Metadata Only (No Triage, No Visual)"]

    triage_drop_r2 = round(meta_triage_m["r2_dollar"] - meta_only_m["r2_dollar"], 4)
    triage_mae_increase = round(meta_only_m["mae_dollar"] - meta_triage_m["mae_dollar"], 2)

    summary_payload = {
        "test_sample_count": len(X_test),
        "configurations": results,
        "adjuster_triage_contribution": {
            "r2_delta_when_adding_triage_to_metadata": triage_drop_r2,
            "mae_reduction_dollar_when_adding_triage_to_metadata": triage_mae_increase,
            "interpretation": (
                f"Dropping adjuster triage features (damageDescription, sector_cost_weight, runsDrives) "
                f"causes test R² to drop from {meta_triage_m['r2_dollar']:.4f} to {meta_only_m['r2_dollar']:.4f} "
                f"and test MAE to worsen by ${triage_mae_increase:,.2f}."
            ),
        },
    }

    out_json = EVAL_DIR / "triage_ablation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n" + "=" * 80)
    print("ADJUSTER TRIAGE ABLATION STUDY RESULTS (Test Set N = 2,878)")
    print("=" * 80)
    header = f"{'Configuration':<45} | {'Feats':<5} | {'MAE ($)':<10} | {'RMSE ($)':<10} | {'MAPE (%)':<8} | {'R²':<7} | {'Log R²':<7}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<45} | {m['feature_count']:<5} | ${m['mae_dollar']:<9,.2f} | ${m['rmse_dollar']:<9,.2f} | {m['mape_pct']:<7.2f}% | {m['r2_dollar']:<7.4f} | {m['r2_log']:<7.4f}")
    print("=" * 80)
    print(f"\n[OK] Results saved to: {out_json}")


if __name__ == "__main__":
    main()
