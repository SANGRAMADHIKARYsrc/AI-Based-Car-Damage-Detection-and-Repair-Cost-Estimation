"""Phase 9 — Cost Model Test Evaluation & Interval Calibration.

Evaluates the champion cost regressor and quantile uncertainty models on the
unseen test set (N = 2,878), analyzes error across cost tiers, and produces
publication-grade figures.
"""

import json
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, List
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.cost.train_cost_models import compute_metrics


def evaluate_test_set(
    data_dir: str = "data/processed",
    models_dir: str = "models/cost",
    eval_dir: str = "evaluation/cost",
) -> Dict[str, Any]:
    """Execute evaluation on unseen test split."""
    data_path = Path(data_dir)
    models_path = Path(models_dir)
    eval_path = Path(eval_dir)
    eval_path.mkdir(parents=True, exist_ok=True)

    print("Loading test dataset and champion models...")
    X_test = np.load(data_path / "cost_X_test.npy")
    y_test_log = np.load(data_path / "cost_y_test_log.npy")
    y_test_raw = np.load(data_path / "cost_y_test_raw.npy")

    best_model = joblib.load(models_path / "best_cost_model.joblib")
    quantile_models = joblib.load(models_path / "quantile_models.joblib")

    with open(data_path / "cost_features_metadata.json", "r") as f:
        meta = json.load(f)
    feature_names = meta["transformed_feature_names"]

    # 1. Point Predictions
    pred_test_log = best_model.predict(X_test)
    pred_test_raw = np.expm1(pred_test_log)

    overall_metrics = compute_metrics(
        y_test_raw, pred_test_raw, y_test_log, pred_test_log
    )

    print(
        f"Test Set Performance ($N = {len(y_test_raw)}$):\n"
        f"  MAE:       ${overall_metrics['mae_dollar']:,.2f}\n"
        f"  Median AE: ${overall_metrics['median_ae_dollar']:,.2f}\n"
        f"  RMSE:      ${overall_metrics['rmse_dollar']:,.2f}\n"
        f"  MAPE:      {overall_metrics['mape_pct']:.2f}%\n"
        f"  R²:        {overall_metrics['r2_dollar']:.4f} (Log R²: {overall_metrics['r2_log']:.4f})"
    )

    # 2. Prediction Interval Evaluation (90% Nominal Coverage)
    q05_log = quantile_models["q_5"].predict(X_test)
    q95_log = quantile_models["q_95"].predict(X_test)

    q05_raw = np.expm1(q05_log)
    q95_raw = np.expm1(q95_log)

    # Ensure monotonic interval
    lower_bound = np.minimum(q05_raw, pred_test_raw)
    upper_bound = np.maximum(q95_raw, pred_test_raw)

    inside_interval = (y_test_raw >= lower_bound) & (y_test_raw <= upper_bound)
    picp = float(np.mean(inside_interval) * 100.0)  # Prediction Interval Coverage Probability
    mpiw = float(np.mean(upper_bound - lower_bound))  # Mean Prediction Interval Width

    interval_metrics = {
        "nominal_confidence_pct": 90.0,
        "empirical_coverage_picp_pct": round(picp, 2),
        "mean_interval_width_mpiw_dollar": round(mpiw, 2),
        "median_interval_width_dollar": round(float(np.median(upper_bound - lower_bound)), 2),
    }

    print(
        f"\n90% Prediction Interval Calibration:\n"
        f"  Empirical Coverage (PICP): {picp:.2f}% (Target: 90.0%)\n"
        f"  Mean Interval Width (MPIW): ${mpiw:,.2f}"
    )

    # 3. Tiered Error Breakdown
    tiers = {
        "low_cost (< $5k)": y_test_raw < 5000,
        "mid_cost ($5k - $15k)": (y_test_raw >= 5000) & (y_test_raw < 15000),
        "high_cost ($15k - $30k)": (y_test_raw >= 15000) & (y_test_raw < 30000),
        "severe_loss (>= $30k)": y_test_raw >= 30000,
    }

    tier_metrics = {}
    for tier_name, mask in tiers.items():
        if np.sum(mask) > 0:
            m = compute_metrics(
                y_test_raw[mask],
                pred_test_raw[mask],
                y_test_log[mask],
                pred_test_log[mask],
            )
            tier_metrics[tier_name] = {
                "support": int(np.sum(mask)),
                "mae_dollar": m["mae_dollar"],
                "median_ae_dollar": m["median_ae_dollar"],
                "mape_pct": m["mape_pct"],
            }

    # 4. Feature Importance Extraction
    importances = None
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_)
    else:
        # Use permutation importance for HistGradientBoostingRegressor
        from sklearn.inspection import permutation_importance
        perm_res = permutation_importance(
            best_model, X_test[:600], y_test_log[:600], n_repeats=5, random_state=42
        )
        importances = perm_res.importances_mean

    top_features = []
    if importances is not None:
        idx_sorted = np.argsort(importances)[::-1]
        for idx in idx_sorted[:20]:
            top_features.append(
                {
                    "feature": feature_names[idx],
                    "importance": round(float(importances[idx]), 4),
                }
            )

    report = {
        "model_architecture": type(best_model).__name__,
        "test_records": len(y_test_raw),
        "overall_metrics": overall_metrics,
        "prediction_intervals": interval_metrics,
        "tiered_performance": tier_metrics,
        "top_features": top_features,
    }

    with open(eval_path / "phase9_test_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # -----------------------------------------------------------------------
    # Generate Research Visualizations
    # -----------------------------------------------------------------------
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Figure 1: Predicted vs Actual Scatter Plot
    plt.figure(figsize=(9, 8))
    plt.scatter(
        y_test_raw,
        pred_test_raw,
        alpha=0.25,
        color="#1f77b4",
        edgecolors="none",
        s=25,
    )
    max_val = min(80000, max(np.max(y_test_raw), np.max(pred_test_raw)))
    plt.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect Parity (y = x)")
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.title(
        f"Repair Cost Estimation: Predicted vs Actual (Test Set, N = {len(y_test_raw)})\n"
        f"MAE: ${overall_metrics['mae_dollar']:,.0f} | R²: {overall_metrics['r2_dollar']:.3f}"
    )
    plt.xlabel("Actual Ground Truth Repair Cost ($ USD)")
    plt.ylabel("Estimated Repair Cost ($ USD)")
    plt.legend()
    plt.tight_layout()
    fig1_path = eval_path / "predicted_vs_actual_cost.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved Figure 1 to: {fig1_path}")

    # Figure 2: Residuals Distribution
    residuals = y_test_raw - pred_test_raw
    plt.figure(figsize=(10, 5))
    sns.histplot(
        residuals,
        bins=60,
        kde=True,
        color="#2ca02c",
    )
    plt.axvline(0, color="red", linestyle="--", label="Zero Residual")
    plt.xlim(-25000, 25000)
    plt.title("Prediction Residual Distribution (Actual Cost - Predicted Cost)")
    plt.xlabel("Error ($ USD)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    fig2_path = eval_path / "cost_residuals_distribution.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved Figure 2 to: {fig2_path}")

    # Figure 3: Top 20 Feature Importances
    if top_features:
        feat_df = pd.DataFrame(top_features[:15])
        plt.figure(figsize=(10, 7))
        sns.barplot(
            data=feat_df,
            x="importance",
            y="feature",
            hue="feature",
            palette="mako",
            legend=False,
        )
        plt.title("Top 15 Feature Importances in Cost Regressor")
        plt.xlabel("Relative Feature Importance (MDI / Gain)")
        plt.ylabel("Engineered Feature")
        plt.tight_layout()
        fig3_path = eval_path / "feature_importances_top20.png"
        plt.savefig(fig3_path, dpi=300)
        plt.close()
        print(f"Saved Figure 3 to: {fig3_path}")

    # Figure 4: Sample Prediction Intervals (First 40 Test Samples Sorted by Cost)
    plt.figure(figsize=(12, 6))
    sample_indices = np.argsort(y_test_raw[:45])
    x_axis = np.arange(len(sample_indices))

    plt.plot(
        x_axis,
        y_test_raw[:45][sample_indices],
        "ro",
        markersize=6,
        label="Actual Cost ($)",
    )
    plt.plot(
        x_axis,
        pred_test_raw[:45][sample_indices],
        "b-",
        linewidth=1.5,
        label="Predicted Cost ($)",
    )
    plt.fill_between(
        x_axis,
        lower_bound[:45][sample_indices],
        upper_bound[:45][sample_indices],
        color="skyblue",
        alpha=0.4,
        label="90% Prediction Interval",
    )
    plt.title(
        f"Calibrated 90% Prediction Intervals on Random Test Samples (Empirical Coverage: {picp:.1f}%)"
    )
    plt.xlabel("Test Sample Index (Sorted by Ground Truth Cost)")
    plt.ylabel("Repair Cost ($ USD)")
    plt.legend()
    plt.tight_layout()
    fig4_path = eval_path / "prediction_intervals_sample.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved Figure 4 to: {fig4_path}")

    return report


if __name__ == "__main__":
    evaluate_test_set()
