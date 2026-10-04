"""Phase 9 — Cost Model Training, Benchmarking, and Interval Estimation.

Benchmarks Ridge, Random Forest, XGBoost, and LightGBM regressors on the
actuarial + visual damage feature matrix. Trains quantile models for 90%
prediction intervals and evaluates feature ablation (Actuarial vs Multimodal).
"""

import json
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, List, Tuple
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
import xgboost as xgb

from src.cost.dataset_builder import NUMERICAL_FEATURES


def compute_metrics(
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
) -> Dict[str, float]:
    """Compute comprehensive regression metrics in both raw dollar and log space."""
    mae_dollar = float(mean_absolute_error(y_true_raw, y_pred_raw))
    medae_dollar = float(np.median(np.abs(y_true_raw - y_pred_raw)))
    rmse_dollar = float(np.sqrt(mean_squared_error(y_true_raw, y_pred_raw)))
    mape_pct = float(mean_absolute_percentage_error(y_true_raw, y_pred_raw) * 100.0)
    r2_dollar = float(r2_score(y_true_raw, y_pred_raw))

    mae_log = float(mean_absolute_error(y_true_log, y_pred_log))
    rmse_log = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    r2_log = float(r2_score(y_true_log, y_pred_log))

    return {
        "mae_dollar": round(mae_dollar, 2),
        "median_ae_dollar": round(medae_dollar, 2),
        "rmse_dollar": round(rmse_dollar, 2),
        "mape_pct": round(mape_pct, 2),
        "r2_dollar": round(r2_dollar, 4),
        "mae_log": round(mae_log, 4),
        "rmse_log": round(rmse_log, 4),
        "r2_log": round(r2_log, 4),
    }


def train_and_benchmark(
    data_dir: str = "data/processed",
    output_dir: str = "models/cost",
    eval_dir: str = "evaluation/cost",
) -> Dict[str, Any]:
    """Train multiple regression architectures, select best, and fit quantile intervals."""
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    eval_path = Path(eval_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    eval_path.mkdir(parents=True, exist_ok=True)

    print("Loading preprocessed matrices...")
    X_train = np.load(data_path / "cost_X_train.npy")
    X_val = np.load(data_path / "cost_X_val.npy")

    y_train_log = np.load(data_path / "cost_y_train_log.npy")
    y_val_log = np.load(data_path / "cost_y_val_log.npy")

    y_train_raw = np.load(data_path / "cost_y_train_raw.npy")
    y_val_raw = np.load(data_path / "cost_y_val_raw.npy")

    with open(data_path / "cost_features_metadata.json", "r") as f:
        meta = json.load(f)
    feature_names = meta["transformed_feature_names"]

    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

    # 1. Benchmark Models
    models = {
        "Dummy (Median)": DummyRegressor(strategy="median"),
        "Ridge Regression": Ridge(alpha=10.0, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=16, min_samples_split=10, n_jobs=-1, random_state=42
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=150, max_depth=8, learning_rate=0.08, random_state=42
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        ),
    }

    benchmark_results = {}
    fitted_models = {}

    print("\n--- Model Training & Validation Benchmark ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train_log)
        fitted_models[name] = model

        # Predict log and invert to dollar
        pred_val_log = model.predict(X_val)
        pred_val_raw = np.expm1(pred_val_log)

        metrics = compute_metrics(y_val_raw, pred_val_raw, y_val_log, pred_val_log)
        benchmark_results[name] = metrics
        print(
            f"  -> Val MAE: ${metrics['mae_dollar']:,.2f} | "
            f"MedAE: ${metrics['median_ae_dollar']:,.2f} | "
            f"R²: {metrics['r2_dollar']:.4f} | Log R²: {metrics['r2_log']:.4f}"
        )

    # Select best model based on Validation R² (log space) / MAE
    best_model_name = min(
        benchmark_results.keys(),
        key=lambda k: benchmark_results[k]["mae_dollar"]
        if k != "Dummy (Median)"
        else 999999,
    )
    best_model = fitted_models[best_model_name]
    print(f"\nBest Model: {best_model_name} (MAE: ${benchmark_results[best_model_name]['mae_dollar']:,.2f})")

    # 2. Train Quantile Models for Prediction Intervals (alpha = 0.05, 0.50, 0.95 -> 90% interval)
    print("\nFitting Quantile LightGBM models for 90% confidence intervals...")
    quantile_models = {}
    for alpha in [0.05, 0.50, 0.95]:
        q_model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            verbose=-1,
            random_state=42,
        )
        q_model.fit(X_train, y_train_log)
        quantile_models[f"q_{int(alpha * 100)}"] = q_model

    # 3. Feature Ablation: Actuarial-only vs Actuarial + Visual
    print("\n--- Running Feature Ablation Experiment ---")
    # Identify indices of visual features
    visual_kw = ["visual_", "sector_cost_weight", "base_sector_severity"]
    actuarial_indices = [
        i
        for i, col in enumerate(feature_names)
        if not any(k in col for k in visual_kw)
    ]
    multimodal_indices = list(range(len(feature_names)))

    print(
        f"Ablation setup: Actuarial features = {len(actuarial_indices)} cols | "
        f"Multimodal = {len(multimodal_indices)} cols"
    )

    # Train LightGBM on actuarial only
    model_actuarial = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1,
    )
    model_actuarial.fit(X_train[:, actuarial_indices], y_train_log)
    pred_act_log = model_actuarial.predict(X_val[:, actuarial_indices])
    metrics_actuarial = compute_metrics(
        y_val_raw, np.expm1(pred_act_log), y_val_log, pred_act_log
    )

    # Train LightGBM on multimodal (all)
    metrics_multimodal = benchmark_results["LightGBM"]

    ablation_results = {
        "actuarial_only": {
            "num_features": len(actuarial_indices),
            **metrics_actuarial,
        },
        "multimodal_actuarial_visual": {
            "num_features": len(multimodal_indices),
            **metrics_multimodal,
        },
        "gain": {
            "mae_reduction_dollar": round(
                metrics_actuarial["mae_dollar"] - metrics_multimodal["mae_dollar"], 2
            ),
            "r2_gain": round(
                metrics_multimodal["r2_dollar"] - metrics_actuarial["r2_dollar"], 4
            ),
        },
    }

    print(
        f"Ablation Result:\n"
        f"  Actuarial Only Val MAE: ${metrics_actuarial['mae_dollar']:,.2f} | R²: {metrics_actuarial['r2_dollar']:.4f}\n"
        f"  Multimodal Val MAE:     ${metrics_multimodal['mae_dollar']:,.2f} | R²: {metrics_multimodal['r2_dollar']:.4f}\n"
        f"  Visual Feature Contribution: -${ablation_results['gain']['mae_reduction_dollar']:,.2f} MAE reduction!"
    )

    # 4. Save Artifacts
    best_model_path = out_path / "best_cost_model.joblib"
    joblib.dump(best_model, best_model_path)

    quantile_path = out_path / "quantile_models.joblib"
    joblib.dump(quantile_models, quantile_path)

    summary = {
        "best_model_name": best_model_name,
        "benchmark_val_results": benchmark_results,
        "ablation_experiment": ablation_results,
    }

    with open(eval_path / "phase9_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved best model to: {best_model_path}")
    print(f"Saved quantile models to: {quantile_path}")
    print(f"Saved benchmark results to: {eval_path / 'phase9_benchmark_results.json'}")

    return summary


if __name__ == "__main__":
    train_and_benchmark()
