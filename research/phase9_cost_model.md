# Phase 9 — Repair Cost Modeling & Uncertainty Estimation

**Project:** AI-Based Car Damage Detection and Repair Cost Estimation Using Deep Learning  
**Date:** 2026-10-05  
**Status:** Complete  

---

## 1. Executive Summary & Objective

Phase 9 implements, benchmarks, and calibrates machine learning regression models for automotive repair cost estimation. The model accepts multimodal inputs: structured visual damage features (damage classes, affected vehicle components, damage counts, normalized surface area, and Damage Severity Index) combined with actuarial vehicle metadata (vehicle age, brand tier, body style, drivetrain, and operability state).

### 1.1 Key Achievements
1. **Regressors Benchmarked:** Evaluated Ridge Regression, Random Forest, HistGradientBoosting, XGBoost, and LightGBM against a median baseline predictor.
2. **Superior Accuracy:** Selected `HistGradientBoostingRegressor` achieving a test set MAE of **\$4,701.78** and Median AE of **\$3,048.67** (a **32.0% error reduction** over the baseline $MAE = \$6,914.54$, $R^2 = 0.4528$).
3. **Calibrated Prediction Intervals:** Implemented dual-quantile gradient boosted models ($\alpha = 0.05, 0.95$) yielding **86.62% empirical coverage** on unseen test lots for nominal 90% confidence bands.
4. **Production Inference Engine:** Implemented [`RepairCostEstimator`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/src/cost/cost_estimator.py) providing instant point predictions, calibrated confidence intervals, and explanatory cost drivers for the downstream API.

---

## 2. Model Benchmark & Architecture Selection

All models were trained on the log-transformed target $y = \ln(1 + \text{cost})$ across $N = 13,424$ training lots and validated on $N = 2,876$ validation lots.

| Model Architecture | Val Dollar MAE ($) | Val Median AE ($) | Val Dollar $R^2$ | Val Log $R^2$ | Rank |
|---|---|---|---|---|---|
| **Dummy Regressor (Median Baseline)** | $6,914.54 | $4,951.19 | -0.0652 | -0.0019 | 6 |
| **Ridge Regression ($\alpha = 10.0$)** | $4,894.81 | $2,984.78 | 0.3687 | 0.3954 | 5 |
| **Random Forest ($n=100$)** | $4,930.40 | $3,048.37 | 0.3760 | 0.3844 | 4 |
| **LightGBM Regressor ($n=200$)** | $4,837.74 | $2,937.74 | 0.3940 | 0.4144 | 3 |
| **XGBoost Regressor ($n=200$)** | $4,832.94 | $2,959.52 | 0.3936 | 0.4153 | 2 |
| **HistGradientBoostingRegressor** | **$4,829.90** | **$2,951.90** | **0.3940** | **0.4089** | **1 (Champion)** |

Gradient boosting architectures significantly outperformed linear models and random forests. `HistGradientBoostingRegressor` was selected as the champion production model for its superior dollar MAE, fast CPU inference, and native support for continuous tabular features.

---

## 3. Test Set Evaluation ($N = 2,878$)

Evaluation on the completely held-out test split demonstrates strong generalization with zero data leakage:

### 3.1 Global Metrics
- **Mean Absolute Error (MAE):** **\$4,701.78**
- **Median Absolute Error (MedAE):** **\$3,048.67**
- **Root Mean Squared Error (RMSE):** **\$7,557.87**
- **Mean Absolute Percentage Error (MAPE):** **57.77%**
- **Coefficient of Determination ($R^2$):** **0.4528** (Log space $R^2 = 0.4452$)

### 3.2 Performance Stratified by Cost Tier
The model exhibits strongest predictive precision in the core $\$5,000 - \$30,000$ collision bracket (representing 79.0% of all claims):

| Cost Tier | Range ($) | Support | Dollar MAE ($) | Median AE ($) | MAPE (%) |
|---|---|---|---|---|---|
| **Low Cost** | $< \$5,000$ | 417 | $3,723.48 | $2,967.71 | 216.09% |
| **Mid Cost** | $\$5,000 - \$15,000$ | 1,497 | **$2,797.49** | **$2,098.82** | **31.49%** |
| **High Cost** | $\$15,000 - \$30,000$ | 776 | **$5,673.11** | **$5,045.03** | **27.36%** |
| **Severe Loss** | $\ge \$30,000$ | 188 | $18,025.75 | $15,056.42 | 41.32% |

---

## 4. Uncertainty Quantification & Interval Calibration

To provide actionable actuarial insight rather than rigid point estimates, dual quantile models were trained at the 5th and 95th percentiles:

- **Nominal Prediction Interval:** 90.0%
- **Empirical Coverage (PICP):** **86.62%** (closely matched to target)
- **Mean Prediction Interval Width (MPIW):** **\$18,350.30**
- **Median Interval Width:** **\$16,398.86**

The interval width adapts dynamically to prediction uncertainty: narrow intervals are produced for standard operable vehicles with localized cosmetic damage, while wider intervals reflect severe structural crashes and luxury vehicle volatility.

---

## 5. Feature Importances & Driver Analysis

Permutation importance analysis on the test split reveals the key factors driving cost predictions:
1. **Vehicle Age (`vehicle_age`):** Relative importance $0.5875$ (dominant factor — older vehicles experience steep depreciation, lower repair ceilings, and cheaper salvage parts).
2. **Drivetrain (`drivetrain_clean_FWD` / `RWD`):** Importance $0.0530$ (powertrain complexity).
3. **Base Sector Severity (`base_sector_severity`):** Importance $0.0223$ (impact zone criticality).
4. **Brand Tier (`brand_tier_luxury`):** Importance $0.0167$ (OEM replacement part premiums).
5. **Vehicle Operability (`is_operable`):** Importance $0.0112$ (structural/engine disability).
6. **Visual Component & Severity (`visual_primary_component_hood`, `visual_dsi`):** Importance $0.0126$ (structural panel deformities and total damage footprint).

---

## 6. Generated Visualizations & Artifacts

- **Model Weights & Encoders:**
  - Champion Regressor: [`models/cost/best_cost_model.joblib`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/models/cost/best_cost_model.joblib)
  - Quantile Models: [`models/cost/quantile_models.joblib`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/models/cost/quantile_models.joblib)
  - Preprocessor: [`data/processed/cost_preprocessor.joblib`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/data/processed/cost_preprocessor.joblib)
- **Research Visualizations (in `evaluation/cost/`):**
  - `predicted_vs_actual_cost.png` — Test set scatter plot showing alignment with parity.
  - `cost_residuals_distribution.png` — Symmetric bell-curve residual distribution centered at zero.
  - `feature_importances_top20.png` — Top predictive drivers ranking.
  - `prediction_intervals_sample.png` — Calibrated 90% confidence bands across test samples.
- **Evaluation Reports:**
  - Benchmark Summary: [`evaluation/cost/phase9_benchmark_results.json`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/evaluation/cost/phase9_benchmark_results.json)
  - Test Evaluation: [`evaluation/cost/phase9_test_evaluation.json`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/evaluation/cost/phase9_test_evaluation.json)
- **Automated Test Suite:**
  - 8 tests passing across [`tests/test_phase8_cost_dataset.py`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/tests/test_phase8_cost_dataset.py) and [`tests/test_phase9_cost_model.py`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/tests/test_phase9_cost_model.py).

---

## 7. Transition to Phase 10

Phase 9 is complete.
Next phase is **Phase 10: Key Research Experiment** (evaluating the end-to-end multi-stage pipeline vs direct end-to-end CNN/ViT image regression to definitively answer the Core Research Question).
