# Phase 8 — Repair-Cost Dataset Preparation & Actuarial Feature Engineering

**Project:** AI-Based Car Damage Detection and Repair Cost Estimation Using Deep Learning  
**Date:** 2026-10-05  
**Status:** Complete  

---

## 1. Objective

Resolve **Gap 4 (Disconnect Between Computer Vision Damage Datasets and Actuarial Repair Cost Records)** identified in the Literature Review by building a standardized, multimodal repair-cost dataset. This phase bridges computer vision predictions (Phases 4–7: damage types, components, severity, DSI, area ratios) with actuarial insurance claims from Copart ($N = 19,178$) to enable rigorous tabular and multimodal cost modeling in Phase 9.

### 1.1 Core Research Question Addressed
> *"Can structured damage information extracted from vehicle images (component + damage type + severity + vehicle features) produce useful repair-cost estimates compared with directly predicting cost from an image?"*

To evaluate this in Phase 9, Phase 8 structures the data into modular feature subsets:
1. **Actuarial Baseline Features:** Vehicle age, brand tier, body style, drivetrain, operability.
2. **Visual Damage Bridge Features:** Primary damaged component, primary damage sector, composite sector labor/parts multiplier, damage instance count, damaged area ratio, Damage Severity Index (DSI), and discrete severity grade.

---

## 2. Dataset Structure & Splits

The cleaned Copart auction claim records are partitioned strictly into non-overlapping splits with zero data leakage:

| Split | Number of Lots | Proportion | Mean Repair Cost ($) | Median Repair Cost ($) |
|---|---|---|---|---|
| **Train** | 13,424 | 70.0% | $13,772.50 | $10,920.38 |
| **Validation** | 2,876 | 15.0% | $13,546.73 | $10,876.10 |
| **Test** | 2,878 | 15.0% | $13,601.88 | $10,945.50 |
| **Total** | **19,178** | **100.0%** | **$13,713.06** | **$10,917.68** |

- Verified lot-ID intersection between splits: **0 lots (0.0% leakage)**.

---

## 3. Feature Engineering Specification

### 3.1 Vehicle Actuarial Features
- **Vehicle Age & Non-Linear Transformations:**
  $$\text{age} = 2026 - \text{year}, \quad \text{age}^2, \quad \ln(1 + \text{age})$$
- **Brand Tier Categorization:**
  - *Luxury ($N = 1,769$ train):* Mercedes-Benz, BMW, Audi, Porsche, Land Rover, Lexus, Tesla, Volvo, Cadillac, Lincoln, etc.
  - *Premium ($N = 1,677$ train):* Jeep, GMC, Ram, Chrysler, Buick, Mini, Volkswagen.
  - *Standard ($N = 9,978$ train):* Toyota, Honda, Ford, Chevrolet, Nissan, Hyundai, Kia, Subaru, etc.
- **Canonical Body Style Normalization:** Grouped into 8 standard vehicle architectures: `sedan`, `suv`, `truck`, `hatchback`, `coupe`, `van`, `wagon`, `other`.
- **Drivetrain Normalization:** `FWD`, `AWD`, `RWD`, `4WD`, `UNKNOWN`.
- **Vehicle Operability Indicators:**
  - `is_operable`: 1 if verified "Run & Drive", 0 otherwise.
  - `starts_only`: 1 if vehicle engine starts but does not drive.

### 3.2 Visual Damage Bridge Features
- **Primary Damage Sector:** Mapped to 10 standard vehicle impact zones (`FRONT END`, `REAR END`, `SIDE`, `ROLLOVER`, `ALL OVER`, `MINOR DENT/SCRATCHES`, etc.).
- **Sector Labor & Parts Cost Multiplier (`sector_cost_weight`):** Weighted composite multiplier from Phase 6 automotive component ontology (`COMPONENT_COST_WEIGHTS` in `src/component/ontology.py`).
- **Base Sector Severity (`base_sector_severity`):** Actuarial severity prior ($0.35$ for minor scratches to $0.95$ for rollovers).
- **Damage Instance Count (`visual_damage_count`):** Number of localized damage bounding boxes.
- **Damaged Area Ratio (`visual_damaged_area_ratio`):** Aggregate normalized surface area of detected damage footprints relative to vehicle frame.
- **Damage Severity Index (`visual_dsi`):** Unified continuous severity metric ($10 \le \text{DSI} \le 99$) computed via `src/severity/dsi.py`.
- **Discrete Severity Grade (`visual_severity_grade`):** 0: Minor ($\text{DSI} < 38$), 1: Moderate ($38 \le \text{DSI} < 72$), 2: Severe ($\text{DSI} \ge 72$).
- **Primary Critical Component (`visual_primary_component`):** Component bearing the highest individual DSI score.

---

## 4. Statistical Findings & Exploratory Data Analysis

### 4.1 Target Variable Distribution & Log Transformation
- **Raw Repair Cost:** Extreme right skew ($\text{skewness} = 3.26$, $\text{kurtosis} = 21.28$). Minimum: \$100, Median: \$10,920, Maximum: \$150,000, IQR: \$10,977.
- **Log-Transformed Target $y = \ln(1 + \text{cost})$:** Normalized distribution ($\text{skewness} = -0.67$, $\text{kurtosis} = 2.83$, $\mu = 9.27$, $\sigma = 0.75$).
- **Conclusion:** Standard regression models (Linear, Ridge, MLP) require log target training to satisfy homoscedasticity assumptions; tree ensembles benefit from stability.

### 4.2 Key Actuarial Drivers
1. **Brand Tier Impact:**
   - Luxury vehicles average **\$17,042** (median \$13,471) vs Standard vehicles **\$13,182** (median \$10,504) — a **+28.2% higher median cost** reflecting expensive OEM sensors, composite panels, and higher dealer labor rates.
2. **Operability Discrepancy:**
   - Inoperable vehicles average **\$17,348** (median \$13,968) vs Operable vehicles **\$11,903** (median \$9,942).
   - Independent samples Welch's $t$-test: $t = -23.83, \; p = 3.63 \times 10^{-120}$ (statistically decisive). Structural frame and mechanical powertrain damage dramatically elevate repair costs.
3. **Vehicle Age Correlation:**
   - Strong inverse relationship ($r = -0.538$, Spearman $\rho = -0.619$). Newer vehicles cost substantially more to repair due to integrated ADAS sensors in bumpers/windshields, complex multi-material alloys, and lower salvage parts availability.
4. **Visual Damage Feature Correlation:**
   - `visual_dsi`: $r = +0.185$, $\rho = +0.170$ ($p < 10^{-87}$).
   - `visual_damaged_area_ratio`: $r = +0.170$, $\rho = +0.168$ ($p < 10^{-85}$).
   - `base_sector_severity`: $r = +0.130$ ($p < 10^{-50}$).
   - Mutual Information with target: `vehicle_age` ($0.272$), `is_operable` ($0.036$), `visual_dsi` ($0.029$), `base_sector_severity` ($0.027$).

---

## 5. Artifacts & Code Implementations

1. **Feature Engineering Engine:** [`src/cost/feature_engineering.py`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/src/cost/feature_engineering.py)
   - Contains categorical normalizers, brand tier mapper, domain sector multipliers, and end-to-end multimodal bridge `extract_cost_features_from_visual_pipeline`.
2. **Dataset Builder:** [`src/cost/dataset_builder.py`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/src/cost/dataset_builder.py)
   - Fits scikit-learn `ColumnTransformer` (StandardScaler + OneHotEncoder) strictly on training data.
   - Exports high-performance preprocessed design matrices:
     - `data/processed/cost_X_train.npy` ($13,424 \times 57$)
     - `data/processed/cost_X_val.npy` ($2,876 \times 57$)
     - `data/processed/cost_X_test.npy` ($2,878 \times 57$)
     - `data/processed/cost_y_*_log.npy` and `cost_y_*_raw.npy`
     - Serialized transformer: `data/processed/cost_preprocessor.joblib`
     - Feature schema: `data/processed/cost_features_metadata.json`
3. **Statistical Analysis & Visualizations:** [`src/cost/eda_report.py`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/src/cost/eda_report.py)
   - Report: [`evaluation/cost/phase8_dataset_report.json`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/evaluation/cost/phase8_dataset_report.json)
   - Figures:
     - `evaluation/cost/repair_cost_distributions.png`
     - `evaluation/cost/cost_by_damage_sector.png`
     - `evaluation/cost/cost_by_brand_tier.png`
     - `evaluation/cost/feature_correlation_matrix.png`
4. **Automated Test Suite:** [`tests/test_phase8_cost_dataset.py`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/tests/test_phase8_cost_dataset.py)
   - 4 unit tests passing: schema validation, preprocessor repeatability/serialization, leakage audit, visual extractor compatibility.

---

## 6. Transition to Phase 9

Phase 8 is complete. The structured feature matrices and multimodal bridge are validated.
Next phase is **Phase 9: Cost Model** (training Gradient Boosting / XGBoost / LightGBM / MLP regressors, computing MAE, RMSE, and $R^2$, and constructing confidence intervals for repair cost predictions).
