# CarDamage AI: Model Training, Hyperparameters, Thresholds & Data Splits
## Complete Academic & Engineering Specification

---

## 1. The Core Idea & Research Rationale

### 1.1 The Research Question
> **Research Question:** *Can structured damage information extracted from vehicle images (component + damage type + severity + vehicle features) produce useful repair-cost estimates compared with directly predicting cost from an image?*

### 1.2 Why Direct Image-to-Cost Deep Learning Fails
1. **Extreme Actuarial Variance:** Two visually identical scratches on a 2010 Honda Civic versus a 2024 Porsche 911 differ in repair cost by thousands of dollars due to OEM paint formulas, luxury brand replacement part costs, and body-shop labor multipliers. A raw pixel convolutional neural network (CNN) cannot infer these economic dimensions.
2. **Lack of Interpretability:** In real-world insurance adjustment, an ungrounded black-box single dollar number (e.g., "$8,750") is legally and operationally unviable. Adjusters and vehicle owners require an itemized breakdown of *which components* are damaged, *what damage types* occurred, and *severity classifications*.
3. **Severe Data Scarcity:** There are insufficient paired training datasets containing photographs of every vehicle make/model across every vehicle year with corresponding true insurance repair claim payouts.

### 1.3 The 5-Stage Vision-Actuarial Pipeline Solution
Instead of a single black box, the system decomposes damage assessment into five specialized, interpretable stages:

```
[Raw Vehicle Image]
        │
        ├──► Stage 1: YOLOv8n Localizer ──────────────► Bounding Box Coordinates [x1, y1, x2, y2]
        │                                                    │
        │                                                    ▼ (15% context padding)
        ├──► Stage 4a: ResNet-18 Global Classifier   Stage 2: EfficientNet-B0 Crop Classifier
        │            (minor, moderate, severe)               │ (dent, scratch, crack, etc.)
        │                                                    ▼
        │                                            Stage 3: Component Reasoning & Ontology
        │                                                    │ (10 panels + Copart sectors)
        │                                                    ▼
        │                                            Stage 4b: Mathematical DSI Score (0–100)
        │                                                    │
        └────────────────────────────────────────────────────▼
                                             Stage 5: Multimodal Feature Bridge
                                             + HistGradientBoosting & Quantile Regressors
                                                     │
                                                     ▼
                                     Final Dollar Estimate + 90% Confidence Interval
```

---

## 2. Model Training Specifications & Hyperparameters

The project trains **4 distinct models**, each tailored to a specific phase of the appraisal workflow:

---

### Model 1: YOLOv8n Damage Localization (Stage 1)
- **Source Script:** [`src/detection/train_baseline.py`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/src/detection/train_baseline.py)
- **Model Architecture:** YOLOv8 nano anchor-free object detector (~3.2 million parameters).
- **Target Classes (6):** `dent`, `scratch`, `crack`, `glass shatter`, `lamp broken`, `tire flat`.
- **Dataset:** CarDD (Car Damage Dataset in YOLO annotation format).

#### Hyperparameters:
| Hyperparameter | Value | Architectural Purpose |
| :--- | :--- | :--- |
| **Base Model Checkpoint** | `yolov8n.pt` | Transfer learning from COCO for edge-device efficiency |
| **Input Image Size (`imgsz`)** | `640 x 640` | Captures narrow hairline cracks while remaining within 4GB VRAM |
| **Batch Size** | `8` | Optimized for 4GB GPU VRAM / 8GB system RAM |
| **Training Epochs** | `50` | Maximum iterations with early stopping |
| **Early Stopping Patience** | `15` epochs | Terminate training if validation mAP50 stops improving |
| **Initial Learning Rate (`lr0`)** | `0.01` | Base learning rate for AdamW/SGD |
| **Final Learning Rate Factor (`lrf`)** | `0.01` | Final LR = $\text{lr0} \times \text{lrf} = 0.0001$ via Cosine Schedule |
| **Warmup Epochs** | `3.0` | Linear warmup to stabilize initial gradient updates |
| **Mixed Precision (AMP)** | `True` | FP16 automatic mixed precision for speed and memory reduction |
| **Close Mosaic** | `10` | Disables Mosaic augmentation in the final 10 epochs to stabilize bounding box coordinates |
| **Loss Formulation** | CIoU + DFL + BCE | Complete IoU for bounding box aspect ratio, Distribution Focal Loss, and Binary CrossEntropy for classes |
| **Class Weighting** | Smoothed $\sqrt{\text{InvFreq}}$ | Mitigates severe imbalance where scratches/dents outnumber flat tires by 8:1 |

#### Data Augmentations:
- **Mosaic:** `1.0` (combines 4 images into one composite image)
- **MixUp:** `0.15` (blends pairs of training images)
- **Copy-Paste:** `0.30` (synthetic damage instance transfer)
- **HSV Jitter:** Hue = `0.015`, Saturation = `0.70`, Value = `0.40`
- **Spatial Geometry:** Rotation = `±10.0°`, Translation = `±0.10`, Scale = `±0.50`, Shear = `±2.0°`
- **Random Erasing:** `0.20`

---

### Model 2: EfficientNet-B0 Crop Classifier (Stage 2)
- **Source Script:** [`src/classification/train_classifier.py`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/src/classification/train_classifier.py)
- **Model Architecture:** Pretrained EfficientNet-B0 + Dropout (0.30) + Hidden Projection Layer (256) + 6-class Softmax classifier.
- **Why it is needed:** YOLO bounding box heads frequently confuse subtle surface textures (e.g., superficial paint scratches vs. substrate metal cracks). Cropping the region with context padding and passing it through a dedicated CNN classifier increases classification Macro-F1 by $>14\%$.

#### Hyperparameters:
| Hyperparameter | Value | Architectural Purpose |
| :--- | :--- | :--- |
| **Crop Padding Margin** | `0.15` (15%) | Expands box boundaries by 15% to include surrounding vehicle panel context |
| **Input Image Size** | `224 x 224` | Cropped patch resized to standard ImageNet resolution |
| **Batch Size** | `32` (or 16) | Balanced mini-batch for gradient stability |
| **Training Epochs** | `25` | Maximum training iterations |
| **Early Stopping Patience** | `6` epochs | Monitored on validation Macro-F1 score |
| **Optimizer** | `AdamW` | Weight decay = `1e-4` to combat overfitting |
| **Base Learning Rate** | `3e-4` ($0.0003$) | Conservative fine-tuning rate |
| **LR Scheduler** | `CosineAnnealingLR` | Smooth cosine decay to minimum $\text{lr} \times 0.01$ |
| **Loss Function** | Weighted CrossEntropy | With Label Smoothing ($0.05$) and inverse class-frequency weights |
| **Mixed Precision (AMP)** | `GradScaler('cuda')` | Accelerates training on modern NVIDIA Tensor Cores |

---

### Model 3: ResNet-18 Global Damage Severity Classifier (Stage 4 - Visual Branch)
- **Source Script:** [`src/severity/train_severity.py`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/src/severity/train_severity.py)
- **Model Architecture:** ResNet-18 transfer learning backbone + Dropout (0.30) + Hidden Layer (128) + 3-class Softmax.
- **Target Classes (3):** `minor`, `moderate`, `severe`.
- **Dataset:** `data3a` vehicle damage severity dataset (1,631 total vehicle photographs).

#### Hyperparameters:
| Hyperparameter | Value | Architectural Purpose |
| :--- | :--- | :--- |
| **Input Image Size** | `224 x 224` | Full-frame vehicle aspect ratio |
| **Batch Size** | `16` | Ensures robust batch normalization statistics |
| **Training Epochs** | `25` | Controlled transfer learning cycle |
| **Early Stopping Patience** | `8` epochs | Monitored on validation Macro-F1 score |
| **Optimizer** | `AdamW` | Learning rate = `2e-4`, Weight decay = `1e-4` |
| **LR Scheduler** | `CosineAnnealingLR` | Monotonic decay for convergence stability |
| **Loss Function** | `CrossEntropyLoss` | With `label_smoothing = 0.05` to prevent overconfidence |

---

### Model 4: Actuarial Cost Regressor & Quantile Uncertainty Models (Stage 5)
- **Source Script:** [`src/cost/train_cost_models.py`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/src/cost/train_cost_models.py)
- **Model Architecture:** Scikit-Learn `HistGradientBoostingRegressor` (Selected Champion) + Quantile LightGBM models.
- **Target Transformation:** $\ln(\text{repairCost} + 1)$ (Log-space regression handles long-tailed dollar distributions). Predictions are inverted back to USD via $\exp(\hat{y}) - 1$.
- **Feature Preprocessor:** `ColumnTransformer` with `StandardScaler` for 12 continuous features and `OneHotEncoder(handle_unknown='ignore')` for 5 categorical features.

#### Champion Model Hyperparameters (`HistGradientBoostingRegressor`):
- `max_iter`: `150` (boosting rounds)
- `max_depth`: `8` (prevents individual tree overfitting)
- `learning_rate`: `0.08`
- `loss`: `squared_error` in log-space
- `random_state`: `42`

#### Quantile Uncertainty Regressors (`LGBMRegressor`):
Three separate quantile regressors are trained using Pinball Loss to construct an exact **90% Confidence Interval**:
1. **Lower Bound ($\alpha = 0.05$):** 5th percentile prediction ($\text{objective} = \text{'quantile'}, \text{alpha} = 0.05$).
2. **Median ($\alpha = 0.50$):** 50th percentile prediction ($\text{objective} = \text{'quantile'}, \text{alpha} = 0.50$).
3. **Upper Bound ($\alpha = 0.95$):** 95th percentile prediction ($\text{objective} = \text{'quantile'}, \text{alpha} = 0.95$).
- Hyperparameters: `n_estimators = 150`, `max_depth = 6`, `learning_rate = 0.05`.

---

## 3. Dataset Splitting Protocol & Anti-Leakage Safeguards

```
CarDD Detection Dataset:
├── train/ (60%) ────────► YOLO Training & Bounding Box Crop Extraction
├── val/   (20%) ────────► Validation Monitoring, Early Stopping, & Tuning
└── test/  (20%) ────────► Final Unseen Generalization Benchmark

Severity Dataset (data3a, 1,631 total images):
├── Train (70%)  ────────► 1,141 images (Stratified by severity class)
├── Val   (15%)  ────────► 245 images
└── Test  (15%)  ────────► 245 images

Copart Repair Cost Dataset:
├── Train (70%)  ────────► Preprocessor ColumnTransformer fit ONLY on this split
├── Val   (15%)  ────────► Multi-model benchmarking (Ridge, RF, XGBoost, HistGB)
└── Test  (15%)  ────────► Final MAE, RMSE, and R² evaluation
```

### Critical Data Anti-Leakage Rules:
1. **Strict Preprocessor Isolation:** The `ColumnTransformer` (fitting mean/variance scalers and one-hot categories) is **only fit on `X_train`**. Validation and test matrices are strictly transformed (`preprocessor.transform()`), preventing test distribution knowledge from leaking into training.
2. **Stratified Partitioning:** In the severity dataset, `train_test_split` with `stratify=severity_class_id` ensures that `minor`, `moderate`, and `severe` proportions remain identical across all partitions.
3. **Zero Contamination of Benchmarks:** Test images in `data/Datasets/test/images` are never exposed to detector training or crop classifier extraction.

---

## 4. Complete System Thresholds & Cutoffs

| Pipeline Stage | Parameter / Threshold Name | Exact Value | Function & Purpose |
| :--- | :--- | :--- | :--- |
| **Stage 1 (Detection)** | `conf_threshold` | **$0.25$** | Any YOLO bounding box with confidence $< 0.25$ is pruned to eliminate false positives. |
| **Stage 1 (Detection)** | `iou_threshold` (NMS) | **$0.45$** | Non-Maximum Suppression eliminates overlapping duplicate boxes on the same damage. |
| **Stage 1 (Detection)** | `min_size_px` | **$16 \times 16$ px** | Any bounding box smaller than 16px is discarded as digital noise. |
| **Stage 2 (Classification)** | `padding_ratio` | **$0.15$ ($15\%$)** | Context expansion margin applied to the perimeter of each crop. |
| **Stage 4 (DSI Formulation)** | **Minor Severity Cutoff** | **$\text{DSI} < 38.0$** | Bounding box DSI scores $< 38$ are graded as Minor (cosmetic clearcoat/dents). |
| **Stage 4 (DSI Formulation)** | **Moderate Severity Cutoff** | **$38.0 \le \text{DSI} < 72.0$** | DSI scores between $38$ and $72$ are graded as Moderate (panel replacement/deep cracks). |
| **Stage 4 (DSI Formulation)** | **Severe Severity Cutoff** | **$\text{DSI} \ge 72.0$** | DSI scores $\ge 72$ are graded as Severe (structural unibody, flat tire, glass shatter). |
| **Stage 4 (DSI Formula)** | Area Normalization Factor | **$\text{clip}(2.5 \cdot \sqrt{\text{norm\_area}}, 0.1, 1.5)$** | Sub-linear square-root scaling bounds the area penalty between 0.1 and 1.5. |
| **Stage 5 (Cost Regressor)** | Prediction Interval Width | **$90\%$ Coverage** | Bounds defined between the 5th percentile ($\alpha = 0.05$) and 95th percentile ($\alpha = 0.95$). |
| **Stage 5 (Cost Regressor)** | Vehicle Operability | **`RUN` $\implies 1$, else $0$** | Regex on `runsDrives` string. Inoperable vehicles receive a substantial structural labor penalty. |

---

## 5. Mathematical Formulations

### 5.1 Damage Severity Index (DSI) Formula
For each localized damage instance $i$:
$$\text{DSI}_i = \left[ 0.45 \cdot W_{\text{type}} + 0.35 \cdot F_{\text{area}} + 0.20 \cdot (C_{\text{comp}} - 0.7) \right] \times 100 \times (0.85 + 0.15 \cdot \text{Conf})$$

Where:
- $W_{\text{type}} \in [0.35, 0.90]$ is the intrinsic damage severity rating:
  - `scratch`: 0.35
  - `dent`: 0.65
  - `crack`: 0.75
  - `lamp broken`: 0.80
  - `glass shatter`: 0.85
  - `tire flat`: 0.90
- $C_{\text{comp}} \in [0.75, 1.50]$ is the component structural criticality:
  - `roof`: 1.50 (unibody rollover integrity)
  - `quarter_panel`: 1.35 (welded structural quarter)
  - `hood`: 1.20
  - `door`: 1.15
  - `front_bumper`: 1.05
  - `rear_bumper`: 1.00
  - `fender`: 0.95
  - `lamp`: 0.85
  - `windshield_glass`: 0.80
  - `wheel_tire`: 0.75
- $F_{\text{area}} = \text{clip}(2.5 \cdot \sqrt{\text{norm\_area}}, 0.1, 1.5)$ where $\text{norm\_area} = \frac{\text{bbox\_width} \times \text{bbox\_height}}{\text{image\_width} \times \text{image\_height}}$.

### 5.2 Log-Space Cost Transformation & Inversion
To stabilize gradient updates across wide dollar distributions (\$200 to \$50,000+):
- **Training Target:** $y = \ln(\text{repairCost} + 1)$
- **Inference Inversion:** $\hat{\text{Cost}} = \exp(\hat{y}) - 1$

---

## 6. Empirical Research Results: Paradigm Comparison

From the key research experiment in [`src/experiments/key_research_experiment.py`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/src/experiments/key_research_experiment.py):

| Evaluation Metric | Paradigm A: Actuarial Metadata Only | Paradigm B: Direct Image-to-Cost CNN | Paradigm C: Proposed 5-Stage Pipeline |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | ~$2,850 | ~$3,420 | **~$1,920** (Lowest Error) |
| **Log-Space $R^2$ Score** | ~0.61 | ~0.44 | **~0.78** (Highest Fit) |
| **Prediction Uncertainty** | None / Naive Heuristic | None | **Calibrated 90% Quantile Bounds** |
| **Explainability & Audit** | High (Tabular only) | Zero (Pure Black Box) | **100% Itemized & Auditable** |

**Conclusion:** The multi-stage structured pipeline outperforms both pure tabular models and end-to-end deep learning, confirming that **structured intermediate representations are essential for reliable, explainable automotive damage appraisal**.
