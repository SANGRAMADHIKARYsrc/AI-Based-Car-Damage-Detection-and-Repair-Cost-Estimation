# Phase 5 — Fine-Grained Damage Classification

**Project:** AI-Based Car Damage Detection and Repair Cost Estimation Using Deep Learning  
**Date:** 2026-10-04  
**Status:** Complete  

---

## 1. Objective

Design, train, and evaluate a fine-grained damage classification module operating on localized bounding box crops extracted from vehicle damage detections.

### 1.1 Motivation & Research Question
In Phase 4, YOLOv8n achieved an overall test mAP50 of **0.6494**, but highlighted critical limitations for fine-grained damage categorization:
- **Subtle damage underperformance:** `crack` achieved only **0.2736 AP50** and **0.3862 F1** because thin fracture lines lose textural detail when downsampled to $640 \times 640$ full-vehicle images.
- **Visual ambiguity between classes:** Specular reflection on metallic surfaces causes confusion between minor dents and surface scratches.
- **Hierarchical Two-Stage Pipeline:** By coupling Phase 4 localization with a high-resolution classifier ($224 \times 224$ dedicated to the damage ROI with 15% context margin padding), the system resolves subtle structural textures while extracting visual feature embeddings for downstream repair cost modeling (Phases 8–10).

### 1.2 Target Performance Metrics

| Metric | Target Expectation | Phase 4 Detector Baseline | Phase 5 Achieved (Test Set) |
|--------|-------------------|--------------------------|----------------------------|
| Overall Test Accuracy | $\ge 70.0\%$ | N/A (detection task) | **94.13%** |
| Macro F1-Score | $\ge 0.6500$ | 0.6961 (detection boxes) | **0.9539** |
| Crack F1-Score | $\ge 0.4500$ | **0.3862** (critical bottleneck) | **0.9147** ($\mathbf{+0.5285}$) |
| Glass Shatter F1 | $\ge 0.9000$ | 0.9333 | **0.9787** |
| Tire Flat F1 | $\ge 0.8500$ | 0.8581 | **1.0000** |
| Dent F1 | $\ge 0.6000$ | 0.6267 | **0.9292** |
| Scratch F1 | $\ge 0.5500$ | 0.5616 | **0.9364** |
| Lamp Broken F1 | $\ge 0.7500$ | 0.7932 | **0.9645** |
| Per-Crop Inference | $< 10.0$ ms | RTX 2050 (AMP enabled) | **2.76 ms** |

---

## 2. Model Architecture & Rationale

**Backbone Selection: EfficientNet-B0**
- **Parameter count:** 5.3M (compact, low latency, matches mobile/edge deployment goals).
- **Compound scaling:** Balances network depth, width, and resolution for sharp texture discrimination.
- **Pretrained weights:** Initialized from ImageNet-1K (`DEFAULT` weights).
- **Bottleneck Feature Extraction:** Exposes a 1280-dimensional embedding vector via `.forward_features()` for the Phase 9 tabular repair-cost regressor.

### Classification Head Architecture
```
Input Crop [3 x 224 x 224]
       ↓
EfficientNet-B0 Backbone (1280 features)
       ↓
Adaptive Average Pooling
       ↓
Dropout (p = 0.3)
       ↓
Linear (1280 → 256)
       ↓
Batch Normalization + ReLU
       ↓
Dropout (p = 0.21)
       ↓
Linear (256 → 6 classes)
       ↓
Softmax Probabilities
```

---

## 3. Dataset & Preprocessing

### 3.1 Context Margin Padding
Bounding boxes directly cropped from vehicle images often lack surface context (e.g. whether a fracture line is on glass, metal sheet, or tire rubber). We applied a **15% context margin padding** around each bounding box:
$$\text{pad}_w = 0.15 \times w, \quad \text{pad}_h = 0.15 \times h$$
$$x_1 = \max(0, x_c - w/2 - \text{pad}_w), \quad y_1 = \max(0, y_c - h/2 - \text{pad}_h)$$
$$x_2 = \min(W, x_c + w/2 + \text{pad}_w), \quad y_2 = \min(H, y_c + h/2 + \text{pad}_h)$$

**Extracted Crop Inventory:**
- **Train:** 6,183 crops (`scratch`: 2,558, `dent`: 1,802, `crack`: 629, `lamp broken`: 494, `glass shatter`: 475, `tire flat`: 225)
- **Validation:** 1,739 crops
- **Test:** 784 crops
- **Total:** 8,706 crops indexed in `data/processed/damage_crops/crops_metadata.csv`.

### 3.2 Imbalance-Resilient Loss Formulation
CarDD exhibits an $11.3:1$ imbalance between `scratch` ($N=2,560$) and `tire flat` ($N=225$). We applied the smoothed square-root weights formulated in Phase 3:
- `dent`: 0.5882
- `scratch`: 0.4940
- `crack`: 0.9797
- `glass shatter`: 1.1469
- `lamp broken`: 1.1247
- `tire flat`: 1.6665

Loss function: Class-weighted Cross-Entropy with Label Smoothing ($\epsilon = 0.05$):
$$\mathcal{L}_{\text{CE}} = - \sum_{c=1}^C w_c \cdot y'_c \log(p_c), \quad y'_c = (1 - \epsilon)y_c + \frac{\epsilon}{C}$$

---

## 4. Training Configuration & Hyperparameters

- **Hardware:** NVIDIA GeForce RTX 2050 (4GB VRAM), PyTorch 2.4.1+cu124, AMP enabled.
- **Optimizer:** AdamW ($\text{lr} = 3 \times 10^{-4}$, $\text{weight\_decay} = 10^{-4}$).
- **Scheduler:** CosineAnnealingLR ($T_{\max} = 25$, $\eta_{\min} = 3 \times 10^{-6}$).
- **Batch Size:** 32.
- **Augmentation:** Random Resized Crop ($224 \times 224$), Random Horizontal Flip ($p=0.5$), Random Rotation ($\pm 15^\circ$), Color Jitter (brightness 0.2, contrast 0.2, saturation 0.2).
- **Execution Time:** 36.39 minutes across 25 epochs.
- **Best Validation Macro-F1:** **0.9551** (Epoch 23).

---

## 5. Evaluation Results

### 5.1 Test Set Per-Class Metrics ($N = 784$ crops)

| Class | Support | Precision | Recall | F1-Score | Phase 4 Baseline F1 | Gain |
|-------|---------|-----------|--------|----------|---------------------|------|
| **dent** | 236 | 0.9139 | 0.9449 | **0.9292** | 0.6267 | $+0.3025$ |
| **scratch** | 307 | 0.9379 | 0.9349 | **0.9364** | 0.5616 | $+0.3748$ |
| **crack** | 69 | 0.9833 | 0.8551 | **0.9147** | 0.3862 | $\mathbf{+0.5285}$ |
| **glass shatter** | 71 | 0.9857 | 0.9718 | **0.9787** | 0.9333 | $+0.0454$ |
| **lamp broken** | 69 | 0.9444 | 0.9855 | **0.9645** | 0.7932 | $+0.1713$ |
| **tire flat** | 32 | 1.0000 | 1.0000 | **1.0000** | 0.8581 | $+0.1419$ |
| **Overall / Macro** | **784** | **0.9609** | **0.9487** | **0.9539** | **0.6961** | $\mathbf{+0.2578}$ |

### 5.2 Test Confusion Matrix

```
                Predicted:
                dent  scratch  crack  glass  lamp  tire
True dent:      223     12       0      0      1     0    (Recall: 94.49%)
True scratch:    16    287       1      1      2     0    (Recall: 93.49%)
True crack:       3      7      59      0      0     0    (Recall: 85.51%)
True glass:       1      0       0     69      1     0    (Recall: 97.18%)
True lamp:        1      0       0      0     68     0    (Recall: 98.55%)
True tire:        0      0       0      0      0    32    (Recall: 100.0%)
```

### 5.3 Key Empirical Findings
1. **Resolution Bottleneck Solved for Cracks:** When evaluated within localized $224 \times 224$ crops, thin fracture lines are sharply defined. The model achieved **0.9833 precision** and **0.9147 F1** for cracks, resolving the Phase 4 bottleneck ($0.3862$ F1).
2. **Dent vs. Scratch Separation:** Dent vs. scratch was the largest remaining source of confusion (16 scratches predicted as dents, 12 dents predicted as scratches). This is physically consistent with real-world vehicle damage where dents often coincide with paint scratches.
3. **Perfect Tire Flat Identification:** $32/32$ ($100\%$ precision, $100\%$ recall) achieved through smoothed sqrt loss weighting.
4. **Sub-3ms Inference Speed:** At **2.76 ms per crop** on the RTX 2050, running classification on 5 detected damage regions in an image adds only $\sim 14\text{ ms}$, maintaining real-time throughput ($> 30\text{ FPS}$).

---

## 6. Two-Stage Hierarchical Inference Pipeline

We integrated the Phase 4 detector and Phase 5 classifier into `src/classification/two_stage_infer.py`:
- Stage 1: YOLOv8n detector localizes candidate damage bounding boxes.
- Stage 2: EfficientNet-B0 classifier processes padded crops, refining class predictions and confidence probabilities.
- Tested on held-out vehicle test images with visual outputs saved to `evaluation/classification/two_stage_samples/`.

---

## 7. Output Artifacts

```
models/classification/phase5_damage_classifier/
├── best_classifier.pt          # Best model checkpoint (Val Macro-F1 = 0.9551)
├── last_classifier.pt          # Epoch 25 final checkpoint
├── results.csv                 # Per-epoch loss, acc, macro-f1, per-class f1
├── results.png                 # Loss and accuracy curves
└── experiment_record.json      # Full environment & training metadata

data/processed/damage_crops/
├── crops_metadata.csv          # 8,706 samples with coordinates and paths
├── train/                      # 6,183 crops
├── val/                        # 1,739 crops
└── test/                       # 784 crops

evaluation/classification/
├── phase5_test_report.json     # Comprehensive test metrics JSON
├── phase5_confusion_matrix_test.png # Heatmap confusion matrix
└── two_stage_samples/          # Annotated sample images and summary JSON
```

---

## 8. Transition to Phase 6

- **Validation Gate:** Macro F1 target was $\ge 0.65$. Achieved **0.9539** ($\mathbf{+30.39\%}$ above gate).
- **Crack F1 Target:** Target was $\ge 0.45$. Achieved **0.9147** ($\mathbf{+46.47\%}$ above gate).
- **Decision:** **PASS.** Damage classification module is fully validated and locked.
- **Next Phase:** **Phase 6: Vehicle Component Identification** — identifying vehicle structural parts (bumper, hood, fender, door, quarter panel, roof) to contextualize detected damages for the repair cost engine.
