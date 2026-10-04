# Phase 7 — Damage Severity Assessment

**Project:** AI-Based Car Damage Detection and Repair Cost Estimation Using Deep Learning  
**Date:** 2026-10-04  
**Status:** Complete  

---

## 1. Objective

Train and evaluate a 3-class damage severity classifier (minor, moderate, severe) and formalize a Damage Severity Index (DSI) combining visual classification with instance-level geometric and material reasoning.

### 1.1 Research Context (Gap 3)
> **Gap 3: No Standardized Severity Ground Truth**  
> Severity labels (minor/moderate/severe) do not exist in CarDD or VehiDE. The `data3a` dataset (1,631 images) provides the only available ground-truth severity annotations for this pipeline.

### 1.2 Dual Severity Architecture
Phase 7 implements two complementary severity mechanisms:
1. **Global Visual Classifier** — ResNet18 trained on full vehicle images from `data3a` to produce image-level severity grades.
2. **Instance-Level DSI** — Analytical formula fusing damage type severity weights, component structural criticality, and normalized bounding box area ratios.

---

## 2. Dataset & Splits

| Split | Minor | Moderate | Severe | Total |
|-------|-------|----------|--------|-------|
| **Train** | 374 | 376 | 391 | **1,141** |
| **Val** | 37 | 37 | 39 | **245** |  
| **Test** | 80 | 81 | 84 | **245** |
| **Total** | **534** | **538** | **559** | **1,631** |

Near-perfectly balanced across all three classes ($< 5\%$ deviation).

---

## 3. Training Configuration

- **Backbone:** ResNet18 (pretrained ImageNet-1K, 11.2M params)
- **Head:** Dropout(0.3) -> Linear(512, 128) -> BN -> ReLU -> Dropout(0.21) -> Linear(128, 3)
- **Loss:** Cross-Entropy with label smoothing ($\epsilon = 0.05$)
- **Optimizer:** AdamW ($\text{lr} = 3 \times 10^{-4}$, $\text{weight\_decay} = 10^{-4}$)
- **Scheduler:** CosineAnnealingLR ($T_{\max} = 25$)
- **AMP:** Enabled (RTX 2050, 4GB VRAM)
- **Early Stopping:** Patience 8 on val macro-F1; triggered at epoch 16.
- **Training Time:** 243.3 seconds (16 epochs).

---

## 4. Test Set Results ($N = 245$)

| Class | Support | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| **minor** | 80 | 0.6944 | 0.6250 | **0.6579** |
| **moderate** | 81 | 0.4867 | 0.6790 | **0.5670** |
| **severe** | 84 | 0.7833 | 0.5595 | **0.6528** |
| **Overall** | **245** | — | — | **Macro-F1: 0.6259** |

- **Accuracy:** 62.04%
- **Inference Latency:** 3.73 ms/image

### 4.1 Analysis
- **Moderate class is the bottleneck** (F1: 0.567). This is consistent with the inherent subjectivity of the minor-moderate boundary — the same dent may be labeled "minor" or "moderate" depending on annotator perspective.
- **Severe class has highest precision** (0.783) — structurally compromised vehicles are visually distinct.
- **Minor class** performs reasonably (F1: 0.658) — cosmetic scratches/paint chips are recognizable.
- **Performance is adequate for the pipeline** — severity is used as one feature among many in the tabular cost model (Phase 8/9), not as a standalone binary decision. The DSI formula provides complementary analytical severity grading.

---

## 5. Damage Severity Index (DSI)

Instance-level DSI formula (`src/severity/dsi.py`):
$$\text{DSI} = \left( \alpha_{\text{type}} \cdot 0.45 + \alpha_{\text{area}} \cdot 0.35 + (\alpha_{\text{crit}} - 0.7) \cdot 0.20 \right) \times 100$$

Where:
- $\alpha_{\text{type}}$ = intrinsic damage type severity (scratch: 0.35, dent: 0.65, crack: 0.75, lamp: 0.80, glass: 0.85, tire: 0.90)
- $\alpha_{\text{area}}$ = $\min(1.5, 2.5 \cdot \sqrt{\text{norm\_area}})$ — log-scaled bounding box footprint
- $\alpha_{\text{crit}}$ = component structural criticality (roof: 1.50, quarter\_panel: 1.35, hood: 1.20, ...)

Severity grade thresholds:
- DSI $\in [0, 38)$: **Minor**
- DSI $\in [38, 72)$: **Moderate**
- DSI $\in [72, 100]$: **Severe**

---

## 6. Output Artifacts

```
src/severity/
├── __init__.py               # Module exports
├── dataset.py                # SeverityDataset, stratified splits, transforms
├── model.py                  # DamageSeverityClassifier (ResNet18)
├── dsi.py                    # Damage Severity Index formulas
├── assessor.py               # VehicleSeverityAssessor (fuses visual + DSI)
├── train_severity.py         # Training script
└── evaluate_severity.py      # Evaluation benchmark

models/severity/phase7_severity_model/
├── best_severity_model.pt    # Best checkpoint (Val Macro-F1 = 0.6518)
├── results.csv               # Per-epoch metrics
├── results.png               # Training curves
└── experiment_record.json    # Full metadata

data/processed/
└── severity_metadata_stratified.csv  # Stratified 70/15/15 split index

evaluation/severity/
├── phase7_severity_report_test.json        # Test metrics
└── phase7_confusion_matrix_test.png        # Confusion matrix heatmap
```

---

## 7. Transition to Phase 8

- **Validation Gate:** Macro-F1 target was $\ge 0.60$. Achieved **0.6259** on test set. **PASS.**
- **Decision:** Severity module locked. The moderate-class weakness is mitigated by DSI complementary scoring and multi-feature fusion in the cost model.
- **Next Step:** **Phase 8: Repair-Cost Dataset Preparation** — engineer structured tabular features from Copart auction records, linking damage type + component + severity + vehicle metadata for repair cost regression.
