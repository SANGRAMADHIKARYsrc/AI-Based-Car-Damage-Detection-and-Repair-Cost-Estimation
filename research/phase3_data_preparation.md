# Phase 3 — Data Preparation and Pipeline Setup

**Project:** AI-Based Car Damage Detection and Repair Cost Estimation Using Deep Learning  
**Date:** 2026-09-26  
**Status:** Complete  

---

## 1. Executive Summary

Phase 3 established the standardized data preprocessing, format conversion, and imbalance mitigation pipelines:
1. **Class Weights Formulation:** Evaluated inverse frequency, square-root smoothing, and Effective Number of Samples (ENS) to mitigate the $11.3:1$ imbalance between `scratch` and `tire flat`. Exported to `data/processed/class_weights.yaml`.
2. **Annotation Standardization (VIA → COCO JSON):** Converted 13,945 images and 36,081 polygon annotations from VehiDE VIA format into standardized COCO instance segmentation JSON format (`data/annotations/instances_vehide_train.json` and `instances_vehide_val.json`).
3. **Cross-Dataset Mapping:** Formalized bidirectional semantic class mapping between CarDD 6-class detection, VehiDE 7-class segmentation, and Copart damage categories in `data/class_mapping.yaml`.
4. **Severity Dataset Indexing:** Built `data/processed/severity_metadata.csv` indexing 1,631 images across minor, moderate, and severe classes for 3-class classification head training.
5. **Copart Repair Cost Dataset Cleaning:** Screened 26,144 auction records to produce 19,178 high-quality records with positive repair costs ($\$100 \le \text{cost} \le \$150,000$), log target transforms, and vehicle age feature engineering, split 70/15/15 into train, validation, and test splits (`data/processed/copart_repair_cost_clean.csv`).
6. **Augmentation Strategies:** Implemented domain-specific augmentation configurations in `src/utils/augmentations.py` (Mosaic 1.0, MixUp 0.15, Copy-Paste 0.3, horizontal flips only).

---

## 2. Class Imbalance Mitigation & Loss Weighting

### 2.1 Formulae Evaluated

- **Inverse Class Frequency (ICF):**
  $$W_{c}^{\text{ICF}} = \frac{N_{\text{total}}}{K \times N_c}$$

- **Smoothed Square-Root Weighting:**
  $$W_{c}^{\text{sqrt}} = \sqrt{\frac{\max_j(N_j)}{N_c}}, \quad \bar{W}_c = \frac{W_c}{\frac{1}{K}\sum_k W_k}$$

- **Effective Number of Samples (ENS):**
  $$E_n = \frac{1 - \beta^n}{1 - \beta}, \quad W_{c}^{\text{ENS}} \propto \frac{1}{E_{N_c}} \quad (\beta = 0.999)$$

### 2.2 Computed Weight Table
| Class Name | Train Instances | ICF Weight | Smoothed Sqrt Weight (Recommended) | ENS Weight |
|---|---|---|---|---|
| dent | 1,806 | 0.5732 | **0.5882** | 0.4938 |
| scratch | 2,560 | 0.4044 | **0.4940** | 0.4472 |
| crack | 651 | 1.5901 | **0.9797** | 0.8622 |
| glass shatter | 475 | 2.1793 | **1.1469** | 1.0911 |
| lamp broken | 494 | 2.0955 | **1.1247** | 1.0583 |
| tire flat | 225 | 4.6007 | **1.6665** | 2.0474 |

> **Design Choice:** Smoothed Sqrt Weighting is adopted for training loss weighting (`recommended_loss_weights_tensor`). Raw ICF ($4.60\times$) destabilizes early gradient descent for tiny bounding boxes, whereas smoothed weights prevent gradient explosion while penalizing missed detections on minority classes (`tire flat`, `lamp broken`, `glass shatter`).

---

## 3. VehiDE Conversion to COCO Format

VIA JSON polygon annotations converted to standard COCO JSON format:
- **Shoelace Polygon Area Computation:** Verified non-zero area for all polygon boundaries ($N \ge 3$ vertices).
- **Bounding Box Derivation:** $[x_{\min}, y_{\min}, w, h]$ derived for every polygon.
- **Artifacts Created:**
  - `data/annotations/instances_vehide_train.json`: 11,621 images, 30,046 annotations.
  - `data/annotations/instances_vehide_val.json`: 2,324 images, 6,035 annotations.

---

## 4. Tabular Repair Cost Dataset Preparation (Copart)

### 4.1 Filter and Pipeline Criteria
1. Range filter: $\$100 \le \text{repairCost} \le \$150,000$. Removes zero values, data entry artifacts, and extreme multi-million dollar commercial outliers.
2. Valid model year: $1990 \le \text{year} \le 2026$.
3. Feature engineering:
   - $\text{vehicle\_age} = 2026 - \text{year}$
   - $\text{log\_repair\_cost} = \ln(1 + \text{repairCost})$

### 4.2 Split Statistics
- **Total Clean Records:** 19,178
- **Train Split (70%):** 13,424 records
- **Validation Split (15%):** 2,876 records
- **Test Split (15%):** 2,878 records
- **File:** `data/processed/copart_repair_cost_clean.csv`

---

## 5. Transition to Phase 4

Phase 3 is complete. All training data formats, class weights, metadata splits, and augmentation policies are locked.
Next phase is **Phase 4: Baseline Damage Detector** (YOLO training script, baseline validation, metric tracking).
