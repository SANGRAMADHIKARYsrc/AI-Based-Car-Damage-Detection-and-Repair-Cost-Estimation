# Project Session Log: 01chat — Phases 1 to 3 Complete

**Project:** AI-Based Car Damage Detection and Repair Cost Estimation Using Deep Learning  
**Date:** 2026-09-26  
**Status:** Phases 1–3 Complete, Ready for Guide Presentation & Phase 4  

---

## 1. Project Background & Research Question

### Core Research Question
> Can structured damage information extracted from vehicle images (component + damage type + severity + vehicle features) produce useful repair-cost estimates compared with directly predicting cost from an image?

### Pipeline Architecture
```
Car Image
    ↓
Image Preprocessing & Augmentation
    ↓
Damage Detection (YOLO) → Bounding Boxes & Damage Types
    ↓
Vehicle Component Identification (Bumper, Hood, Door, Fender)
    ↓
Severity Assessment (Minor, Moderate, Severe)
    ↓
Vehicle Metadata + Cost Features (Make, Year, Drivetrain, Damage Extent)
    ↓
Repair-Cost Estimation Model (Tabular Regressor / Interval Output)
    ↓
Final Inspection & Cost Report
```

---

## 2. Phase 1: Research & Literature Review (Summary)

- **10 Core Papers Analyzed:**
  - CarDD (Wang et al., 2023) — Object detection & instance segmentation benchmark.
  - VehiDE (Huynh et al., 2023/2024) — 13,945 images, 32k+ annotations.
  - HL-YOLO (2025) — YOLO11 + HetConv + LSKA on CarDD.
  - Smart-Claim (2026) — YOLO + LLM integration.
  - Tabular Insurance ML (Random Forest, XGBoost) on claim records.
- **5 Critical Gaps Identified:**
  1. *Primary Gap:* No end-to-end pipeline connecting structured visual damage to cost estimation.
  2. Absence of component-level annotations in raw damage datasets.
  3. No standardized severity ground truth in detection benchmarks.
  4. Disconnect between CV damage datasets and actuarial repair cost records.
  5. Scarcity of cross-dataset generalization evaluations.

---

## 3. Phase 2: Dataset Quality Audit & Leakage Analysis

### Audit Results Across 4 Datasets
1. **CarDD YOLO Dataset (4,000 images, 8,740 instances):**
   - Train: 2,816 images (6,211 boxes)
   - Val: 810 images (1,744 boxes)
   - Test: 374 images (785 boxes)
   - Class imbalance: `scratch` 41.1% (2,560) vs `tire flat` 3.6% (225) — $11.3\times$ ratio.
   - BBox stats: all coordinates valid in $[0, 1]$; median area 8.0% of frame.
2. **Severity Classification (`data3a`, 1,631 images):**
   - Minor: 534 | Moderate: 538 | Severe: 559. Perfectly balanced.
3. **VehiDE VIA Polygons (13,945 images, 36,081 polygons):**
   - 7 classes in Vietnamese: `tray_son` (scratch), `mop_lom` (dent), `rach` (tear/crack), `be_den` (broken lamp), `vo_kinh` (shattered glass), `thung` (puncture), `mat_bo_phan` (missing part).
4. **Copart Auction Dataset (26,144 unique lots):**
   - 19,235 positive repair cost records.
   - Median repair cost: $10,927.50 | Mean: $13,984.60.
5. **Data Leakage & Integrity Check:**
   - MD5 hash deduplication across all splits = **0 overlapping images (0.0% leakage)**.
   - Fixed `data/Datasets/data.yaml` from Kaggle paths to local paths.

---

## 4. Phase 3: Data Preparation & Preprocessing

### Executed Implementations
1. **Class Weights Formulation (`data/processed/class_weights.yaml`):**
   - Calculated smoothed square-root weights to prevent minority class omission:
     - `dent`: 0.5882
     - `scratch`: 0.4940
     - `crack`: 0.9797
     - `glass shatter`: 1.1469
     - `lamp broken`: 1.1247
     - `tire flat`: 1.6665
2. **COCO Format Converter (`src/utils/data_preparation.py`):**
   - Converted VehiDE VIA polygons to COCO standard:
     - `data/annotations/instances_vehide_train.json` (11,621 images, 30,046 annotations)
     - `data/annotations/instances_vehide_val.json` (2,324 images, 6,035 annotations)
3. **Cross-Dataset Semantic Mapping (`data/class_mapping.yaml`):**
   - Maps between CarDD 6 classes, VehiDE 7 classes, severity tags, and vehicle components.
4. **Severity Dataset Metadata Indexing (`data/processed/severity_metadata.csv`):**
   - Indexed all 1,631 images with relative paths, splits, and numerical labels (0: minor, 1: moderate, 2: severe).
5. **Clean Tabular Cost Dataset (`data/processed/copart_repair_cost_clean.csv`):**
   - 19,178 clean lots ($100 \le \text{cost} \le \$150,000$, valid vehicle age, log-transformed target).
   - Stratified split: Train 13,424 (70%), Val 2,876 (15%), Test 2,878 (15%).
6. **Augmentation Suite (`src/utils/augmentations.py`):**
   - Mosaic (1.0), MixUp (0.15), Copy-Paste (0.3), HSV jitter, horizontal flips.

---

## 5. Guide Teacher Presentation Plan

### Core Points
1. **Why not black-box image-to-cost?** Unexplainable; car make and repair labor rates dominate repair costs, not pixels alone. Structured pipeline bridges visual extraction with actuarial regression.
2. **Class imbalance solution:** Loss weighting tensor + targeted copy-paste augmentation.
3. **Zero leakage guarantee:** Cryptographic MD5 hash audit confirmed zero cross-split overlap.

### Scripts to Demonstrate
```bash
# 1. Run full dataset audit
python src/utils/analyze_datasets.py

# 2. Run data preparation and conversion
python src/utils/data_preparation.py

# 3. Interactive notebook
jupyter notebook notebooks/01_dataset_eda.ipynb
```

---

## 6. Next Steps (Phases 4–14)
- **Phase 4:** Baseline Damage Detector (Train YOLO on CarDD, evaluate mAP@50 and mAP@50-95).
- **Phase 5:** Damage Classification head enhancements.
- **Phase 6:** Vehicle Component Identification.
- **Phase 7:** Severity Assessment head.
- **Phase 8 & 9:** Repair Cost estimation models.
- **Phase 10:** Key research experiment (Structured vs Direct).
- **Phase 14:** Web Application (FastAPI backend + React frontend).
