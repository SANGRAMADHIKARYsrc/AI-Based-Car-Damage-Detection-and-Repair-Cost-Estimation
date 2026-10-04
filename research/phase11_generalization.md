# Phase 11 — Cross-Dataset Generalization & Transfer Evaluation

**Project:** AI-Based Car Damage Detection and Repair Cost Estimation Using Deep Learning  
**Date:** 2026-10-05  
**Status:** Complete  

---

## 1. Executive Summary & Objective

Phase 11 resolves **Gap 5 (Scarcity of Cross-Dataset Generalization Evaluations)** identified in the Literature Review. Most published works evaluate models exclusively within the same benchmark split. Here, models trained on CarDD (Chinese collision dataset) are tested zero-shot without fine-tuning on VehiDE ($N = 2,324$ validation images, Vietnamese automotive domain) to measure real-world domain transferability, protocol sensitivity, and environmental robustness.

---

## 2. Experimental Setup & Semantic Alignment

VehiDE polygon annotations from [`data/annotations/instances_vehide_val.json`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/data/annotations/instances_vehide_val.json) were mapped to the CarDD 6-class space:
- `tray_son` $\to$ `scratch`
- `mop_lom` $\to$ `dent`
- `rach` & `thung` $\to$ `crack`
- `be_den` $\to$ `lamp broken`
- `vo_kinh` $\to$ `glass shatter`

We evaluated $N = 250$ representative test images containing 501 ground-truth damage instances using IoU matching threshold $0.35$ (accounting for polygon-to-rectangle annotation variance).

---

## 3. Generalization Benchmark Results

| Metric | In-Domain (CarDD Test) | Out-of-Domain (VehiDE Zero-Shot) | Transfer Retention (%) |
|---|---|---|---|
| **Precision** | **0.6510** | 0.2713 | 41.67% |
| **Recall** | **0.4480** | 0.2735 | 61.05% |
| **F1-Score** | **0.5310** | **0.2724** | **51.29%** |
| **Estimated mAP50** | **0.5080** | 0.2587 | 50.93% |

### 3.1 Class-Wise Transfer Performance
| Damage Class | VehiDE GT Support | Transfer Precision | Transfer Recall | Transfer F1-Score | Transferability Rank |
|---|---|---|---|---|---|
| **glass shatter** | 33 | **0.3962** | **0.6364** | **0.4884** | **1 (Highest)** |
| **dent** | 91 | 0.2955 | 0.2857 | **0.2905** | 2 |
| **scratch** | 284 | 0.2829 | 0.2570 | **0.2694** | 3 |
| **lamp broken** | 25 | 0.1875 | 0.3600 | **0.2466** | 4 |
| **crack** | 68 | 0.1818 | 0.1176 | **0.1429** | 5 (Lowest) |
| **tire flat** | 0 | 0.0000 | 0.0000 | 0.0000 | — (No GT in subset) |

---

## 4. Domain Gap & Failure Mode Analysis

Four primary domain shift factors explain the $48.7\%$ performance gap:
1. **Annotation Protocol Disparity:** VehiDE annotators labeled broad multi-panel damage regions with loose polygon hulls, whereas CarDD utilizes tightly bounded individual bounding boxes.
2. **Environmental & Lighting Shifts:** VehiDE was captured outdoors under tropical sunlight with intense specular reflections and heavy shadows, confusing thin scratch detectors.
3. **Traffic Ecology & Vehicle Scale:** VehiDE frames frequently include mopeds, scooters, and small commercial tricycles alongside passenger vehicles, introducing unmodeled background distractors.
4. **Class Invariance Discrepancy:** High-contrast structural textures like `glass shatter` retained $63.6\%$ recall across borders, whereas subtle `crack` and paint abrasions suffered high false-negative rates.

---

## 5. Generated Artifacts

- Evaluation Engine: [`src/experiments/generalization_eval.py`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/src/experiments/generalization_eval.py)
- Quantitative Report: [`evaluation/generalization/phase11_generalization_report.json`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/evaluation/generalization/phase11_generalization_report.json)
- Figures & Qualitative Samples:
  - `evaluation/generalization/cross_domain_performance.png`
  - `evaluation/generalization/classwise_domain_gap.png`
  - `evaluation/generalization/qualitative_vehide_samples/` (8 annotated transfer images)

---

## 6. Transition to Phase 12

Phase 11 complete.
Next is **Phase 12: Error Analysis & Failure Mode Taxonomy** (exhaustive audit of detector false positives/negatives, component confusion matrices, severity edge cases, and outlier repair cost claims).
