# Phase 12 — Multi-Stage Error Analysis & Failure Mode Taxonomy

**Project:** AI-Based Car Damage Detection and Repair Cost Estimation Using Deep Learning  
**Date:** 2026-10-05  
**Status:** Complete  

---

## 1. Executive Summary & Diagnostic Scope

Phase 12 conducts an end-to-end diagnostic audit of error propagation and edge-case failure modes across all five stages of the pipeline:
1. **Stage 1: Object Detection (YOLOv8n)** — Missed small-scale damages and specular sun reflections.
2. **Stage 2: Damage Classification (EfficientNet-B0)** — Continuum confusion between superficial abrasion (`scratch`) and plastic deformation (`dent`).
3. **Stage 3: Component Localization (Ontology Prior)** — Seam boundary overlap between adjacent panels (`fender` vs `front_bumper`).
4. **Stage 4: Severity Assessment (ResNet18 / DSI)** — Borderline subjectivity between moderate repair and total replacement.
5. **Stage 5: Repair Cost Regressor (HistGradientBoosting)** — Residual outliers ($p95 = \$14,486.13$) driven by concealed mechanical/unibody damage.

---

## 2. Multi-Stage Failure Mode Taxonomy

### Stage 1: YOLO Object Detection
- **Small-Scale Receptive Field Attenuation:** Detection recall drops to **$28.4\%$** on damage footprints smaller than $32 \times 32$ pixels. Fine hairline scratches lack sufficient convolutional feature activation in deep P4/P5 layers.
  - *Engineering Mitigation:* High-resolution tiled inference ($1024 \times 1024$) via Slicing Aided Hyper Inference (SAHI).
- **Specular Highlight False Alarms:** $14.2\%$ of false positives stem from curved chrome bumpers and wet clearcoat reflecting bright sunlight streaks.
  - *Mitigation:* Photometric HSV jitter and polarized lens augmentations.

### Stage 2: Fine-Grained Damage Classification
- **Scratch vs Dent Boundary Continuum ($4.1\%$ confusion):** In physical collisions, sheet metal deformation is frequently accompanied by paint clearcoat gouges, creating an ambiguous overlap between `dent` and `scratch`.
  - *Mitigation:* Multi-label sigmoid head rather than mutually exclusive softmax.

### Stage 3: Automotive Component Localization
- **Seam Boundary Ambiguities ($12.3\%$ shift):** The wheel arch transition where the plastic bumper wrap-around clips into the steel fender panel is difficult for 2D bounding boxes to partition unambiguously.
  - *Mitigation:* Future integration of pixel-level semantic body segmentation.

### Stage 4: Damage Severity Assessment
- **Subjective Moderate vs Severe Threshold ($32.1\%$ confusion):** Human annotators disagree on borderline insurance classifications where a panel can theoretically either be repaired by an experienced technician or replaced with new OEM metal.
  - *Mitigation:* Cumulative link ordinal regression loss.

### Stage 5: Actuarial Cost Regressor
- **Concealed Mechanical & Unibody Frame Damage:** Underpredictions ($N = 129$ severe outlier cases) occur when exterior cosmetic damage masks bent suspension control arms, cracked subframes, or blown airbag steering modules.
- **Older Salvage Threshold Overpredictions ($N = 15$ cases):** On vehicles aged $> 15$ years, physical OEM replacement part quotes exceed actual fair-market salvage vehicle valuation.

---

## 3. Residual Diagnostics on Held-Out Test Set ($N = 2,878$)

- **Mean Absolute Error:** **\$4,701.78** | **Median AE:** **\$3,048.67**
- **95th Percentile Error Ceiling:** **\$14,486.13**
- **Age-Dependent Dispersion:** Vehicles aged $0 - 5$ years exhibit the highest variance in repair cost estimates due to expensive advanced driver assistance (ADAS) radar/lidar sensors housed behind bumpers.

---

## 4. Generated Artifacts

- Diagnostic Engine: [`src/experiments/error_analysis.py`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/src/experiments/error_analysis.py)
- Quantitative Taxonomy: [`evaluation/error_analysis/phase12_error_taxonomy.json`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/evaluation/error_analysis/phase12_error_taxonomy.json)
- Figures:
  - `evaluation/error_analysis/error_distribution_by_stage.png`
  - `evaluation/error_analysis/cost_error_by_vehicle_age.png`
  - `evaluation/error_analysis/outlier_residuals_by_tier.png`

---

## 5. Transition to Phase 13

Phase 12 complete.
Next is **Phase 13: Complete End-to-End Pipeline Integration** (unifying YOLO, EfficientNet, Component Identifier, DSI, and Cost Estimator into a single production Python pipeline with JSON and PDF inspection reporting).
