# Phase 4 — Baseline Damage Detector

**Project:** AI-Based Car Damage Detection and Repair Cost Estimation Using Deep Learning  
**Date:** 2026-09-29  
**Status:** Complete  

---

## 1. Objective

Train and evaluate a baseline YOLOv8n object detection model on the CarDD 6-class damage dataset. Establishes initial performance floor for the detection stage of the pipeline.

### 1.1 Target Metrics (Baseline Expectations)

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| mAP50 | 0.45–0.65 | First pass, no hyperparameter tuning |
| mAP50-95 | 0.25–0.40 | Strict IoU averaging |
| Precision | 0.50–0.70 | Across all 6 classes |
| Recall | 0.40–0.60 | Minority classes will underperform |
| Inference | <15ms | RTX 2050, YOLOv8n at 640px |

> **Note:** These are *baseline* expectations. Phase 5+ will add classification heads and hyperparameter optimization.

---

## 2. Model Selection Rationale

**YOLOv8n (nano)** selected as baseline:
- 3.2M parameters — fast iteration cycle
- COCO pretrained — strong transfer learning initialization
- Native support for class-weighted loss, mosaic, copy-paste
- Fits in 4GB VRAM with AMP (mixed precision)

Comparison candidates for Phase 11 (Generalization):
| Model | Params | FLOPs | Use Case |
|-------|--------|-------|----------|
| YOLOv8n | 3.2M | 8.7G | Baseline (this phase) |
| YOLOv8s | 11.2M | 28.6G | Accuracy improvement candidate |
| YOLOv8m | 25.9M | 78.9G | If VRAM permits (unlikely on 4GB) |

---

## 3. Training Configuration

### 3.1 Hardware
- **GPU:** NVIDIA GeForce RTX 2050 (4GB VRAM, compute 8.6)
- **Driver:** 592.00
- **Mixed Precision:** Enabled (AMP) — required for 4GB VRAM

### 3.2 Dataset
- **Source:** CarDD (Car Damage Detection Dataset)
- **Split:** Train 2,816 / Val 810 / Test 374
- **Classes:** 6 — dent, scratch, crack, glass shatter, lamp broken, tire flat
- **Format:** YOLO (normalized xywh per line in .txt labels)

### 3.3 Hyperparameters
```yaml
epochs: 100
batch_size: 16
image_size: 640
patience: 15              # early stopping
optimizer: auto            # AdamW + cosine LR
lr0: 0.01
lrf: 0.01                 # final_lr = lr0 * lrf
warmup_epochs: 3
close_mosaic: 10           # disable mosaic last 10 epochs
amp: true
seed: 42
```

### 3.4 Augmentation (from Phase 3)
```yaml
hsv_h: 0.015    hsv_s: 0.7     hsv_v: 0.4
degrees: 10     translate: 0.1  scale: 0.5
shear: 2.0      perspective: 0.0005
fliplr: 0.5     flipud: 0.0
mosaic: 1.0     mixup: 0.15    copy_paste: 0.3
erasing: 0.2
```

### 3.5 Class Weights (Smoothed Sqrt, Phase 3)
| Class | Weight | Train Instances |
|-------|--------|-----------------|
| dent | 0.5882 | 1,806 |
| scratch | 0.4940 | 2,560 |
| crack | 0.9797 | 651 |
| glass shatter | 1.1469 | 475 |
| lamp broken | 1.1247 | 494 |
| tire flat | 1.6665 | 225 |

---

## 4. Scripts Created

| File | Purpose |
|------|---------|
| `src/detection/train_baseline.py` | Training with config loading, validation, metric logging |
| `src/detection/evaluate_baseline.py` | Per-class evaluation, confusion matrix, speed benchmark |
| `src/detection/predict_samples.py` | Annotated prediction visualizations |
| `experiments/phase4_baseline_detector.yaml` | Experiment configuration (all hyperparameters) |

---

## 5. Execution Steps

### Step 1: Install Dependencies
```bash
pip install ultralytics>=8.0.0
```

### Step 2: Train Baseline
```bash
python src/detection/train_baseline.py --config experiments/phase4_baseline_detector.yaml
```

### Step 3: Evaluate on Validation Set
```bash
python src/detection/evaluate_baseline.py --split val
```

### Step 4: Evaluate on Test Set
```bash
python src/detection/evaluate_baseline.py --split test
```

### Step 5: Generate Sample Predictions
```bash
python src/detection/predict_samples.py --n 20
```

---

## 6. Output Artifacts

After training completes:
```
models/detection/phase4_yolov8n_baseline/
├── weights/
│   ├── best.pt                   # Best mAP50 checkpoint
│   └── last.pt                   # Final epoch checkpoint
├── experiment_record.json        # Full reproducibility metadata
├── results.csv                   # Per-epoch training metrics
├── confusion_matrix.png
├── results.png                   # Training curves
├── P_curve.png                   # Precision vs confidence
├── R_curve.png                   # Recall vs confidence
├── F1_curve.png
└── PR_curve.png                  # Precision-Recall curve

evaluation/detection/
├── phase4_val_report.json        # Validation metrics + per-class
├── phase4_test_report.json       # Test metrics + per-class
└── phase4_predictions/           # Annotated sample images
```

---

## 7. Results

### 7.1 Overall Metrics
| Metric | Validation Set (810 images) | Test Set (374 images) | Target Expectation |
|--------|-----------------------------|-----------------------|--------------------|
| **Precision** | **0.7147** | **0.7012** | 0.50–0.70 |
| **Recall** | **0.6890** | **0.6911** | 0.40–0.60 |
| **F1** | **0.7016** | **0.6961** | >0.55 |
| **mAP50** | **0.7168** | **0.6494** | 0.45–0.65 |
| **mAP50-95** | **0.5326** | **0.4925** | 0.25–0.40 |
| **Inference Time** | **2.6 ms** | **5.7 ms** | <15 ms |
| **Throughput** | **~384 FPS** | **~76.4 FPS (batch+post)** | Real-time (>30 FPS) |

### 7.2 Per-Class Breakdown (Test Set)
| Class | Instances | Precision | Recall | F1 | AP50 | AP50-95 | Notes |
|-------|-----------|-----------|--------|----|------|---------|-------|
| **dent** | 236 | 0.6441 | 0.6102 | 0.6267 | 0.5080 | 0.2920 | Moderate deformation, glare sensitivity |
| **scratch** | 307 | 0.5697 | 0.5537 | 0.5616 | 0.4869 | 0.2556 | Thin linear marks, variable contrast |
| **crack** | 70 | 0.4762 | 0.3248 | 0.3862 | 0.2736 | 0.1362 | Hardest class; subtle fracture lines |
| **glass shatter** | 71 | 0.8861 | 0.9859 | 0.9333 | 0.9737 | 0.8492 | Distinct spiderweb texture; near-perfect |
| **lamp broken** | 69 | 0.7894 | 0.7971 | 0.7932 | 0.7820 | 0.6335 | Geometric component boundary helps |
| **tire flat** | 32 | 0.8418 | 0.8750 | 0.8581 | 0.8719 | 0.7882 | Strong performance despite minority count |

> **Key Finding:** Smoothed square-root class weights (`tire flat: 1.6665`, `glass shatter: 1.1469`) prevented minority starvation: `tire flat` scored **0.8719 AP50** and `glass shatter` scored **0.9737 AP50** on the test set.

---

## 8. Known Risks and Mitigations

| Risk | Mitigation | Outcome |
|------|-----------|---------|
| 4GB VRAM OOM | AMP enabled, batch=16 for nano | Trained stable at 1.88G - 2.03G VRAM |
| 8GB Host RAM Exhaustion | Set `workers=2`, `plots=False` during epoch validation | Training completed without memory error |
| Minority class starvation | Smoothed sqrt weights + copy-paste augmentation | `tire flat` test AP50 reached 0.872 |
| Crack underperformance | Documented for Phase 5 & 12 | Crack AP50 = 0.274; target for specialized head |

---

## 9. Transition to Phase 5

- **Validation Gate:** mAP50 test target was $\ge 0.50$. Achieved **0.6494** ($\mathbf{+14.94\%}$ above gate).
- **Decision:** **PASS.** Baseline detector is locked and ready.
- **Model Checkpoints:** Saved to `models/detection/phase4_yolov8n_baseline/weights/best.pt`.
- **Next Step:** **Phase 5: Damage Classification** — extract detected bounding box crops and train specialized classifier heads to refine fine-grained damage categorization.
