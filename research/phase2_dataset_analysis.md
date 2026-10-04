# Phase 2 — Dataset Analysis and Quality Audit

**Project:** AI-Based Car Damage Detection and Repair Cost Estimation Using Deep Learning  
**Date:** 2026-09-26  
**Status:** Complete  

---

## 1. Executive Summary

Phase 2 evaluated and audited all four dataset sources in `data/Datasets`:
1. **CarDD YOLO Dataset:** 4,000 images, 8,740 bounding-box instances across 6 classes. Verified zero corruptions, zero invalid coordinates, and zero split leakage. Severe class imbalance quantified (11.3:1 between scratch and tire flat).
2. **Car Damage Severity Dataset (`data3a`):** 1,631 images across 3 severity levels (Minor, Moderate, Severe). Balanced distribution confirmed. Zero split leakage.
3. **VehiDE Dataset (`0Train_via_annos.json`, `0Val_via_annos.json`):** 13,945 images, 36,081 polygon annotations across 7 classes in Vietnamese. Cross-dataset mapping established.
4. **Copart Auction Dataset:** 26,144 unique vehicle listing records, 19,235 records with positive ground-truth repair cost estimates (mean $13,984.60, median $10,927.50).

---

## 2. CarDD YOLO Dataset Audit

### 2.1 Split Statistics
| Split | Images | Label Files | Missing Labels | Corrupted Images | Total Instances | Mean Inst/Img |
|---|---|---|---|---|---|---|
| Train | 2,816 | 2,816 | 0 | 0 | 6,211 | 2.21 |
| Validation | 810 | 810 | 0 | 0 | 1,744 | 2.15 |
| Test | 374 | 374 | 0 | 0 | 785 | 2.10 |
| **Total** | **4,000** | **4,000** | **0** | **0** | **8,740** | **2.19** |

### 2.2 Class Distribution Across Splits
| Class ID | Class Name | Train Count | Val Count | Test Count | Total Count | Overall Share | Ratio to Min |
|---|---|---|---|---|---|---|---|
| 0 | dent | 1,806 | 501 | 236 | 2,543 | 29.1% | 8.0x |
| 1 | scratch | 2,560 | 728 | 307 | 3,595 | 41.1% | 11.3x |
| 2 | crack | 651 | 177 | 70 | 898 | 10.3% | 2.8x |
| 3 | glass shatter | 475 | 135 | 71 | 681 | 7.8% | 2.1x |
| 4 | lamp broken | 494 | 141 | 69 | 704 | 8.1% | 2.2x |
| 5 | tire flat | 225 | 62 | 32 | 319 | 3.6% | 1.0x (base) |

### 2.3 Bounding Box & Geometric Characteristics
- **Invalid Bounding Boxes:** 0 (all $cx, cy, w, h \in [0.0, 1.0]$)
- **Mean Normalized BBox Area:** $0.171$ (~17.1% of frame area)
- **Median Normalized BBox Area:** $0.080$ (~8.0% of frame area)
- **Min BBox Area:** $0.000175$ (fine surface scratches and minor cracks)
- **Max BBox Area:** $1.000000$ (full-frame severe impact damage)
- **Dominant Aspect Ratio / Resolution:** $1000 \times 667$ (3:2 landscape)

---

## 3. Severity Classification Dataset (`data3a`)

### 3.1 Distribution
| Split | 01-Minor | 02-Moderate | 03-Severe | Total | Corrupted | Common Resolution |
|---|---|---|---|---|---|---|
| Training | 452 | 463 | 468 | 1,383 | 0 | $259 \times 194$ |
| Validation | 82 | 75 | 91 | 248 | 0 | $259 \times 194$ |
| **Total** | **534** | **538** | **559** | **1,631** | **0** | **$259 \times 194$** |

### 3.2 Evaluation Notes
- **Balance:** Near 1:1:1 class ratio ($32.7\%$ minor, $33.0\%$ moderate, $34.3\%$ severe).
- **Resolution:** Thumbnails ($259 \times 194$). Image upsampling or backbone receptive-field consideration required for multi-task heads.

---

## 4. VehiDE Polygon Dataset (`0Train_via_annos.json`, `0Val_via_annos.json`)

Identified as VehiDE (Huynh et al., 2023/2024), providing pixel-accurate polygon segmentation for cross-dataset evaluation.

### 4.1 Class Translation and Instance Counts
| Vietnamese Name | Translated Damage Type | Train Polygons | Val Polygons | Total Polygons | Share |
|---|---|---|---|---|---|
| `tray_son` | scratch | 12,267 | 2,380 | 14,647 | 40.6% |
| `mop_lom` | dent | 4,709 | 972 | 5,681 | 15.7% |
| `rach` | tear / crack | 4,546 | 963 | 5,509 | 15.3% |
| `mat_bo_phan` | missing part | 2,370 | 448 | 2,818 | 7.8% |
| `be_den` | broken lamp | 2,324 | 458 | 2,782 | 7.7% |
| `thung` | puncture / hole | 2,006 | 417 | 2,423 | 6.7% |
| `vo_kinh` | glass shatter | 1,824 | 397 | 2,221 | 6.2% |
| **Total** | | **30,046** | **6,035** | **36,081** | **100.0%** |

### 4.2 Image Counts
- **Training:** 11,621 images (in `image/image/`)
- **Validation:** 2,324 images (in `validation/validation/`)
- **Total:** 13,945 images

---

## 5. Copart Repair Cost Dataset

Evaluated 30 daily auction snapshots ($26,144$ unique auction lot records).

### 5.1 Financial Feature Properties
- **Records with Positive Repair Cost:** 19,235 ($73.6\%$)
- **Mean Repair Cost:** $\$13,984.60$
- **Median Repair Cost:** $\$10,927.50$
- **Standard Deviation:** $\$21,954.53$
- **Interquartile Range (IQR):** $\$6,709.11$ – $\$17,703.85$
- **Minimum Positive Cost:** $\$100.00$
- **Maximum Cost:** $\$2,576,733.00$ (exotic/commercial outliers requiring log transform or capping)

### 5.2 Top Damage Categories in Auction Data
1. `FRONT END`: 12,566 lots ($48.1\%$)
2. `REAR END`: 3,903 lots ($14.9\%$)
3. `SIDE`: 3,438 lots ($13.2\%$)
4. `MINOR DENT/SCRATCHES`: 1,534 lots ($5.9\%$)
5. `MECHANICAL`: 1,047 lots ($4.0\%$)

### 5.3 Top Vehicle Makes
Toyota ($3,206$), Ford ($2,922$), Chevrolet ($2,673$), Honda ($2,574$), Nissan ($1,923$). Median vehicle model year is 2017.

---

## 6. Data Integrity and Leakage Analysis

MD5 hashing executed across all image files and splits.

| Integrity Check | Result | Implication |
|---|---|---|
| CarDD Train vs Val Overlap | **0 images (0.0%)** | Zero leakage |
| CarDD Train vs Test Overlap | **0 images (0.0%)** | Zero leakage |
| CarDD Val vs Test Overlap | **0 images (0.0%)** | Zero leakage |
| CarDD Internal Duplicates | **0 images (0.0%)** | Clean dataset |
| data3a Train vs Val Overlap | **0 images (0.0%)** | Zero leakage |
| CarDD vs data3a Cross Overlap | **0 images (0.0%)** | Datasets are independent |
| Corrupted or Unreadable Images | **0 files** | All valid JPEG/PNG files |

---

## 7. Action Items Completed and Mitigation Plan

1. **Config Path Fix:** Fixed `data/Datasets/data.yaml` from Kaggle paths to local paths (`train/images`, `val/images`, `test/images`).
2. **Imbalance Mitigation Strategy:** 
   - Apply focal loss / class weighting during YOLO training for `tire flat` ($3.6\%$) and `glass shatter` ($7.8\%$).
   - Use mosaic augmentation and copy-paste augmentation targeted at minority classes in Phase 3.
3. **Class Alignment:** Built formal class translation dictionary for VehiDE polygons to align with CarDD classes.
4. **Repair Cost Preprocessing:** Established log transform ($\log(1 + \text{repairCost})$) and outlier filtering ($<\$100$ and $>\$75,000$) for Phase 8 tabular model.

---

## 8. Transition to Phase 3

Phase 2 is fully complete. Next phase is **Phase 3: Data Preparation**, which will implement:
- Annotation conversion pipelines (VIA polygon to COCO format).
- Class weight computation.
- Preprocessing and augmentation pipelines.
- Final train/val/test artifact generation.
