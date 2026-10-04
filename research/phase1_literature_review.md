# Phase 1 — Research and Literature Review

**Project:** AI-Based Car Damage Detection and Repair Cost Estimation Using Deep Learning
**Date:** 2026-09-26
**Status:** Complete

---

## 1. Research Scope

Literature was surveyed across these areas:
- Vehicle damage detection (object detection)
- Car damage classification (damage type)
- Vehicle damage severity assessment
- Automotive image segmentation (instance/semantic)
- Vehicle repair-cost estimation
- Insurance damage assessment automation
- Computer vision for vehicle inspection

---

## 2. Literature Review Table

| # | Paper | Year | Dataset | Model | Task | Key Metrics | Limitations | Source |
|---|-------|------|---------|-------|------|-------------|-------------|--------|
| 1 | **CarDD: A New Dataset for Vision-Based Car Damage Detection** — Wang X., Li W., Wu Z. | 2023 | CarDD (4,000 images, 9,000+ instances, 6 classes) | Faster R-CNN, Mask R-CNN, SOD baselines | Object detection, instance segmentation, salient object detection | Benchmarks for detection and segmentation on 6 damage types | No component-level annotation (bumper/door/etc.); no severity labels; no cost information | IEEE Trans. Intelligent Transportation Systems, Vol. 24(7), pp. 7202–7214; DOI: 10.1109/TITS.2023.3258480 |
| 2 | **VehiDE Dataset: New Dataset for Automatic Vehicle Damage Detection in Car Insurance** — Huynh N.T. et al. | 2023/2024 | VehiDE (13,945 images, 32,000+ annotations, 8 damage categories) | Mask R-CNN, Cascade Mask R-CNN | Classification, object detection, instance segmentation, SOD | Benchmarks across multiple architectures | No component identification; no severity or cost labels; larger but similar scope to CarDD | Taylor & Francis (published online) |
| 3 | **HL-YOLO: Improving Vehicle Damage Detection with Heterogeneous Convolutions and Large-Kernel Attention** | 2025 | Vehicle damage dataset (CarDD-based) | HL-YOLO (YOLO11 + HetConv + LSKA) | Object detection | Precision +2.5%, Recall +5.8%, mAP50 +3.9%, mAP50-95 +3.1% over YOLO11 baseline | Increased inference latency; detection only — no severity or cost estimation; single dataset evaluation | World Electric Vehicle Journal (MDPI) |
| 4 | **Smart Car Damage Assessment Using Enhanced YOLO Algorithm and Image Processing Techniques** — Ramazhan M.R.S., Bustamam A., Anwar R. | 2025 | CarDD | YOLOv9-CS (YOLOv9 + CBAM + SIoU loss) | Detection + severity index | ~1.75% improvement over baseline YOLO on CarDD; proposed Damage Severity Index (DSI) formula | DSI is heuristic-based (area ratio + confidence + count), not learned from ground truth severity; no component mapping; no cost estimation | MDPI journal |
| 5 | **Automated Vehicle Damage Classification Using Deep Learning Approaches** — (multiple authors, survey-type) | 2023 | Various custom datasets | ResNet50, EfficientNet-B3, VGG16, MobileNet (transfer learning) | Damage type classification (scratch, dent, crack, etc.) | Reported accuracies 90–95% on controlled datasets | Small custom datasets; no standardized benchmark; classification only — no localization; no severity or cost | PubMed Central / NIH |
| 6 | **Instance Segmentation for Car Damage Detection with Mask-RCNN** — (various implementations) | 2022–2023 | Custom annotated datasets, CarDD | Mask R-CNN (Detectron2) | Instance segmentation of damage regions | High pixel-level accuracy for damage localization | Computationally expensive; no real-time capability; no component identification separate from damage; no cost link | River Publishers; Zenodo |
| 7 | **Comparative Analysis of SipMask vs YOLACT for Vehicle Part Localization** | 2023 | Custom automotive part dataset | SipMask, YOLACT | Real-time instance segmentation of vehicle components | YOLACT higher accuracy; SipMask better in noisy conditions | Limited to component identification — not damage detection; small dataset | CSUSB ScholarWorks |
| 8 | **Smart-Claim: AI-Based Vehicle Damage Detection with LLM Integration** | 2026 | CarDD-based + custom data | YOLO + NVIDIA Nemotron-3 (Llama 3.1) | Detection + LLM-based repair insight generation | Reduced inspection time; enhanced report quality | Cost estimation via LLM — not grounded in structured pricing data; reproducibility concerns with LLM outputs; no formal cost evaluation metrics (MAE/RMSE) | IRE Journals |
| 9 | **Deep Learning for Vehicle Damage Assessment and Repair Cost Estimation** — (Mask R-CNN + YOLOv7 pipeline) | 2023 | Custom insurance dataset | Mask R-CNN + YOLOv7 | Detection + severity + cost lookup | High reliability for damage-type and side recognition | Cost estimation uses rule-based lookup, not a learned model; dataset not public; limited generalization evidence | edu.krd |
| 10 | **Insurance Claim Prediction Using ML** — (Random Forest, XGBoost, SVM for claim cost) | 2022–2023 | Tabular insurance claim data | Random Forest, XGBoost, SVM | Claim cost classification / regression | Improved classification accuracy for high-cost claims | No image input — purely tabular; not specific to visible damage; conflates claim cost with repair cost | ResearchGate; MDPI |

---

## 3. Research Gaps Identified

### Gap 1: No End-to-End Pipeline Connecting Structured Visual Damage → Cost Estimation
Most detection papers stop at bounding boxes + damage type. Most cost papers use tabular insurance data without image-derived features. **No published work rigorously evaluates whether structured damage information (component + type + severity) extracted from images improves cost estimation compared to direct image-to-cost regression.**

### Gap 2: Absence of Component-Level Annotations in Damage Datasets
CarDD and VehiDE annotate *damage types* (dent, scratch, crack, etc.) but do NOT annotate *which vehicle component* is damaged (bumper, door, hood, fender). Component identification is essential for cost estimation — repairing a bumper dent costs differently from a door dent.

### Gap 3: No Standardized Severity Ground Truth
Severity labels (minor/moderate/severe) do not exist in CarDD or VehiDE. The YOLOv9-CS paper proposes a heuristic DSI formula but it is not learned from ground-truth severity annotations. No dataset provides validated severity labels.

### Gap 4: Disconnect Between Damage Detection Datasets and Cost Data
No public dataset links detected damage to actual repair costs. Detection datasets have images + damage annotations. Cost datasets have tabular repair records. Nobody has systematically bridged these two.

### Gap 5: Limited Cross-Dataset Generalization Studies
Most papers evaluate on a single dataset. Very few studies train on one damage dataset and test on another to measure domain shift, class mapping issues, and real-world robustness.

---

## 4. Selected Primary Research Gap

### **Gap 1: No End-to-End Pipeline Comparing Structured Damage Information vs. Direct Image-to-Cost Estimation**

This is the PRIMARY research gap this project will address.

---

## 5. Why This Gap Matters

**Academic significance:**
- Directly tests whether intermediate structured representations (component + damage type + severity) provide value over black-box image-to-cost regression
- Provides a reproducible experimental framework for comparing modular vs. end-to-end approaches
- Addresses the interpretability vs. accuracy tradeoff — structured pipelines are explainable; direct regression is not

**Practical significance:**
- Insurance companies need explainable estimates — a structured pipeline can justify each cost component
- A structured approach enables integration with existing repair databases and pricing catalogs
- If the structured pipeline performs comparably or better, it validates the investment in component + severity + type annotations

**Research question (restated):**
> Can structured damage information extracted from vehicle images (component + damage type + severity + vehicle features) produce repair-cost estimates that are as good as or better than directly predicting cost from the raw image?

**Hypothesis to test (not pre-assumed):**
- Model A: Image → CNN → Cost prediction
- Model B: Image → Detection → Component → Type → Severity → Vehicle features → Cost prediction
- The experiment will determine which approach is superior. We do NOT assume Model B wins.

---

## 6. Secondary Gaps Addressed

This project will also partially address:
- **Gap 2** — by investigating component identification (using separate datasets or documented annotation if needed)
- **Gap 3** — by defining and evaluating severity criteria
- **Gap 4** — by creating a structured cost dataset with documented sources

Gap 5 (cross-dataset generalization) will be addressed in Phase 11 if compatible datasets are available.

---

## 7. Limitations of This Review

1. Some papers found via web search may be from non-peer-reviewed venues (conference proceedings, preprints, journal-style open-access). Each citation should be verified for venue quality before inclusion in the final research paper.
2. Full-text access was not available for all papers — some details are from abstracts and secondary sources.
3. The field is evolving rapidly (YOLO versions increment frequently). More recent papers may exist by the time this project concludes.
4. Papers using LLMs for cost insight (Smart-Claim) are very recent and may not have undergone rigorous peer review.

---

## 8. Next Step

**→ Phase 2: Dataset Analysis**

Before training anything, we must:
1. Download and inspect every listed dataset
2. Count images, annotations, classes
3. Check for class imbalance, resolution distribution, corrupted/duplicate images
4. Create dataset documentation
5. Determine which datasets are usable given their licenses

---

## References (for verification)

1. Wang, X., Li, W., Wu, Z. (2023). CarDD: A New Dataset for Vision-Based Car Damage Detection. IEEE TITS, 24(7), 7202–7214. https://doi.org/10.1109/TITS.2023.3258480
2. Huynh, N.T. et al. (2023/2024). VehiDE Dataset. Taylor & Francis. https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection
3. HL-YOLO (2025). World Electric Vehicle Journal (MDPI). https://www.mdpi.com/
4. Ramazhan, M.R.S., Bustamam, A., Anwar, R. (2025). Smart Car Damage Assessment Using Enhanced YOLO Algorithm. MDPI.
5. CarDD project page: https://cardd-ustc.github.io/
6. CarDD YOLO annotations: https://www.kaggle.com/datasets/gabrielfcarvalho/cardd-with-yolo-annotations-images-labels
