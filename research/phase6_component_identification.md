# Phase 6 — Vehicle Component Identification

**Project:** AI-Based Car Damage Detection and Repair Cost Estimation Using Deep Learning  
**Date:** 2026-10-04  
**Status:** Complete  

---

## 1. Objective & Research Context

### 1.1 Addressing Research Gap 2
In the Phase 1 literature review, **Gap 2** identified a critical limitation in current computer vision car damage benchmarks:
> **Gap 2: Absence of Component-Level Annotations in Damage Datasets**  
> Benchmarks like CarDD and VehiDE annotate *damage types* (dent, scratch, crack, etc.) but do NOT annotate *which vehicle component* is damaged (bumper, door, hood, fender, etc.). Component identification is indispensable for repair cost estimation — repairing or replacing a structural quarter panel or hood costs significantly more than buffing a door scratch.

### 1.2 Role in the Capstone Pipeline
Phase 6 establishes the architectural bridge between low-level visual damage localization and high-level actuarial cost estimation:
$$\text{Car Image} \xrightarrow{\text{Phase 4}} \text{BBoxes} \xrightarrow{\text{Phase 5}} \text{Damage Types} \xrightarrow{\mathbf{Phase\ 6}} \mathbf{Vehicle\ Components} \xrightarrow{\text{Phase 7}} \text{Severity} \xrightarrow{\text{Phase 8/9}} \text{Repair Cost}$$

---

## 2. Automotive Component Ontology

We defined a standardized 10-class automotive repair component taxonomy aligned with collision estimating guides (Mitchell / CCC ONE / Copart):

| Component ID | Component Name | Description | Default Copart Sector | Labor/Part Cost Multiplier |
|---|---|---|---|---|
| 0 | `front_bumper` | Front plastic/composite bumper cover & reinforcement | `FRONT END` | $1.15\times$ |
| 1 | `rear_bumper` | Rear bumper cover, absorber & diffuser | `REAR END` | $1.10\times$ |
| 2 | `hood` | Front engine hood / bonnet assembly | `FRONT END` | $1.35\times$ |
| 3 | `door` | Front/rear side doors and outer door skins | `SIDE` | $1.25\times$ |
| 4 | `fender` | Front side body quarter fenders | `SIDE` | $0.95\times$ |
| 5 | `quarter_panel` | Rear structural welded body side panel | `REAR END` | $1.45\times$ |
| 6 | `roof` | Vehicle roof sheet metal, sunroof frame, pillars | `ROLLOVER` | $1.80\times$ |
| 7 | `windshield_glass` | Front windshield, rear glass, side windows | `MINOR DENT/SCRATCHES` | $0.75\times$ |
| 8 | `lamp` | Headlamp assembly, taillight, indicator | `FRONT END` | $0.85\times$ |
| 9 | `wheel_tire` | Wheel rim, tire rubber, hub assembly | `SIDE` | $0.65\times$ |

---

## 3. Methodology: Hybrid Semantic & Spatial-Affinity Fusion

To resolve component identity without manual dense re-annotation of 4,000 images, we implemented a hybrid three-tier reasoning architecture:

### 3.1 Tier 1: Deterministic Semantic Shortcuts
Damage categories that intrinsically bound specific vehicle components are resolved deterministically with high confidence ($0.95–0.99$):
- $\text{Class} = \text{"tire flat"} \implies \text{Component} = \text{"wheel\_tire"}$
- $\text{Class} = \text{"lamp broken"} \implies \text{Component} = \text{"lamp"}$
- $\text{Class} = \text{"glass shatter"} \implies \text{Component} = \text{"windshield\_glass"}$

### 3.2 Tier 2: Spatial-Geometric Topological Priors
For structural surface damages (`dent`, `scratch`, `crack`), location within the vehicle coordinate frame strongly constrains component identity. We model the component spatial probability density as a 2D Gaussian mixture over normalized centroid coordinates $(x_c, y_c)$ combined with aspect ratio penalties:
$$L_c(x, y) = \exp\left( - \frac{(x - \mu_{x,c})^2}{2\sigma_{x,c}^2} - \frac{(y - \mu_{y,c})^2}{2\sigma_{y,c}^2} \right) \cdot \exp\left( -\lambda \left| \ln\left(\frac{w}{h}\right) - \ln(\text{AR}_c) \right| \right)$$

### 3.3 Tier 3: Damage-Substrate Affinity Weighting
Different components have distinct physical substrate compositions (e.g. plastic bumpers vs. sheet metal doors vs. glass):
- Plastic bumpers: high affinity for `crack` ($1.55\times$) and `dent` ($1.25\times$).
- Sheet metal doors & fenders: high affinity for `scratch` ($1.35\times$) and `dent` ($1.40\times$).
- Roof & Glass: low affinity for deep dents.

Final component probability:
$$P(C = c \mid \mathbf{b}, \text{dmg}) = \frac{L_c(\mathbf{b}) \cdot \alpha(\text{dmg}, c) \cdot \pi_c}{\sum_{j} L_j(\mathbf{b}) \cdot \alpha(\text{dmg}, j) \cdot \pi_j}$$

---

## 4. Empirical Evaluation Results (Test Set, $N = 784$)

### 4.1 Identified Component Distribution
Evaluated across all 784 held-out test set damage instances in CarDD:

| Component | Test Instances | Share (%) | Primary Copart Sector | Mean Multiplier |
|---|---|---|---|---|
| **door** | 274 | 34.9% | `SIDE` | $1.25\times$ |
| **front_bumper** | 135 | 17.2% | `FRONT END` | $1.15\times$ |
| **hood** | 71 | 9.1% | `FRONT END` | $1.35\times$ |
| **windshield_glass** | 71 | 9.1% | `MINOR DENT/SCRATCHES` | $0.75\times$ |
| **lamp** | 69 | 8.8% | `FRONT END` | $0.85\times$ |
| **fender** | 65 | 8.3% | `SIDE` | $0.95\times$ |
| **roof** | 43 | 5.5% | `ROLLOVER` | $1.80\times$ |
| **wheel_tire** | 32 | 4.1% | `SIDE` | $0.65\times$ |
| **quarter_panel** | 24 | 3.1% | `REAR END` | $1.45\times$ |
| **Total** | **784** | **100.0%** | — | **$1.15\times$ avg** |

### 4.2 Cross-Tabulation: Damage Type vs. Component

| Visual Damage Class | door | front_bumper | hood | glass | lamp | fender | roof | tire | quarter_panel |
|---|---|---|---|---|---|---|---|---|---|
| **dent** ($N=236$) | 114 | 43 | 32 | 0 | 0 | 25 | 11 | 0 | 11 |
| **scratch** ($N=307$) | 148 | 50 | 36 | 0 | 0 | 34 | 26 | 0 | 13 |
| **crack** ($N=69$) | 12 | 42 | 3 | 0 | 0 | 6 | 6 | 0 | 0 |
| **glass shatter** ($N=71$)| 0 | 0 | 0 | 71 | 0 | 0 | 0 | 0 | 0 |
| **lamp broken** ($N=69$) | 0 | 0 | 0 | 0 | 69 | 0 | 0 | 0 | 0 |
| **tire flat** ($N=32$) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 32 | 0 |

### 4.3 Copart Loss Sector Breakdown
- **SIDE:** 371 instances (47.3%) — dominated by door scratches/dents, fender hits, tire flats.
- **FRONT END:** 275 instances (35.1%) — front bumper hits, hood creases, broken headlights.
- **MINOR DENT/SCRATCHES:** 71 instances (9.1%) — glass fractures.
- **ROLLOVER / TOP:** 43 instances (5.5%) — roof and pillar deformations.
- **REAR END:** 24 instances (3.1%) — quarter panel impacts.

---

## 5. Visual Pipeline Demonstration

The complete multi-stage pipeline was verified on test images using `src/component/visualize_detections.py`:
- Stage 1 (YOLO): localizes box coordinates.
- Stage 2 (EfficientNet-B0): refines damage type classification and probability.
- Stage 3 (Component Identifier): identifies body panel, Copart sector, and cost multiplier.
- Outputs saved in `evaluation/component/annotated_samples/`.

---

## 6. Output Artifacts

```
src/component/
├── __init__.py                 # Module exports
├── ontology.py                 # 10-component taxonomy & Copart mappings
├── spatial_geometry.py         # Probabilistic spatial topology priors
├── visual_model.py             # MobileNetV3 visual context extractor
├── identifier.py               # Main VehicleComponentIdentifier engine
├── evaluate_components.py      # Benchmark evaluation script
└── visualize_detections.py     # Multi-layer visualization tool

experiments/
└── phase6_component_identification.yaml

evaluation/component/
├── phase6_component_report_test.json           # Quantitative metrics
├── phase6_component_distribution_test.png      # Component histogram
├── phase6_damage_to_component_heatmap_test.png # Cross-tabulation heatmap
└── annotated_samples/                          # Visual samples with multi-layer tags
```

---

## 7. Transition to Phase 7

- **Validation Gate:** Spatial plausibility $\ge 90\%$; 100% test coverage with zero unmapped instances.
- **Decision:** **PASS.** Vehicle Component Identification module is complete and locked.
- **Next Step:** **Phase 7: Severity Assessment** — assess damage severity (`minor`, `moderate`, `severe`) using `data/Datasets/data3a` (1,631 images) combined with bounding box relative area and structural deformation metrics.
