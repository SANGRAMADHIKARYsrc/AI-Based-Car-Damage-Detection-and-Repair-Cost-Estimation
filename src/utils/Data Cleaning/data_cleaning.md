# Car Damage AI — Data Cleaning & Preprocessing Guide

Comprehensive guide detailing dataset sanitation, dataflow pipelines, validation heuristics, and the underlying causal necessity for each cleaning routine across the 4 foundational datasets.

---

## 1. End-to-End Dataflow Diagram

```
Raw Sources:
├── CarDD YOLO (Images + .txt Labels)
├── VehiDE VIA (Polygons + Vietnamese JSON)
├── Data3a Severity (Minor / Mod / Severe Images)
└── Copart Auction Data (100k+ Raw Listing CSVs)
                       │
                       ▼
           [ Data Cleaning Pipeline ]
                       │
  ┌────────────────────┼────────────────────┬────────────────────┐
  ▼                    ▼                    ▼                    ▼
[CarDD Cleaner]   [VehiDE Cleaner]    [Severity Cleaner]   [Copart Cleaner]
• Verify images   • Clean polygons    • Verify image files • Drop duplicates
• Clamp [0, 1]    • Shoelace area     • Drop unreadable    • Coerce types
• Class weights   • Map taxonomy      • Map class IDs (0-2)• Outlier filter
  (Inv, ENS)      • Export to COCO    • Indexed metadata   • Age & log cost
  │                    │                    │                    │
  ▼                    ▼                    ▼                    ▼
class_weights.yaml  instances_coco.json severity_metadata.csv copart_clean.csv
  │                    │                    │                    │
  └────────────────────┴────────────────────┴────────────────────┘
                               │
                               ▼
                   [ Model Ready Datasets ]
            (YOLOv8n / EfficientNet / ResNet18 / HGBR)
```

---

## 2. Directory Structure

All modular cleaning scripts are located in [`src/utils/Data Cleaning`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/src/utils/Data%20Cleaning):

```
src/utils/Data Cleaning/
├── cardd_cleaner.py       # CarDD image integrity & class weighting
├── vehide_cleaner.py      # VehiDE polygon sanitation & COCO converter
├── severity_cleaner.py    # Severity image audit & structured indexer
├── copart_cost_cleaner.py # Copart tabular outlier cleaning & feature engineering
├── run_all_cleaning.py    # Master runner executing all cleaning stages
└── data_cleaning.md       # Full documentation and process reference
```

---

## 3. Dataset Cleaning Modules & Cause Functions

### 3.1 CarDD Cleaner ([`cardd_cleaner.py`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/src/utils/Data%20Cleaning/cardd_cleaner.py))

#### Primary Cause & Need
CarDD suffers from extreme class imbalance (scratch has 3,595 instances while flat tire has only 319 instances; an 11.3:1 ratio). Standard unweighted cross-entropy / bounding box loss causes detectors to ignore rare structural failures. Additionally, orphan labels or corrupted JPEGs crash batch GPU training workers.

#### Key Functions
1. `validate_and_clean_cardd(dataset_dir, output_dir)`:
   - **Cause**: Prevents runtime CUDA pipeline crashes from zero-byte files, truncated JPEGs, and orphan label files without corresponding images.
   - **Process**: Opens each image using PIL, verifies byte stream integrity, checks matching stem in labels, and counts instance distributions across train, val, and test splits.
   - **Output**: Writes `class_weights.yaml` and `class_weights.json` to `data/processed/`.
2. **Class Weighting Formulations**:
   - **Inverse Frequency Weight**:
     $$w_c = \frac{N}{K \cdot N_c}$$
   - **Smoothed Square-Root Weight**:
     $$w_c^{\text{sqrt}} = \sqrt{\frac{\max_i(N_i)}{N_c}}$$
   - **Effective Number of Samples (ENS)**:
     $$E_n = \frac{1 - \beta}{1 - \beta^{N_c}} \quad (\beta = 0.999)$$

---

### 3.2 VehiDE Cleaner ([`vehide_cleaner.py`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/src/utils/Data%20Cleaning/vehide_cleaner.py))

#### Primary Cause & Need
VehiDE annotations are in VGG Image Annotator (VIA) format with non-standard polygon coordinates and Vietnamese category labels (e.g., `VetTray`, `MopMeo`, `VoKinh`). Many polygons are degenerate (e.g., 1 or 2 collinear points) which produce zero area and crash IoU bounding box calculations.

#### Key Functions
1. `polygon_area(x, y)`:
   - **Cause**: Computes exact ground-truth damage pixel area for filtering micro-noise annotations.
   - **Mathematical Logic (Shoelace Formula)**:
     $$A = \frac{1}{2} \left| \sum_{i=1}^{n} (x_i y_{i+1} - x_{i+1} y_i) \right|$$
2. `clean_and_convert_vehide(datasets_dir, mapping_file, output_dir)`:
   - **Cause**: Translates custom Vietnamese annotations to standard COCO JSON format compatible with PyTorch/Torchvision detectors.
   - **Filtering Rules**:
     - Discards regions where $\text{len}(x) < 3$ or $\text{len}(x) \neq \text{len}(y)$.
     - Discards bounding boxes with width $w \le 0$ or height $h \le 0$.
     - Replaces Vietnamese class names using [`class_mapping.yaml`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/data/class_mapping.yaml) to match CarDD damage names.
   - **Output**: `instances_vehide_train.json` and `instances_vehide_val.json` in `data/annotations/`.

---

### 3.3 Severity Dataset Cleaner ([`severity_cleaner.py`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/src/utils/Data%20Cleaning/severity_cleaner.py))

#### Primary Cause & Need
The `data3a` dataset stores images loosely inside nested directories (`01-minor`, `02-moderate`, `03-severe`). Some files are corrupt, have arbitrary aspect ratios, or lack standardized metadata indices needed for PyTorch `Dataset` loaders.

#### Key Functions
1. `clean_and_index_severity_dataset(data3a_dir, repo_root, output_dir)`:
   - **Cause**: Creates an immutable, auditable metadata table mapping each image to its ground-truth severity level and relative file path.
   - **Process**: Validates image headers with PIL, checks for non-image files, encodes target classes to zero-indexed integers (`minor`: 0, `moderate`: 1, `severe`: 2), and logs image resolutions.
   - **Output**: Generates `data/processed/severity_metadata.csv` (1,631 valid records).

---

### 3.4 Copart Repair Cost Cleaner ([`copart_cost_cleaner.py`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/src/utils/Data%20Cleaning/copart_cost_cleaner.py))

#### Primary Cause & Need
Raw auction listings contain duplicate bids, placeholder values ($0 or $1 repair estimates), clerical typo extremes (repair cost > vehicle market ceiling), non-passenger heavy machinery, and missing damage fields. Training directly on raw data severely destabilizes gradient boosting regressors.

#### Key Functions
1. `clean_and_prepare_copart_costs(csv_dir, output_dir, random_seed)`:
   - **Cause**: Cleans noisy tabular salvage data into normalized actuarial collision repair distributions.
   - **Cleaning & Outlier Rules**:
     1. **Deduplication**: `drop_duplicates(subset=["lotId"])` removes re-listed auction lots.
     2. **Numeric Coercion**: `pd.to_numeric(..., errors="coerce")` cleans unformatted currency strings or invalid characters.
     3. **Repair Cost Boundary**: Enforces $\$100 \le \text{repairCost} \le \$150,000$. Drops non-claim placeholders ($0) and severe clerical typos (e.g. $9,999,999).
     4. **Vehicle Year Boundary**: Enforces $1990 \le \text{year} \le 2026$. Drops vintage museum vehicles whose parts pricing does not reflect modern collision supply chains.
     5. **Damage Null Check**: Drops records without a primary `damageDescription`.
   - **Feature Engineering**:
     - $\text{vehicle\_age} = 2026 - \text{year}$
     - $\text{log\_repair\_cost} = \ln(1 + \text{repairCost})$ (controls long right-tail skewness)
   - **Split Stratification**: Deterministic 70% Train, 15% Validation, 15% Test split with fixed seed (`random_state=42`) to ensure zero leakage across experimental phases.
   - **Output**: `data/processed/copart_repair_cost_clean.csv`.

---

## 4. Execution

To execute all cleaning operations sequentially:

```bash
cd "c:\Users\ASUS\OneDrive\Desktop\Antigravity\Capstone Project\car-damage-ai"
python "src/utils/Data Cleaning/run_all_cleaning.py"
```

To run individual modules:

```bash
python "src/utils/Data Cleaning/cardd_cleaner.py"
python "src/utils/Data Cleaning/vehide_cleaner.py"
python "src/utils/Data Cleaning/severity_cleaner.py"
python "src/utils/Data Cleaning/copart_cost_cleaner.py"
```
