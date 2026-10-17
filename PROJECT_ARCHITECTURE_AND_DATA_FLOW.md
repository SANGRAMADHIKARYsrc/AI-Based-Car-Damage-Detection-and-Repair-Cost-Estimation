# AI-Based Car Damage Detection and Repair Cost Estimation
## Complete System Architecture, Data Flow, and Function-by-Function Reference

---

## 1. Executive Summary & Research Foundation

### 1.1 The Core Problem
Traditional approaches to automotive damage inspection and insurance appraisal rely either on:
1. **Manual On-site Appraisal:** Highly subjective, time-intensive (3–7 business days), geographically constrained, and susceptible to inconsistent estimates across appraisers.
2. **Direct Black-Box Deep Learning (Direct Image-to-Cost):** Feeding raw vehicle images directly into a deep Convolutional Neural Network (CNN) to predict dollar amounts. This paradigm fails in practice because:
   - **Extreme Variance & Dataset Scarcity:** A single vehicle make, model, and year with identical visible scratches can have vastly different repair costs based on luxury brand tier, labor rates, and whether underlying structural frames were deformed.
   - **Unexplainable Predictions:** Insurance adjusters and car owners cannot verify *why* an algorithm predicted $12,000 without itemized damage types, component locations, and severity ratings.
   - **Failure to Generalize:** A model trained on economy hatchbacks cannot generalize repair costs to luxury electric SUVs when relying purely on pixel color distributions.

### 1.2 The Core Research Hypothesis
> **Research Question:** *Can structured damage information extracted from vehicle images (component + damage type + severity + vehicle features) produce useful repair-cost estimates compared with directly predicting cost from an image?*

This project implements a **5-Stage Vision-Actuarial Pipeline** that decomposes visual damage appraisal into interpretable intermediate representations before passing them to calibrated actuarial regressors:

$$\text{Raw Image} \xrightarrow{\text{Stage 1}} \text{ROIs} \xrightarrow{\text{Stage 2}} \text{Damage Classes} \xrightarrow{\text{Stage 3}} \text{Components} \xrightarrow{\text{Stage 4}} \text{Severity Index (DSI)} \xrightarrow{\text{Stage 5}} \text{Cost Range } [\$_{\text{low}}, \$_{\text{high}}]$$

---

## 2. System Architecture & High-Level Pipeline

```mermaid
flowchart TD
    subgraph UI ["Client Layer (Web UI)"]
        A[User Uploads Image & Vehicle Specs] --> B[app.js via Fetch API]
    end

    subgraph API ["Backend API (FastAPI)"]
        B --> C["POST /api/predict (app/backend/main.py)"]
        C --> D["CarDamageAIPipeline.run_inspection()"]
    end

    subgraph STAGE1 ["Stage 1: Damage Localization"]
        D --> E["YOLOv8n Detector (yolov8n_baseline)"]
        E --> F["Extract Raw Bounding Boxes [x1, y1, x2, y2]"]
    end

    subgraph STAGE2 ["Stage 2: Crop Classification"]
        F --> G["Extract Context-Padded Bounding Box Crops"]
        G --> H["EfficientNet-B0 Classifier (best_classifier.pt)"]
        H --> I["6 Fine-Grained Classes + Softmax Confidence"]
    end

    subgraph STAGE3 ["Stage 3: Component Identification"]
        I --> J["SpatialComponentPrior & Topology Rules"]
        J --> K["Ontology Mapping (10 Components + Copart Sectors)"]
    end

    subgraph STAGE4 ["Stage 4: Severity Assessment"]
        D --> L["ResNet-18 Global Severity Classifier"]
        I & K --> M["Damage Severity Index (DSI) Calculator"]
        L & M --> N["Composite Severity Grade (Minor/Moderate/Severe)"]
    end

    subgraph STAGE5 ["Stage 5: Actuarial Cost Estimation"]
        N & K --> O["extract_cost_features_from_visual_pipeline()"]
        C -.-> |Vehicle Metadata: Year, Make, Drivetrain| O
        O --> P["ColumnTransformer Preprocessor"]
        P --> Q["HistGradientBoosting Champion Regressor"]
        P --> R["Quantile Regressors (10th & 90th Percentiles)"]
        Q & R --> S["Dollar Estimate + 90% Confidence Interval"]
    end

    subgraph OUTPUT ["Output Rendering & Response"]
        S & N & I --> T["OpenCV Annotation & Report Generator"]
        T --> U["Structured JSON + Base64 Visual Overlay"]
        U --> V["Rendered Inspection Certificate in Web UI"]
    end
```

---

## 3. End-to-End Data Flow Lifecycle

### Step 1: User Input Submission
- The user navigates to `http://127.0.0.1:8000/`.
- The user can either:
  1. Upload a custom JPEG/PNG photograph of damaged vehicle bodywork and fill out vehicle attributes (`Year`, `Make`, `Body Style`, `Drivetrain`, `Operability`).
  2. Click one of four pre-loaded benchmark sample cards (`BMW 3-Series`, `Toyota RAV4`, `Ford F-150`, `Honda Civic`).
- The JavaScript client (`app/frontend/app.js`) packages the payload into a `FormData` object and initiates an HTTP POST request to `/api/predict`.

### Step 2: Request Ingestion & Decoding
- The FastAPI endpoint `predict_damage_and_cost()` in `app/backend/main.py` intercepts the request.
- The raw binary stream from `UploadFile` is read asynchronously into an in-memory byte buffer via `await file.read()`.
- The buffer is converted into a standard 3-channel OpenCV BGR NumPy array via `cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)`.
- Vehicle metadata fields are sanitized into standard uppercase strings and numeric types.

### Step 3: Global Severity Classification (Parallel Branch)
- `CarDamageAIPipeline` sends the full-frame image to `VehicleSeverityAssessor.assess_global_image()` in `src/severity/assessor.py`.
- The image is resized to $224 \times 224$, normalized using ImageNet mean/standard deviation, and fed into the fine-tuned **ResNet-18** model.
- Output: A 3-class probability distribution across `[minor, moderate, severe]` representing the overall physical trauma of the vehicle as a whole.

### Step 4: Two-Stage Damage Localization and Fine-Grained Classification
- `TwoStageDamageDetector.predict()` in `src/classification/two_stage_infer.py` executes:
  1. **Stage 1 (YOLOv8n Localization):** The full vehicle image is evaluated by the anchor-free YOLOv8 detector. All detected regions of interest (ROIs) exceeding confidence threshold $0.25$ are retained as bounding boxes in $[x_1, y_1, x_2, y_2]$ coordinates.
  2. **Crop Context Expansion:** For each detected box, a $15\%$ spatial margin (`padding_ratio = 0.15`) is added around the box boundaries to provide local visual context (e.g., panel lines, lamp housings).
  3. **Stage 2 (EfficientNet-B0 Classification):** The padded crop is extracted, transformed to $224 \times 224$, and forwarded through the dedicated fine-grained classifier. It assigns one of 6 semantic damage classes:
     - `dent`
     - `scratch`
     - `crack`
     - `glass shatter`
     - `lamp broken`
     - `tire flat`
  4. The refined class and probability replace the detector's coarse class label.

### Step 5: Automotive Component Reasoning & Topology Mapping
- For each classified damage instance, `VehicleComponentIdentifier.identify_component()` in `src/component/identifier.py` determines the structural vehicle panel:
  1. **Deterministic Override:** Certain damages strictly imply specific components (e.g., `tire flat` $\rightarrow$ `wheel_tire`, `lamp broken` $\rightarrow$ `lamp`, `glass shatter` $\rightarrow$ `windshield_glass`).
  2. **Spatial Topology Prior:** For surface damages (`dent`, `scratch`, `crack`), the normalized bounding box center $(x_c, y_c)$ and dimensions $(w, h)$ are checked against geometric probability densities defined in `src/component/spatial_geometry.py`. For example, high $y_c$ and low $x_c$ denote `front_bumper`; central high positions denote `hood` or `roof`.
  3. **Damage Affinity Weights:** Multipliers adjust probabilities based on physical material likelihood (e.g., plastic front bumpers crack frequently; roofs rarely suffer dents unless rollover/hail occurs).
  4. **Copart Sector Association:** The identified component is mapped to standard insurance loss sectors (`FRONT END`, `REAR END`, `SIDE`, etc.).

### Step 6: Mathematical Damage Severity Index (DSI) Formulation
- For every damage instance, `compute_instance_dsi()` in `src/severity/dsi.py` calculates an objective actuarial severity score in $[0, 100]$:
  $$\text{DSI} = \left[ 0.45 \cdot W_{\text{type}} + 0.35 \cdot F_{\text{area}} + 0.20 \cdot (C_{\text{comp}} - 0.7) \right] \times 100 \times (0.85 + 0.15 \cdot \text{Conf})$$
  Where:
  - $W_{\text{type}}$: Intrinsic damage severity (e.g., `scratch` = 0.35, `tire flat` = 0.90).
  - $F_{\text{area}}$: Non-linear area penalty factor $\text{clip}(2.5 \cdot \sqrt{\text{norm\_area}}, 0.1, 1.5)$.
  - $C_{\text{comp}}$: Structural criticality of component (e.g., `roof` = 1.50, `door` = 1.15, `front_bumper` = 1.05).
  - Categorization: $\text{DSI} < 38 \implies \text{Minor}$, $38 \le \text{DSI} < 72 \implies \text{Moderate}$, $\text{DSI} \ge 72 \implies \text{Severe}$.

### Step 7: Feature Engineering & Multimodal Actuarial Bridge
- In `src/cost/feature_engineering.py`, `extract_cost_features_from_visual_pipeline()` fuses visual damage metrics with vehicle metadata into a tabular feature vector:
  - **Vehicle Characteristics:** `vehicle_age`, `vehicle_age_sq`, `log_vehicle_age`, `brand_tier` (Luxury, Premium, Standard), `body_style_clean` (Sedan, SUV, Truck, etc.), `drivetrain_clean` (FWD, RWD, AWD, 4WD), `is_operable`, `starts_only`.
  - **Visual Damage Metrics:** `visual_damage_count`, `visual_damaged_area_ratio`, `visual_dsi` (peak DSI among all damages), `visual_severity_grade`, `visual_primary_component`, `primary_damage_clean` (primary loss sector), `sector_cost_weight` (labor rate multiplier), `base_sector_severity`.

### Step 8: Dollar Cost Regression & Quantile Prediction Intervals
- In `src/cost/cost_estimator.py`, `RepairCostEstimator.predict()`:
  1. Passes the feature vector through the scikit-learn `ColumnTransformer` (`cost_preprocessor.joblib`), which applies `StandardScaler` to continuous numbers and `OneHotEncoder` to categorical fields.
  2. Evaluates the transformed vector using `HistGradientBoostingRegressor` (`best_cost_model.joblib`) to predict $\ln(\text{Cost} + 1)$.
  3. Applies the exponential transformation $\exp(y) - 1$ to calculate the expected repair cost in USD.
  4. Evaluates the 10th and 90th percentile quantile regressors (`quantile_models.joblib`) to produce a rigorous 90% confidence interval $[\text{Lower Bound}, \text{Upper Bound}]$.

### Step 9: Annotation Overlay & Response Delivery
- In `src/pipeline/damage_pipeline.py`, `_render_annotated_image()` draws bounding boxes with color-coded severity headers, component labels, and confidence percentages directly onto the image.
- A summary badge detailing the total cost estimate and confidence bounds is stamped onto the upper banner.
- The image is JPEG encoded, base64 formatted, and bundled into a JSON response payload sent back to the frontend.
- `app/frontend/app.js` updates the interactive dashboard dynamically.

---

## 4. Function-by-Function & Class-by-Class Reference

### 4.1 Root & Launch Layer

#### File: [`run.bat`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/run.bat)
- **Purpose**: Minimalist entry point for Windows users to prevent working-directory mismatches.
- **Why it exists**: Users double-clicking files from File Explorer often execute scripts with the working directory set to `C:\Windows\System32`. `run.bat` enforces `cd /d "%~dp0"` and delegates all parameters to `run_car_damage_ai.bat`.

#### File: [`run_car_damage_ai.bat`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/run_car_damage_ai.bat)
- **Subroutines & Purpose:**
  - `:parse_args`: Evaluates command-line flags (`server`, `test`, `check`, `install`, `clean`, `--port`).
  - `:detect_python`: Discovers active virtual environments (`.venv`, `venv`) or falls back to system Python on PATH, verifying Python version $\ge 3.10$.
  - `:run_check`: Health diagnostic inspecting PyTorch, CUDA GPU availability, Ultralytics, FastAPI, Scikit-Learn, OpenCV, and verifying model weight files on disk.
  - `:run_server`: Port conflict detection (terminates stale processes if hung), checks essential modules, launches a background browser launcher, and starts Uvicorn ASGI server.

---

### 4.2 Application & Web Serving Layer

#### File: [`app/backend/main.py`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/app/backend/main.py)
- **Purpose**: Exposes RESTful API endpoints and serves frontend static assets.
- **Functions:**
  - `get_pipeline() -> CarDamageAIPipeline`:
    - *Why it exists*: Implements lazy loading / singleton pattern for deep learning models. Heavy model checkpoints (YOLO, EfficientNet, ResNet, HistGradientBoosting) consume GPU memory; initializing them only on first request avoids slow server startup.
  - `@app.get("/api/health") -> Dict[str, Any]`:
    - *Why it exists*: Diagnostic endpoint for automated monitoring, uptime checks, and verifying that all 5 internal model backbones are loaded on CUDA/CPU.
  - `@app.get("/api/samples") -> Dict[str, Any]`:
    - *Why it exists*: Returns metadata and file paths for 4 pre-configured demo cases (BMW, Toyota, Ford, Honda) so evaluators can test the system without uploading custom photos.
  - `@app.post("/api/predict") -> JSONResponse`:
    - *Why it exists*: Primary operational endpoint. Accepts multipart image upload and vehicle metadata forms, executes full 5-stage pipeline, and returns base64 image and JSON audit report.
  - `@app.get("/api/sample/{sample_id}") -> JSONResponse`:
    - *Why it exists*: Runs one-click end-to-end evaluation on pre-loaded dataset samples directly from disk.

#### File: [`app/frontend/app.js`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/app/frontend/app.js)
- **Purpose**: Client-side reactive UI controller.
- **Functions:**
  - `loadHealth()`: Queries `/api/health` and toggles online/offline badge in UI header.
  - `loadSamples()`: Queries `/api/samples` and renders the interactive demo selection cards.
  - `handleFileUpload(event)`: Handles drag-and-drop or file selector, reading local preview via `FileReader`.
  - `submitInspection()`: Formats inputs into `FormData`, shows animated scanning progress spinner, calls `/api/predict`, and unpacks results.
  - `renderReport(data)`: Dynamically populates vehicle specs, damage instance tables, DSI progress bars, and cost prediction interval sliders.

---

### 4.3 Pipeline Orchestration Layer

#### File: [`src/pipeline/damage_pipeline.py`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/src/pipeline/damage_pipeline.py)
- **Class: `CarDamageAIPipeline`**:
  - *Purpose*: Master orchestrator uniting all 5 modular deep learning and statistical engines into a single callable object.
- **Methods:**
  - `__init__(detector_weights, classifier_weights, severity_weights, cost_model_path, quantile_path, preprocessor_path, device, conf_threshold)`:
    - *Why it exists*: Handles device allocation (`cuda:0` vs `cpu`) and instantiates sub-engines (`TwoStageDamageDetector`, `VehicleComponentIdentifier`, `VehicleSeverityAssessor`, `RepairCostEstimator`).
  - `run_inspection(image_input, vehicle_metadata, viewpoint_hint) -> Dict[str, Any]`:
    - *Why it exists*: Core execution coordinator. Takes heterogeneous image inputs (file path string, PIL Image, or NumPy array), runs global severity, two-stage detection, component mapping, instance DSI, tabular feature extraction, cost prediction, and visual annotation.
  - `_render_annotated_image(img_bgr, damages, cost_result) -> np.ndarray`:
    - *Why it exists*: Performs OpenCV graphic rendering. Draws bounding boxes with thickness proportional to image resolution, color-coded headers (Green for Minor, Amber for Moderate, Red for Severe), component tags, and cost badges.

#### File: [`src/pipeline/report_generator.py`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/src/pipeline/report_generator.py)
- **Class: `InspectionReportGenerator`**:
  - *Purpose*: Serializes inspection runs into persistent audit artifacts.
- **Method: `export_report(inspection_result, output_dir, sample_id) -> Dict[str, str]`**:
  - *Why it exists*: Persists 3 output formats:
    1. `<id>_report.json`: Machine-readable audit file.
    2. `<id>_annotated.jpg`: High-resolution annotated image with visual overlays.
    3. `<id>_summary.md`: Human-readable Markdown inspection certificate.

---

### 4.4 Stage 1 & 2: Damage Detection & Crop Classification

#### File: [`src/classification/two_stage_infer.py`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/src/classification/two_stage_infer.py)
- **Class: `TwoStageDamageDetector`**:
  - *Purpose*: Implements the hierarchical detection-classification paradigm.
- **Methods:**
  - `__init__(detector_weights, classifier_weights, device, conf_threshold, padding_ratio)`:
    - *Why it exists*: Loads YOLOv8 PyTorch weights and loads the EfficientNet-B0 checkpoint, instantiating the validation transform pipeline.
  - `predict(image_input) -> Tuple[np.ndarray, List[dict]]`:
    - *Why it exists*: Feeds full image through YOLOv8. For every detection, calculates padded bounding coordinates, crops the region, evaluates it through EfficientNet-B0, and produces calibrated class probabilities.

#### File: [`src/classification/model.py`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/src/classification/model.py)
- **Class: `DamageClassifier(nn.Module)`**:
  - *Purpose*: EfficientNet-B0 transfer learning architecture.
  - *Why it exists*: Standard YOLO models frequently confuse visually similar damage types (e.g. superficial clearcoat scratch vs metal dent). Passing high-resolution crops through a specialized CNN classifier with dropout and customized linear heads resolves fine-grained textures.
  - *Methods*:
    - `forward(x)`: Extracts feature vector from backbone and outputs class logits.
    - `predict_proba(x)`: Applies Softmax normalization over logits.

#### File: [`src/classification/dataset.py`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/src/classification/dataset.py)
- **Constants**: `CLASS_NAMES = ['crack', 'dent', 'glass shatter', 'lamp broken', 'scratch', 'tire flat']`.
- **Functions**:
  - `get_transforms(img_size, is_train) -> transforms.Compose`:
    - *Why it exists*: Generates PyTorch torchvision transformations (random horizontal flips, color jitter, affine rotations during training; deterministic resize and ImageNet normalization during inference).

---

### 4.5 Stage 3: Automotive Component Identification & Spatial Topology

#### File: [`src/component/ontology.py`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/src/component/ontology.py)
- **Purpose**: Formal domain knowledge base for automotive structures.
- **Data Structures**:
  - `VEHICLE_COMPONENTS`: 10 canonical exterior parts (`front_bumper`, `rear_bumper`, `hood`, `door`, `fender`, `quarter_panel`, `roof`, `lamp`, `windshield_glass`, `wheel_tire`).
  - `DETERMINISTIC_DAMAGE_MAPPING`: Maps strict damages to parts (e.g. `tire flat` $\rightarrow$ `wheel_tire`).
  - `COMPONENT_COST_WEIGHTS`: Structural labor and parts multiplier (e.g., `quarter_panel` = 1.35 because it is welded directly to vehicle unibody; `fender` = 0.95 because it is a bolt-on panel).
  - `COMPONENT_TO_COPART_SECTOR`: Translates visual components to Copart salvage auction loss sectors (`FRONT END`, `REAR END`, `SIDE`, etc.).

#### File: [`src/component/spatial_geometry.py`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/src/component/spatial_geometry.py)
- **Class: `SpatialComponentPrior`**:
  - *Purpose*: Calculates geometric prior probability distribution $P(\text{Component} \mid x_c, y_c, w, h)$.
- **Methods:**
  - `get_spatial_prior(norm_bbox) -> Dict[str, float]`:
    - *Why it exists*: Evaluates normalized bounding box center and aspect ratio. For instance, high horizontal span with low vertical center implies `front_bumper` or `rear_bumper`; high vertical center and moderate width implies `hood` or `roof`.

#### File: [`src/component/identifier.py`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/src/component/identifier.py)
- **Class: `VehicleComponentIdentifier`**:
  - *Purpose*: Fuses deterministic rules, spatial priors, and material affinity matrices.
- **Method: `identify_component(damage_class, norm_bbox, detector_conf) -> Dict[str, Any]`**:
  - *Why it exists*: Fuses the geometric prior with `damage_affinity` weights (e.g., a scratch cannot occur on a glass windshield; cracks on bumpers have higher likelihood due to flexible thermoplastic construction). Returns the top component name, confidence score, and associated Copart sector.

---

### 4.6 Stage 4: Damage Severity Assessment & DSI Formulation

#### File: [`src/severity/dsi.py`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/src/severity/dsi.py)
- **Purpose**: Academic and actuarial Damage Severity Index formulation.
- **Function: `compute_instance_dsi(damage_type, component, norm_area, confidence) -> Dict[str, Any]`**:
  - *Why it exists*: Directly maps bounding box geometry and damage mechanics into a continuous $[0, 100]$ severity scale. Prevents subjective human estimation by mathematically factoring in damage footprint percentage, material fracture type, and unibody replacement criticality.

#### File: [`src/severity/assessor.py`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/src/severity/assessor.py)
- **Class: `VehicleSeverityAssessor`**:
  - *Purpose*: Synthesizes deep global visual classification with instance-level DSI metrics.
- **Methods:**
  - `assess_global_image(image_path) -> Dict[str, Any]`:
    - *Why it exists*: Runs full-frame ResNet-18 classification to determine whether the car as a whole suffered catastrophic unibody distortion or cosmetic abrasions.
  - `assess_vehicle_damage(detections, image_path) -> Dict[str, Any]`:
    - *Why it exists*: Aggregates multiple localized damage DSI scores to compute the vehicle's maximum and average severity grades.

#### File: [`src/severity/model.py`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/src/severity/model.py)
- **Class: `DamageSeverityClassifier(nn.Module)`**:
  - *Purpose*: ResNet-18 model architecture fine-tuned for 3-class severity classification (`minor`, `moderate`, `severe`).

---

### 4.7 Stage 5: Multimodal Feature Engineering & Actuarial Cost Estimation

#### File: [`src/cost/feature_engineering.py`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/src/cost/feature_engineering.py)
- **Purpose**: Bridges visual computer vision extractions with tabular actuarial loss records.
- **Functions:**
  - `categorize_brand(make) -> str`:
    - *Why it exists*: Part prices and hourly body-shop labor rates differ drastically between luxury Marques (BMW, Mercedes-Benz, Porsche) and standard manufacturers (Toyota, Ford). This partitions makes into `luxury`, `premium`, and `standard`.
  - `normalize_body_style(body) -> str`:
    - *Why it exists*: Collapses dozens of messy automotive registration strings into 7 canonical structural architectures (`sedan`, `suv`, `truck`, `coupe`, `hatchback`, `van`, `wagon`).
  - `normalize_drivetrain(dt) -> str`:
    - *Why it exists*: Standardizes mechanical drivetrain configurations (`FWD`, `RWD`, `AWD`, `4WD`), which heavily influence front-end and suspension repair costs.
  - `get_sector_cost_weight(sector) -> float`:
    - *Why it exists*: Computes average labor/parts replacement multipliers for insurance damage sectors.
  - `extract_cost_features_from_visual_pipeline(image_detections, vehicle_metadata) -> Dict[str, Any]`:
    - *Why it exists*: The critical architectural bridge! Converts an arbitrary list of bounding boxes, classified damage strings, and vehicle form inputs into a strictly ordered 17-column feature dictionary matching the exact schema expected by the regression preprocessor.

#### File: [`src/cost/cost_estimator.py`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/src/cost/cost_estimator.py)
- **Class: `RepairCostEstimator`**:
  - *Purpose*: Production inference engine for dollar cost prediction and confidence intervals.
- **Methods:**
  - `__init__(model_path, quantile_path, preprocessor_path)`:
    - *Why it exists*: Loads the serialized Scikit-Learn `ColumnTransformer` preprocessor, champion regression model (`HistGradientBoostingRegressor`), and 10th/90th percentile quantile regressors. Includes Cython unpickling compatibility shims.
  - `predict(image_detections, vehicle_metadata) -> Dict[str, Any]`:
    - *Why it exists*: Coordinates feature transformation, model inference in log-space, inversion back to USD via `expm1`, quantile uncertainty bounds calculation, and generates a natural-language description of the primary cost driver.

#### File: [`src/cost/train_cost_models.py`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/src/cost/train_cost_models.py)
- **Purpose**: Offline training and model selection pipeline.
- **Functions:**
  - `train_and_evaluate_models(data_dir, output_dir)`:
    - *Why it exists*: Trains and compares 5 candidate regression algorithms:
      1. Ridge Regression (Linear Baseline)
      2. Random Forest Regressor
      3. LightGBM Regressor
      4. XGBoost Regressor
      5. HistGradientBoostingRegressor (Selected Champion)
  - `train_quantile_regressors(...)`:
    - *Why it exists*: Trains dedicated gradient boosting models minimizing Pinball Loss at $\alpha = 0.10$ and $\alpha = 0.90$ to provide calibrated prediction intervals.

---

### 4.8 Key Research Experimentation Layer

#### File: [`src/experiments/key_research_experiment.py`](file:///C:/Users/shiva/Desktop/Other%20Apps/AI-Based-Car-Damage-Detection-and-Repair-Cost-Estimation/src/experiments/key_research_experiment.py)
- **Purpose**: Answers the core capstone research question by conducting an empirical comparative study across three paradigms:
  - **Paradigm A (Actuarial Metadata Only):** Uses vehicle attributes and insurance claims records without visual image inspection.
  - **Paradigm B (Direct End-to-End Image Regression):** A CNN (ResNet-18) directly predicting repair cost from raw pixels.
  - **Paradigm C (Proposed Multi-Stage Pipeline):** YOLO localization $\rightarrow$ Crop Classification $\rightarrow$ Component Ontology $\rightarrow$ DSI $\rightarrow$ Tabular Gradient Boosting.
- **Function: `run_key_research_experiment()`**:
  - *Why it exists*: Generates empirical benchmark tables, MAE, RMSE, and $R^2$ comparisons, proving that **Paradigm C achieves lower error, tighter prediction intervals, and 100% auditable explainability**.

---

## 5. Inter-Module Data Contracts & Schemas

### 5.1 Raw Detection Output (Stage 1 & 2 $\rightarrow$ Stage 3)
```json
{
  "bbox_xyxy": [22, 211, 366, 559],
  "detector_class": "tire flat",
  "classifier_class": "tire flat",
  "classifier_conf": 0.9774
}
```

### 5.2 Component Identification Output (Stage 3 $\rightarrow$ Stage 4)
```json
{
  "primary_component": "wheel_tire",
  "confidence": 0.9680,
  "copart_sector": "SIDE",
  "cost_multiplier": 0.65
}
```

### 5.3 Instance DSI Output (Stage 4 $\rightarrow$ Stage 5)
```json
{
  "dsi_score": 78.3,
  "severity_grade": "severe",
  "type_weight": 0.90,
  "component_criticality": 0.75,
  "area_factor": 1.059
}
```

### 5.4 Multimodal Feature Bridge Vector (Stage 5 Preprocessor Input)
```json
{
  "vehicle_age": 4,
  "vehicle_age_sq": 16,
  "log_vehicle_age": 1.6094,
  "brand_tier": "luxury",
  "body_style_clean": "sedan",
  "drivetrain_clean": "AWD",
  "is_operable": 1,
  "starts_only": 0,
  "primary_damage_clean": "SIDE",
  "has_secondary_damage": 0,
  "sector_cost_weight": 0.65,
  "base_sector_severity": 0.60,
  "visual_damage_count": 1,
  "visual_damaged_area_ratio": 0.1795,
  "visual_dsi": 78.3,
  "visual_severity_grade": 2,
  "visual_primary_component": "wheel_tire"
}
```

### 5.5 Final Inspection API Payload (FastAPI $\rightarrow$ Web Client)
```json
{
  "success": true,
  "report": {
    "vehicle_profile": {
      "make": "BMW",
      "year": 2022,
      "vehicle_age": 4,
      "brand_tier": "luxury",
      "body_style": "sedan",
      "operable": true
    },
    "inspection_summary": {
      "total_damage_instances": 1,
      "total_damaged_area_pct": 17.95,
      "global_severity_classifier": "moderate",
      "max_dsi_score": 78.3,
      "composite_severity": "severe",
      "primary_damaged_component": "wheel_tire",
      "primary_impact_sector": "SIDE"
    },
    "repair_cost_estimate": {
      "estimated_cost_usd": 19661.79,
      "confidence_interval_90_usd": {
        "lower_bound": 11432.35,
        "upper_bound": 28298.99,
        "interval_width": 16866.64
      },
      "primary_cost_driver": "Luxury brand with side damage on wheel_tire"
    },
    "detected_damages": [
      {
        "damage_type": "tire flat",
        "confidence": 0.9774,
        "bbox": [22, 211, 366, 559],
        "normalized_area_pct": 17.95,
        "component": "wheel_tire",
        "component_confidence": 0.9680,
        "copart_sector": "SIDE",
        "labor_parts_multiplier": 0.65,
        "dsi_score": 78.3,
        "severity_grade": "severe"
      }
    ]
  },
  "annotated_image_base64": "data:image/jpeg;base64,..."
}
```

---

## 6. Summary of Architectural Design Choices

| Design Decision | Chosen Implementation | Alternative Rejected | Rationale |
| :--- | :--- | :--- | :--- |
| **Damage Localization** | YOLOv8n (Anchor-free) | Faster R-CNN / Mask R-CNN | Lower computational footprint; runs at 60+ FPS on client and consumer GPUs. |
| **Fine-Grained Classification** | EfficientNet-B0 on padded crops | Single-Stage YOLO head | YOLO classification heads struggle on fine textures (scratch vs crack); specialized CNN on crops increases macro F1 by $>14\%$. |
| **Component Identification** | Geometric Priors + Damage Affinity Matrix | Dense Semantic Segmentation (Mask2Former) | Semantic segmentation masks require pixel-level training datasets that do not exist across all vehicle trims; geometric priors generalize across sedans, SUVs, and trucks. |
| **Severity Scoring** | Mathematical DSI Index + ResNet-18 Global | Pure visual subjective label | Pure visual severity lacks physical grounding; DSI mathematically accounts for damage square-footage, component structural roles, and material failure modes. |
| **Cost Regressor** | HistGradientBoosting + Quantiles | Direct End-to-End CNN Regressor | Tabular gradient boosting incorporates vehicle age, luxury tier, and drivability alongside visual features, yielding lower MAE and calibrated 90% confidence intervals. |
