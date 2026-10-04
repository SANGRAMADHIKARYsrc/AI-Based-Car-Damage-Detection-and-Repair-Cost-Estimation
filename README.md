# AI-Based Car Damage Detection and Repair Cost Estimation

## Research Project — B.Tech CSE AI/ML, 7th Semester

### Research Question

> Can structured damage information extracted from vehicle images (component + damage type + severity + vehicle features) produce useful repair-cost estimates compared with directly predicting cost from an image?

### Pipeline

```
Car Image
    ↓
Image Preprocessing
    ↓
Damage Detection (YOLO)
    ↓
Damage Localization (bounding boxes)
    ↓
Damage Type Classification
    ↓
Vehicle Component Identification
    ↓
Severity Assessment (Minor/Moderate/Severe)
    ↓
Vehicle + Cost Features
    ↓
Repair-Cost Estimation (range, not exact)
    ↓
Final Damage Report
```

### Project Structure

```
car-damage-ai/
├── data/                  # Datasets
│   ├── raw/               # Original downloaded data
│   ├── processed/         # Prepared training data
│   ├── annotations/       # Converted annotations
│   └── cost_data/         # Repair cost records
├── models/                # Trained model weights
│   ├── detection/
│   ├── classification/
│   ├── severity/
│   └── cost/
├── notebooks/             # Jupyter notebooks for EDA
├── src/                   # Source code
│   ├── detection/
│   ├── classification/
│   ├── severity/
│   ├── cost/
│   ├── pipeline/
│   └── utils/
├── evaluation/            # Evaluation results
│   ├── detection/
│   ├── classification/
│   ├── severity/
│   └── cost/
├── app/                   # Web application
│   ├── backend/           # FastAPI
│   └── frontend/          # React
├── research/              # Literature review, reports
├── reports/               # Generated reports
└── experiments/           # Experiment configs and logs
```

### Technology Stack

- **Language:** Python 3.10+
- **Deep Learning:** PyTorch, Ultralytics YOLO
- **CV:** OpenCV
- **Data:** NumPy, Pandas
- **ML:** Scikit-learn, XGBoost
- **Backend:** FastAPI
- **Frontend:** React + Tailwind CSS
- **Database:** SQLite (dev) → PostgreSQL (prod)
- **Deployment:** Docker

### Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Research & Literature Review | ✅ Complete |
| 2 | Dataset Analysis | ✅ Complete |
| 3 | Data Preparation | ✅ Complete |
| 4 | Baseline Damage Detector | ✅ Complete |
| 5 | Damage Classification | ✅ Complete |
| 6 | Component Identification | ✅ Complete |
| 7 | Severity Assessment | ✅ Complete |
| 8 | Repair-Cost Dataset | ✅ Complete |
| 9 | Cost Model | ✅ Complete |
| 10 | Key Research Experiment | ✅ Complete |
| 11 | Generalization | ✅ Complete |
| 12 | Error Analysis | ✅ Complete |
| 13 | Complete Pipeline | ✅ Complete |
| 14 | Web Application | ✅ Complete |

### Future Scope & Commercialization Roadmap

#### 1. Commercial Product Verticals
- **Insurer FNOL (First Notice of Loss) Triage:** Mobile-first claim filing; instant auto-settlement on cosmetic damages ($<\$1,500$) and automatic total-loss routing for structural crashes ($>\$15,000$).
- **B2B Fleet & Car Rental Check-In/Out:** Automated comparison of departure vs return photos for rental fleets (Hertz, Avis, Turo, ride-share fleets) to eliminate dispute friction and recover deposit costs.
- **Collision Repair Center Lead Generation:** Embeddable instant-quote web widget and WhatsApp bot for body shops to generate qualified consumer estimates 24/7.
- **Used Car Marketplace Condition Grading:** Objective visual condition scoring (DSI 0–100) integrated into valuation algorithms for auction platforms and dealership trade-ins.

#### 2. Technical & Enterprise Upgrades
- **Multi-Angle 360° Fusion:** Aggregate 4–8 exterior photographs (front, rear, flanks, close-ups) to de-duplicate damage bounding boxes into a unified vehicle appraisal.
- **Industry Actuarial API Integration:** Connect estimation pipeline with live collision estimating databases (Mitchell, Audatex, CCC ONE) for VIN-specific OEM part numbers and regional labor rates.
- **Anti-Fraud & Image Tamper Detection:** Cryptographic EXIF verification (device hash, timestamp, GPS geotag) combined with generative AI / Photoshop manipulation detectors.
- **Permissive License & Edge Deployment:** Migrate detector backbone to Apache-2.0 architectures (RT-DETR, YOLOv10) for royalty-free commercial distribution; containerize into Docker microservices optimized with TensorRT / ONNX Runtime (<15 ms latency).

### Reproducibility

All experiments record: dataset version, model, hyperparameters, random seed, hardware, metrics, date.

### License

Research project — not for commercial use. Commercial deployment requires proprietary training data and enterprise model licensing.
