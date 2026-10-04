# Phase 14 — Web Application & Deployment Interface

**Project:** AI-Based Car Damage Detection and Repair Cost Estimation Using Deep Learning  
**Date:** 2026-10-05  
**Status:** Complete  

---

## 1. Executive Summary & Objective

Phase 14 delivers a state-of-the-art, interactive web application serving the complete 5-stage multimodal deep learning pipeline. Built with FastAPI on the backend and modern vanilla HTML5/CSS3/JavaScript on the frontend, it enables insurance adjusters, body shop technicians, and vehicle owners to upload collision photographs, input vehicle parameters, and immediately receive comprehensive damage itemizations, localized body component overlays, severity scores, and actuarial repair cost estimates with 90% confidence intervals.

---

## 2. System Architecture

```
User Web Browser (Client)
   │
   ├─► Drag & Drop Vehicle Photo (JPEG/PNG)
   ├─► Vehicle Form (Make, Model Year, Body Architecture, Drivetrain, Operability)
   │
   ▼ HTTP POST /api/predict (Multipart Form)
FastAPI Backend Server (Port 8000)
   │
   ▼ CarDamageAIPipeline
   ├─► Stage 1: YOLOv8n Bounding Box Damage Localization
   ├─► Stage 2: EfficientNet-B0 Damage Type Refinement (15% padding)
   ├─► Stage 3: Automotive Component Ontology (10 body panels)
   ├─► Stage 4: Damage Severity Index (DSI & ResNet18)
   └─► Stage 5: HistGradientBoosting & Quantile Regressors
   │
   ▼ HTTP Response (JSON + Base64 Visual Overlay)
Client UI Render
   ├─► Top KPI Cards: Estimated Dollar Cost, DSI Badge, Primary Damage Sector
   ├─► Cost Driver Alert: Explaining the actuarial rationale
   ├─► Multi-Layer Inspection Viewer (Toggle AI Overlay vs Raw Original)
   ├─► Component Itemization Table (Labor weights, severity, detection confidence)
   └─► Report Exporter (Download JSON / Save Annotated Photo)
```

---

## 3. UI/UX Design & Aesthetics

- **Aesthetic System:** Curated dark mode with glassmorphic cards (`backdrop-filter: blur(16px)`), radial glowing gradients, and subtle borders.
- **Typography:** Google Fonts pairing of **Outfit** (headings, metric displays) and **Inter** (tabular data, form labels).
- **Interactive Features:**
  - 1-Click Test Benchmarks (`BMW 3-Series`, `Toyota RAV4`, `Ford F-150`, `Honda Civic`).
  - Drag-and-drop file upload with live thumbnail preview.
  - Interactive layer toggle (AI Overlay with color-coded bounding boxes and banner vs Original Photo).
  - One-click JSON audit export and annotated photo download.
  - Fully responsive across desktop, tablet, and mobile displays.

---

## 4. Endpoints & API Contract

1. `GET /api/health`
   - Returns operational status, compute device (`cuda` / `cpu`), and boolean loading status for all 5 internal models.
2. `GET /api/samples`
   - Returns catalog of test benchmark presets with associated vehicle specifications.
3. `GET /api/sample/{sample_id}`
   - Triggers pipeline on designated benchmark sample and returns structured report + base64 overlay image.
4. `POST /api/predict`
   - Accepts multipart form data with image file and vehicle metadata.
   - Returns complete JSON report and base64 encoded visual overlay.

---

## 5. Artifacts & Code Implementations

- Backend Entrypoint: [`app/backend/main.py`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/app/backend/main.py)
- Web Client Structure: [`app/frontend/index.html`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/app/frontend/index.html)
- Design Stylesheet: [`app/frontend/style.css`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/app/frontend/style.css)
- Frontend Logic: [`app/frontend/app.js`](file:///c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone%20Project/car-damage-ai/app/frontend/app.js)
- Server Verification: Live and verified on `http://127.0.0.1:8000`.

---

## 6. Project Completion Summary

With the successful deployment of Phase 14, **all 14 development phases of the Capstone Project are 100% complete**.
The project provides a comprehensive, research-grade, and production-ready solution to AI-based vehicle damage detection and repair cost estimation.
