"""Phase 14 — FastAPI Backend for Car Damage AI Application.

Exposes RESTful endpoints for:
- Live multi-stage damage detection & repair cost estimation
- Pre-loaded vehicle sample demonstrations
- System health & model status
- Static web client serving
"""

import base64
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image

from src.pipeline.damage_pipeline import CarDamageAIPipeline

app = FastAPI(
    title="Car Damage AI & Cost Estimator API",
    description="Multimodal deep learning system for damage detection, component reasoning, and actuarial cost prediction.",
    version="1.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (lazy loaded)
pipeline: Optional[CarDamageAIPipeline] = None


def get_pipeline() -> CarDamageAIPipeline:
    global pipeline
    if pipeline is None:
        pipeline = CarDamageAIPipeline(device="cuda")
    return pipeline


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    """Verify backend and server readiness."""
    global pipeline
    if pipeline is not None:
        models_loaded = {
            "yolo_detector": pipeline.two_stage_detector.detector is not None,
            "crop_classifier": pipeline.two_stage_detector.classifier is not None,
            "component_identifier": pipeline.component_identifier is not None,
            "severity_assessor": pipeline.severity_assessor is not None,
            "cost_regressor": pipeline.cost_estimator.model is not None,
        }
        device = pipeline.device
    else:
        models_loaded = {"status": "lazy_load_on_first_inference"}
        device = "cuda"
    return {
        "status": "healthy",
        "device": device,
        "models_loaded": models_loaded,
    }


@app.get("/api/samples")
def list_samples() -> Dict[str, Any]:
    """Provide pre-loaded demo images and metadata for instant testing."""
    test_img_dir = REPO_ROOT / "data" / "Datasets" / "test" / "images"
    sample_files = sorted(list(test_img_dir.glob("*.jpg")))[:4]

    presets = [
        {"id": "sample_1", "name": "Front Collision (BMW 3-Series)", "year": 2022, "make": "BMW", "bodyStyle": "SEDAN", "drivetrain": "AWD", "runsDrives": "Run & Drive Verified"},
        {"id": "sample_2", "name": "Front Bumper Scratch (Toyota RAV4)", "year": 2018, "make": "TOYOTA", "bodyStyle": "SUV", "drivetrain": "FWD", "runsDrives": "Run & Drive Verified"},
        {"id": "sample_3", "name": "Side Panel Dent (Ford F-150)", "year": 2015, "make": "FORD", "bodyStyle": "PICKUP", "drivetrain": "4WD", "runsDrives": "Run & Drive Verified"},
        {"id": "sample_4", "name": "Severe Crash (Honda Civic)", "year": 2020, "make": "HONDA", "bodyStyle": "SEDAN", "drivetrain": "FWD", "runsDrives": "DEFAULT"},
    ]

    out = []
    for i, p in enumerate(presets):
        if i < len(sample_files):
            p["filename"] = sample_files[i].name
            out.append(p)
    return {"samples": out}


@app.post("/api/predict")
async def predict_damage_and_cost(
    file: UploadFile = File(...),
    year: int = Form(2021),
    make: str = Form("TOYOTA"),
    bodyStyle: str = Form("SEDAN"),
    drivetrain: str = Form("FWD"),
    runsDrives: str = Form("Run & Drive Verified"),
) -> JSONResponse:
    """Run full multimodal inspection on uploaded vehicle photograph."""
    try:
        pipe = get_pipeline()
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file format.")

        vehicle_metadata = {
            "year": year,
            "make": make.strip().upper(),
            "bodyStyle": bodyStyle.strip(),
            "drivetrain": drivetrain.strip(),
            "runsDrives": runsDrives.strip(),
        }

        # Run pipeline
        result = pipe.run_inspection(image_input=img_bgr, vehicle_metadata=vehicle_metadata)

        # Encode annotated image to base64
        _, buffer = cv2.imencode(".jpg", result["annotated_image"])
        b64_image = base64.b64encode(buffer).decode("utf-8")

        return JSONResponse(
            content={
                "success": True,
                "report": result["report"],
                "annotated_image_base64": f"data:image/jpeg;base64,{b64_image}",
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/api/sample/{sample_id}")
def run_sample_inspection(sample_id: str) -> JSONResponse:
    """Run inspection on one of the pre-loaded benchmark sample images."""
    try:
        pipe = get_pipeline()
        test_img_dir = REPO_ROOT / "data" / "Datasets" / "test" / "images"
        sample_files = sorted(list(test_img_dir.glob("*.jpg")))

        id_map = {
            "sample_1": (sample_files[0] if len(sample_files) > 0 else None, {"year": 2022, "make": "BMW", "bodyStyle": "SEDAN", "drivetrain": "AWD", "runsDrives": "Run & Drive Verified"}),
            "sample_2": (sample_files[1] if len(sample_files) > 1 else None, {"year": 2018, "make": "TOYOTA", "bodyStyle": "SUV", "drivetrain": "FWD", "runsDrives": "Run & Drive Verified"}),
            "sample_3": (sample_files[2] if len(sample_files) > 2 else None, {"year": 2015, "make": "FORD", "bodyStyle": "PICKUP", "drivetrain": "4WD", "runsDrives": "Run & Drive Verified"}),
            "sample_4": (sample_files[3] if len(sample_files) > 3 else None, {"year": 2020, "make": "HONDA", "bodyStyle": "SEDAN", "drivetrain": "FWD", "runsDrives": "DEFAULT"}),
        }

        if sample_id not in id_map or id_map[sample_id][0] is None:
            raise HTTPException(status_code=404, detail="Sample not found.")

        img_path, veh_meta = id_map[sample_id]
        result = pipe.run_inspection(image_input=img_path, vehicle_metadata=veh_meta)

        _, buffer = cv2.imencode(".jpg", result["annotated_image"])
        b64_image = base64.b64encode(buffer).decode("utf-8")

        return JSONResponse(
            content={
                "success": True,
                "report": result["report"],
                "annotated_image_base64": f"data:image/jpeg;base64,{b64_image}",
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# Mount frontend static directory if exists
frontend_dir = REPO_ROOT / "app" / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
