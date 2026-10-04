"""Phase 13 — Integration Tests for Complete End-to-End Car Damage AI Pipeline.

Tests:
1. Pipeline initialization and weights loading
2. Full inspection workflow on a CarDD test image
3. Report generator export integrity (JSON, Markdown, and Annotated Image)
"""

import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from src.pipeline.damage_pipeline import CarDamageAIPipeline
from src.pipeline.report_generator import InspectionReportGenerator


@pytest.fixture(scope="module")
def pipeline():
    return CarDamageAIPipeline(device="cpu")  # use cpu for stable test execution


def test_pipeline_initialization(pipeline):
    assert pipeline.two_stage_detector is not None
    assert pipeline.component_identifier is not None
    assert pipeline.severity_assessor is not None
    assert pipeline.cost_estimator is not None


def test_end_to_end_inspection(pipeline, tmp_path):
    # Find a real test image from CarDD
    test_img_dir = REPO_ROOT / "data" / "Datasets" / "test" / "images"
    test_images = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
    if not test_images:
        # Fallback to train
        test_img_dir = REPO_ROOT / "data" / "Datasets" / "train" / "images"
        test_images = list(test_img_dir.glob("*.jpg"))

    assert len(test_images) > 0, "No sample vehicle image found for testing"
    sample_img = test_images[0]

    vehicle_metadata = {
        "year": 2022,
        "make": "BMW",
        "bodyStyle": "SEDAN",
        "drivetrain": "AWD",
        "runsDrives": "Run & Drive Verified",
    }

    result = pipeline.run_inspection(sample_img, vehicle_metadata)

    assert "report" in result
    assert "annotated_image" in result
    report = result["report"]

    assert "vehicle_profile" in report
    assert "inspection_summary" in report
    assert "repair_cost_estimate" in report
    assert report["repair_cost_estimate"]["estimated_cost_usd"] > 0
    assert result["annotated_image"].shape[0] > 0

    # Test report export
    export_paths = InspectionReportGenerator.export_report(
        result, output_dir=str(tmp_path), sample_id="test_car"
    )

    assert Path(export_paths["json_path"]).exists()
    assert Path(export_paths["image_path"]).exists()
    assert Path(export_paths["markdown_path"]).exists()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
