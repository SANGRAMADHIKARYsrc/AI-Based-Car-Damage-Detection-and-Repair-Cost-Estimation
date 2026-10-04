"""Phase 13 — Complete Pipeline Demonstration Script.

Runs end-to-end inspection on real CarDD test images and outputs
inspection certificates with high-resolution visual overlays.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.damage_pipeline import CarDamageAIPipeline
from src.pipeline.report_generator import InspectionReportGenerator


def run_pipeline_demo():
    print("=== Running End-to-End Pipeline Demonstration ===")
    pipeline = CarDamageAIPipeline(device="cuda")

    test_img_dir = REPO_ROOT / "data" / "Datasets" / "test" / "images"
    test_images = sorted(list(test_img_dir.glob("*.jpg")))[:3]

    vehicles = [
        {"year": 2022, "make": "BMW", "bodyStyle": "SEDAN", "drivetrain": "AWD", "runsDrives": "Run & Drive Verified"},
        {"year": 2017, "make": "TOYOTA", "bodyStyle": "SUV", "drivetrain": "FWD", "runsDrives": "Run & Drive Verified"},
        {"year": 2014, "make": "FORD", "bodyStyle": "PICKUP", "drivetrain": "4WD", "runsDrives": "DEFAULT"},
    ]

    out_dir = REPO_ROOT / "reports" / "pipeline_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (img_path, veh_meta) in enumerate(zip(test_images, vehicles), 1):
        print(f"\nProcessing Sample {idx}: {img_path.name} ({veh_meta['year']} {veh_meta['make']})...")
        res = pipeline.run_inspection(img_path, veh_meta)

        paths = InspectionReportGenerator.export_report(
            res, output_dir=str(out_dir), sample_id=f"inspection_sample_{idx}"
        )
        rep = res["report"]
        print(
            f"  Damages Found:    {rep['inspection_summary']['total_damage_instances']}\n"
            f"  Composite DSI:    {rep['inspection_summary']['max_dsi_score']} ({rep['inspection_summary']['composite_severity']})\n"
            f"  Estimated Cost:   ${rep['repair_cost_estimate']['estimated_cost_usd']:,.2f}\n"
            f"  90% CI:           [${rep['repair_cost_estimate']['confidence_interval_90_usd']['lower_bound']:,.2f} - ${rep['repair_cost_estimate']['confidence_interval_90_usd']['upper_bound']:,.2f}]\n"
            f"  Annotated Image:  {paths['image_path']}"
        )

    print(f"\nAll demonstration reports successfully generated in: {out_dir}")


if __name__ == "__main__":
    run_pipeline_demo()
