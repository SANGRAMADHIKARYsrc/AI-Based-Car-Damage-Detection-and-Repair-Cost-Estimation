"""Master Data Cleaning Orchestrator for Car Damage AI.

Executes all cleaning operations across all 4 project datasets:
1. CarDD YOLO damage detection audit and class weighting.
2. VehiDE polygon annotation cleaning & COCO formatting.
3. Severity (data3a) image integrity audit & metadata generation.
4. Copart auction repair cost deduplication, outlier filtering, and feature engineering.
"""

from pathlib import Path
import time
import json

from cardd_cleaner import validate_and_clean_cardd
from vehide_cleaner import clean_and_convert_vehide
from severity_cleaner import clean_and_index_severity_dataset
from copart_cost_cleaner import clean_and_prepare_copart_costs


def run_full_data_cleaning():
    """Run all dataset cleaning procedures and generate consolidated audit log."""
    start_time = time.time()
    repo_root = Path(__file__).resolve().parents[3]
    datasets_dir = repo_root / "data" / "Datasets"
    processed_dir = repo_root / "data" / "processed"
    annotations_dir = repo_root / "data" / "annotations"
    class_mapping = repo_root / "data" / "class_mapping.yaml"

    processed_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("      CAR DAMAGE AI — COMPLETE DATA CLEANING PIPELINE            ")
    print("=================================================================\n")

    report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "tasks": {}}

    # Step 1: CarDD Cleaning
    print("[1/4] Cleaning & Auditing CarDD YOLO Dataset...")
    cardd_dir = datasets_dir if (datasets_dir / "train").exists() else (datasets_dir / "cardd_yolo")
    if cardd_dir.exists():
        cardd_res = validate_and_clean_cardd(cardd_dir, processed_dir)
        report["tasks"]["cardd_yolo"] = cardd_res
        print("  --> CarDD clean. Total instances:", cardd_res["total_instances"])
    else:
        print("  [SKIP] CarDD directory not found.")

    # Step 2: VehiDE Cleaning & COCO Conversion
    print("\n[2/4] Cleaning & Converting VehiDE Polygon Annotations...")
    if datasets_dir.exists() and class_mapping.exists():
        vehide_res = clean_and_convert_vehide(datasets_dir, class_mapping, annotations_dir)
        report["tasks"]["vehide"] = vehide_res
        print("  --> VehiDE clean. Splits processed:", list(vehide_res.keys()))
    else:
        print("  [SKIP] VehiDE data or class_mapping.yaml missing.")

    # Step 3: Severity data3a Cleaning
    print("\n[3/4] Cleaning & Indexing data3a Severity Dataset...")
    data3a_dir = datasets_dir / "data3a"
    if data3a_dir.exists():
        sev_res = clean_and_index_severity_dataset(data3a_dir, repo_root, processed_dir)
        report["tasks"]["severity_data3a"] = sev_res
        print("  --> Severity clean. Valid images:", sev_res.get("total_valid_images"))
    else:
        print("  [SKIP] data3a folder not found.")

    # Step 4: Copart Cost Cleaning
    print("\n[4/4] Deduplicating & Filtering Copart Repair Cost Data...")
    copart_csv_dir = datasets_dir / "copart-dataset-main" / "auction-listings" / "data"
    if copart_csv_dir.exists():
        copart_res = clean_and_prepare_copart_costs(copart_csv_dir, processed_dir)
        report["tasks"]["copart_costs"] = copart_res
        print("  --> Copart clean. Final valid records:", copart_res.get("clean_records"))
    else:
        print("  [SKIP] Copart CSV folder not found.")

    elapsed = round(time.time() - start_time, 2)
    report["elapsed_seconds"] = elapsed

    # Save summary report
    out_summary = processed_dir / "data_cleaning_run_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n=================================================================")
    print(f" Data Cleaning Completed in {elapsed}s")
    print(f" Summary log saved to: {out_summary}")
    print(f"=================================================================\n")


if __name__ == "__main__":
    run_full_data_cleaning()
