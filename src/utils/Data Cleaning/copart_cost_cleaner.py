"""Copart Auction Repair Cost Dataset Cleaner & Outlier Filter.

Cleans and filters tabular actuarial collision repair data:
- Merges partitioned listing CSV files.
- Deduplicates by unique lotId.
- Enforces strict numeric coercion on repairCost, year, mileage.
- Outlier filtering:
    - $100 <= repairCost <= $150,000 (excludes $0 junk claims and insurance data anomalies)
    - 1990 <= vehicle year <= 2026 (excludes antique records and future invalid years)
    - Mandatory non-null primary damage description
- Feature engineering:
    - vehicle_age = 2026 - year
    - log_repair_cost = ln(1 + repairCost)
- Splits into deterministic 70% Train, 15% Validation, 15% Test.
"""

from pathlib import Path
import glob
import numpy as np
import pandas as pd


def clean_and_prepare_copart_costs(csv_dir: Path, output_dir: Path, random_seed: int = 42) -> dict:
    """Clean and filter Copart auction listings for regression model training.

    Args:
        csv_dir: Directory containing shard CSVs of auction listings.
        output_dir: Directory to save copart_repair_cost_clean.csv.
        random_seed: Random state seed for reproducible train/val/test split.

    Returns:
        dict: Cleaning metrics, dropped records count, and final split counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(glob.glob(str(csv_dir / "*.csv")))

    if not csv_paths:
        return {"error": f"No CSV files found in {csv_dir}"}

    dfs = [pd.read_csv(f, low_memory=False) for f in csv_paths]
    raw_df = pd.concat(dfs, ignore_index=True)
    initial_rows = len(raw_df)

    # 1. Deduplication on lotId
    unique_df = raw_df.drop_duplicates(subset=["lotId"]).copy()
    duplicates_dropped = initial_rows - len(unique_df)

    # 2. Strict numeric casting
    unique_df["repairCost"] = pd.to_numeric(unique_df.get("repairCost"), errors="coerce")
    unique_df["year"] = pd.to_numeric(unique_df.get("year"), errors="coerce")
    if "mileage" in unique_df.columns:
        unique_df["mileage"] = pd.to_numeric(unique_df.get("mileage"), errors="coerce")

    # 3. Outlier and validity filtering
    valid_mask = (
        (unique_df["repairCost"] >= 100.0)
        & (unique_df["repairCost"] <= 150000.0)
        & (unique_df["year"] >= 1990)
        & (unique_df["year"] <= 2026)
        & (unique_df["damageDescription"].notna())
    )

    clean_df = unique_df[valid_mask].copy()
    outliers_dropped = len(unique_df) - len(clean_df)

    # 4. Feature engineering
    clean_df["vehicle_age"] = 2026 - clean_df["year"]
    clean_df["log_repair_cost"] = np.log1p(clean_df["repairCost"])

    keep_cols = [
        "lotId",
        "year",
        "vehicle_age",
        "make",
        "modelGroup",
        "bodyStyle",
        "damageDescription",
        "secondaryDamage",
        "drivetrain",
        "transmission",
        "fuelType",
        "runsDrives",
        "repairCost",
        "log_repair_cost",
    ]
    existing_cols = [c for c in keep_cols if c in clean_df.columns]
    final_df = clean_df[existing_cols].copy()

    # 5. Deterministic stratified random split (70 / 15 / 15)
    shuffled = final_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    total_n = len(shuffled)
    n_train = int(0.70 * total_n)
    n_val = int(0.15 * total_n)

    shuffled["split"] = "train"
    shuffled.loc[n_train : n_train + n_val, "split"] = "val"
    shuffled.loc[n_train + n_val :, "split"] = "test"

    out_csv = output_dir / "copart_repair_cost_clean.csv"
    shuffled.to_csv(out_csv, index=False)

    summary = {
        "raw_records": initial_rows,
        "duplicates_removed": duplicates_dropped,
        "outliers_and_invalids_removed": outliers_dropped,
        "clean_records": total_n,
        "split_distribution": shuffled["split"].value_counts().to_dict(),
        "median_repair_cost": float(shuffled["repairCost"].median()),
        "mean_repair_cost": float(round(shuffled["repairCost"].mean(), 2)),
        "output_path": str(out_csv),
    }

    return summary


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    csv_in = repo_root / "data" / "Datasets" / "copart-dataset-main" / "auction-listings" / "data"
    out_dir = repo_root / "data" / "processed"
    if csv_in.exists():
        res = clean_and_prepare_copart_costs(csv_in, out_dir)
        print("Copart cleaning complete:", res)
    else:
        print("Copart input folder not found at:", csv_in)
