"""Phase 6 — Vehicle Component Identification Benchmark Evaluation.

Evaluates component identification across test damage instances:
- Tests deterministic and spatial-affinity component assignment
- Generates damage-to-component cross-tabulation matrix
- Computes Copart loss sector mapping statistics
- Plots distribution visualizations and heatmap
- Exports evaluation/component/phase6_component_report.json
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.component import (
    COMPONENT_COST_WEIGHTS,
    COMPONENT_TO_COPART_SECTOR,
    VEHICLE_COMPONENTS,
    VehicleComponentIdentifier,
)


def evaluate_components(args):
    """Run component identification evaluation on test set."""
    crops_meta_path = PROJECT_ROOT / "data" / "processed" / "damage_crops" / "crops_metadata.csv"
    if not crops_meta_path.exists():
        raise FileNotFoundError(f"Missing {crops_meta_path}")

    df = pd.read_csv(crops_meta_path)
    test_df = df[df["split"] == args.split].copy().reset_index(drop=True)
    print(f"[Phase 6] Evaluating {len(test_df)} damage instances from split='{args.split}'")

    identifier = VehicleComponentIdentifier()

    results = []
    for idx, row in test_df.iterrows():
        damage_class = row["class_name"]
        norm_bbox = (
            float(row["box_x_center"]),
            float(row["box_y_center"]),
            float(row["box_width"]),
            float(row["box_height"]),
        )
        res = identifier.identify_component(damage_class=damage_class, norm_bbox=norm_bbox)
        results.append({
            "crop_id": row["crop_id"],
            "original_image": row["original_image"],
            "damage_class": damage_class,
            "box_x_center": row["box_x_center"],
            "box_y_center": row["box_y_center"],
            "primary_component": res["primary_component"],
            "confidence": res["confidence"],
            "copart_sector": res["copart_sector"],
            "cost_multiplier": res["cost_multiplier"],
            "method": res["method"],
            "top_candidate": res["candidates"][0]["component"],
            "second_candidate": res["candidates"][1]["component"] if len(res["candidates"]) > 1 else None,
        })

    res_df = pd.DataFrame(results)

    # 1. Component distribution
    comp_counts = res_df["primary_component"].value_counts().to_dict()
    sector_counts = res_df["copart_sector"].value_counts().to_dict()
    avg_conf = float(res_df["confidence"].mean())
    avg_cost_mult = float(res_df["cost_multiplier"].mean())

    # 2. Cross-tabulation: Damage Class vs Primary Component
    ct = pd.crosstab(res_df["damage_class"], res_df["primary_component"])

    # Output directory
    out_dir = PROJECT_ROOT / "evaluation" / "component"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Component Distribution Bar Chart
    plt.figure(figsize=(12, 6))
    comp_series = pd.Series(comp_counts).sort_values(ascending=False)
    sns.barplot(x=comp_series.index, y=comp_series.values, palette="mako")
    plt.title(f"Phase 6 — Identified Vehicle Component Distribution ({args.split.upper()} Set, N={len(test_df)})", fontsize=13, pad=12)
    plt.xlabel("Vehicle Component", fontsize=11)
    plt.ylabel("Damage Instance Count", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    for i, v in enumerate(comp_series.values):
        plt.text(i, v + 2, str(v), ha="center", fontweight="bold", fontsize=10)
    plt.tight_layout()
    dist_plot_path = out_dir / f"phase6_component_distribution_{args.split}.png"
    plt.savefig(dist_plot_path, dpi=200)
    plt.close()

    # Plot 2: Cross-Tabulation Heatmap
    plt.figure(figsize=(13, 7))
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu", cbar=True)
    plt.title("Damage Type vs. Identified Vehicle Component", fontsize=14, pad=12)
    plt.xlabel("Identified Vehicle Component", fontsize=11)
    plt.ylabel("Visual Damage Class", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    heatmap_path = out_dir / f"phase6_damage_to_component_heatmap_{args.split}.png"
    plt.savefig(heatmap_path, dpi=200)
    plt.close()

    # Build report payload
    report = {
        "evaluation_split": args.split,
        "sample_count": len(test_df),
        "mean_confidence": round(avg_conf, 4),
        "mean_cost_multiplier": round(avg_cost_mult, 4),
        "component_distribution": comp_counts,
        "copart_sector_distribution": sector_counts,
        "method_breakdown": res_df["method"].value_counts().to_dict(),
        "cross_tabulation": {
            cls_name: {
                comp: int(ct.loc[cls_name, comp]) if comp in ct.columns else 0
                for comp in VEHICLE_COMPONENTS
            }
            for cls_name in ct.index
        },
    }

    report_path = out_dir / f"phase6_component_report_{args.split}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 65)
    print(f" Phase 6 Vehicle Component Identification — {args.split.upper()} Results")
    print("=" * 65)
    print(f" Sample Count           : {len(test_df)}")
    print(f" Mean Confidence Score  : {avg_conf * 100:.2f}%")
    print(f" Mean Repair Multiplier : {avg_cost_mult:.2f}x")
    print("-" * 65)
    print(f"{'Component':<18} | {'Count':<8} | {'Share (%)':<10} | {'Copart Sector':<16}")
    print("-" * 65)
    for comp, count in comp_series.items():
        share = (count / len(test_df)) * 100
        sec = COMPONENT_TO_COPART_SECTOR.get(comp, "N/A")
        print(f"{comp:<18} | {count:<8} | {share:<10.1f}% | {sec:<16}")
    print("=" * 65)
    print(f"Report saved to: {report_path}")
    print(f"Distribution plot saved to: {dist_plot_path}")
    print(f"Heatmap saved to: {heatmap_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6 Component Identification Evaluation")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    evaluate_components(args)
