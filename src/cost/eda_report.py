"""Phase 8 — Exploratory Data Analysis & Statistical Report.

Analyzes the engineered repair cost dataset, evaluates feature correlations,
tests statistical assumptions (log-normality), and generates research figures.
"""

import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

from src.cost.dataset_builder import NUMERICAL_FEATURES


def run_eda(
    data_path: str = "data/processed/cost_features_train.csv",
    output_dir: str = "evaluation/cost",
) -> Dict[str, Any]:
    """Execute complete statistical analysis and export publication-ready artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading engineered training data from {data_path}...")
    df = pd.read_csv(data_path)

    # 1. Target Distribution Statistics
    raw_cost = df["repairCost"].to_numpy()
    log_cost = df["log_repair_cost"].to_numpy()

    target_stats = {
        "raw_cost": {
            "mean": float(np.mean(raw_cost)),
            "std": float(np.std(raw_cost)),
            "median": float(np.median(raw_cost)),
            "iqr": float(np.percentile(raw_cost, 75) - np.percentile(raw_cost, 25)),
            "min": float(np.min(raw_cost)),
            "max": float(np.max(raw_cost)),
            "p5": float(np.percentile(raw_cost, 5)),
            "p25": float(np.percentile(raw_cost, 25)),
            "p75": float(np.percentile(raw_cost, 75)),
            "p95": float(np.percentile(raw_cost, 95)),
            "skewness": float(stats.skew(raw_cost)),
            "kurtosis": float(stats.kurtosis(raw_cost)),
        },
        "log_cost": {
            "mean": float(np.mean(log_cost)),
            "std": float(np.std(log_cost)),
            "median": float(np.median(log_cost)),
            "iqr": float(np.percentile(log_cost, 75) - np.percentile(log_cost, 25)),
            "skewness": float(stats.skew(log_cost)),
            "kurtosis": float(stats.kurtosis(log_cost)),
            "shapiro_sample_stat": float(stats.shapiro(log_cost[:1000]).statistic),
        },
    }

    # 2. Sector-level cost breakdown
    sector_summary = {}
    for sector, group in df.groupby("primary_damage_clean"):
        if len(group) >= 30:
            sector_summary[sector] = {
                "count": int(len(group)),
                "mean_cost": float(group["repairCost"].mean()),
                "median_cost": float(group["repairCost"].median()),
                "std_cost": float(group["repairCost"].std()),
            }

    # 3. Brand tier breakdown
    brand_summary = {}
    for tier, group in df.groupby("brand_tier"):
        brand_summary[tier] = {
            "count": int(len(group)),
            "mean_cost": float(group["repairCost"].mean()),
            "median_cost": float(group["repairCost"].median()),
            "std_cost": float(group["repairCost"].std()),
        }

    # 4. Operability comparison
    operable_group = df[df["is_operable"] == 1]["repairCost"]
    inoperable_group = df[df["is_operable"] == 0]["repairCost"]
    ttest_operable = stats.ttest_ind(operable_group, inoperable_group, equal_var=False)

    operability_stats = {
        "operable_mean_cost": float(operable_group.mean()),
        "operable_median_cost": float(operable_group.median()),
        "inoperable_mean_cost": float(inoperable_group.mean()),
        "inoperable_median_cost": float(inoperable_group.median()),
        "t_statistic": float(ttest_operable.statistic),
        "p_value": float(ttest_operable.pvalue),
    }

    # 5. Numerical Feature Correlations & Mutual Information
    correlations = {}
    for col in NUMERICAL_FEATURES:
        pearson_r, pearson_p = stats.pearsonr(df[col], df["log_repair_cost"])
        spearman_r, spearman_p = stats.spearmanr(df[col], df["log_repair_cost"])
        correlations[col] = {
            "pearson_r": round(float(pearson_r), 4),
            "pearson_p": float(pearson_p),
            "spearman_r": round(float(spearman_r), 4),
            "spearman_p": float(spearman_p),
        }

    # Mutual information
    X_num = df[NUMERICAL_FEATURES].fillna(0)
    mi_scores = mutual_info_regression(X_num, df["log_repair_cost"], random_state=42)
    mutual_info_dict = {
        feat: round(float(score), 4) for feat, score in zip(NUMERICAL_FEATURES, mi_scores)
    }

    # Compile report
    report = {
        "dataset_split": "train",
        "total_records": len(df),
        "target_statistics": target_stats,
        "sector_breakdown": sector_summary,
        "brand_tier_breakdown": brand_summary,
        "operability_impact": operability_stats,
        "feature_correlations": correlations,
        "mutual_information": mutual_info_dict,
    }

    # Save JSON report
    report_json_path = out_dir / "phase8_dataset_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved quantitative report to: {report_json_path}")

    # -----------------------------------------------------------------------
    # Generate Research Figures
    # -----------------------------------------------------------------------
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Figure 1: Target Distributions (Raw vs Log)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(
        df["repairCost"],
        kde=True,
        bins=50,
        color="#2b5c8f",
        ax=axes[0],
    )
    axes[0].set_title("Raw Repair Cost Distribution (Heavily Skewed)")
    axes[0].set_xlabel("Repair Cost ($ USD)")
    axes[0].set_ylabel("Lot Frequency")
    axes[0].axvline(
        target_stats["raw_cost"]["median"],
        color="red",
        linestyle="--",
        label=f"Median: ${target_stats['raw_cost']['median']:,.0f}",
    )
    axes[0].legend()

    sns.histplot(
        df["log_repair_cost"],
        kde=True,
        bins=50,
        color="#2a9d8f",
        ax=axes[1],
    )
    axes[1].set_title("Log-Transformed Target: ln(1 + Cost)")
    axes[1].set_xlabel("log(1 + Repair Cost)")
    axes[1].set_ylabel("Lot Frequency")
    axes[1].axvline(
        target_stats["log_cost"]["mean"],
        color="red",
        linestyle="--",
        label=f"Mean: {target_stats['log_cost']['mean']:.2f}",
    )
    axes[1].legend()

    plt.tight_layout()
    fig1_path = out_dir / "repair_cost_distributions.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Figure 1 saved to: {fig1_path}")

    # Figure 2: Repair Cost by Damage Sector
    top_sectors = [s for s, stats_dict in sector_summary.items() if stats_dict["count"] >= 50]
    df_filtered_sector = df[df["primary_damage_clean"].isin(top_sectors)]

    plt.figure(figsize=(12, 6))
    order = (
        df_filtered_sector.groupby("primary_damage_clean")["repairCost"]
        .median()
        .sort_values(ascending=False)
        .index
    )
    sns.boxplot(
        data=df_filtered_sector,
        x="repairCost",
        y="primary_damage_clean",
        order=order,
        hue="primary_damage_clean",
        palette="viridis",
        legend=False,
        showfliers=False,
    )
    plt.title("Repair Cost Distribution Across Primary Damage Sectors (Excluding Outliers)")
    plt.xlabel("Repair Cost ($ USD)")
    plt.ylabel("Primary Damage Sector")
    plt.tight_layout()
    fig2_path = out_dir / "cost_by_damage_sector.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Figure 2 saved to: {fig2_path}")

    # Figure 3: Cost by Brand Tier
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df,
        x="brand_tier",
        y="repairCost",
        order=["standard", "premium", "luxury"],
        hue="brand_tier",
        palette="Set2",
        legend=False,
        showfliers=False,
    )
    plt.title("Repair Cost Variation by Vehicle Brand Tier")
    plt.xlabel("Brand Tier")
    plt.ylabel("Repair Cost ($ USD)")
    plt.tight_layout()
    fig3_path = out_dir / "cost_by_brand_tier.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Figure 3 saved to: {fig3_path}")

    # Figure 4: Feature Correlation Heatmap
    corr_cols = NUMERICAL_FEATURES + ["log_repair_cost"]
    corr_matrix = df[corr_cols].corr(method="spearman")

    plt.figure(figsize=(11, 9))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        square=True,
    )
    plt.title("Spearman Rank Correlation: Engineered Features vs Target")
    plt.tight_layout()
    fig4_path = out_dir / "feature_correlation_matrix.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Figure 4 saved to: {fig4_path}")

    return report


if __name__ == "__main__":
    run_eda()
