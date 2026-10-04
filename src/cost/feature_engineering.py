"""Phase 8 — Feature Engineering for Repair Cost Estimation.

Bridges automotive actuarial records (Copart) with visual damage extraction
(CarDD / VehiDE / ResNet18 / EfficientNet-B0 / Component Identifier).
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.component.ontology import (
    COMPONENT_COST_WEIGHTS,
    COMPONENT_TO_COPART_SECTOR,
    VEHICLE_COMPONENTS,
)
from src.severity.dsi import (
    COMPONENT_CRITICALITY,
    DAMAGE_TYPE_SEVERITY,
    compute_instance_dsi,
)

# ---------------------------------------------------------------------------
# Brand Tier Categorization
# ---------------------------------------------------------------------------
LUXURY_BRANDS = {
    "MERCEDES-BENZ", "BMW", "AUDI", "PORSCHE", "LAND ROVER", "JAGUAR",
    "LEXUS", "INFINITI", "CADILLAC", "LINCOLN", "TESLA", "VOLVO",
    "ACURA", "ALFA ROMEO", "MASERATI", "BENTLEY", "FERRARI",
    "ASTON MARTIN", "ROLLS-ROYCE", "MCLAREN", "LAMBORGHINI", "GENESIS",
}

PREMIUM_BRANDS = {
    "BUICK", "CHRYSLER", "GMC", "JEEP", "RAM", "MINI", "VOLKSWAGEN",
}


def categorize_brand(make: str) -> str:
    """Categorize make into luxury, premium, or standard."""
    if not isinstance(make, str):
        return "standard"
    make_clean = make.strip().upper()
    if make_clean in LUXURY_BRANDS:
        return "luxury"
    if make_clean in PREMIUM_BRANDS:
        return "premium"
    return "standard"


# ---------------------------------------------------------------------------
# Body Style Normalization
# ---------------------------------------------------------------------------
def normalize_body_style(body: str) -> str:
    """Normalize free-text body style into canonical vehicle architectures."""
    if not isinstance(body, str):
        return "sedan"
    b = body.strip().upper()
    if any(k in b for k in ["SPORT UTILITY", "SUV", "4DR SPORT"]):
        return "suv"
    if any(k in b for k in ["SEDAN"]):
        return "sedan"
    if any(k in b for k in ["PICKUP", "CREW", "EXT CAB", "CAB"]):
        return "truck"
    if any(k in b for k in ["HATCHBACK"]):
        return "hatchback"
    if any(k in b for k in ["COUPE"]):
        return "coupe"
    if any(k in b for k in ["VAN"]):
        return "van"
    if any(k in b for k in ["WAGON"]):
        return "wagon"
    return "other"


# ---------------------------------------------------------------------------
# Drivetrain Normalization
# ---------------------------------------------------------------------------
def normalize_drivetrain(dt: str) -> str:
    """Normalize drivetrain strings to canonical forms."""
    if not isinstance(dt, str):
        return "UNKNOWN"
    d = dt.strip().upper()
    if "ALL" in d or "AWD" in d:
        return "AWD"
    if "FRONT" in d or "FWD" in d:
        return "FWD"
    if "REAR" in d or "RWD" in d:
        return "RWD"
    if "4X4" in d or "FOUR BY FOUR" in d or "4WD" in d:
        return "4WD"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Damage Sector Mapping & Weights
# ---------------------------------------------------------------------------
SECTOR_TO_COMPONENTS: Dict[str, List[str]] = {
    "FRONT END": ["front_bumper", "hood", "lamp"],
    "REAR END": ["rear_bumper", "quarter_panel", "lamp"],
    "SIDE": ["door", "fender", "wheel_tire"],
    "MINOR DENT/SCRATCHES": ["door", "front_bumper", "fender"],
    "ALL OVER": ["front_bumper", "rear_bumper", "hood", "door", "roof"],
    "HAIL": ["hood", "roof", "quarter_panel"],
    "ROLLOVER": ["roof", "quarter_panel", "windshield_glass"],
    "UNDERCARRIAGE": ["wheel_tire", "front_bumper"],
    "MECHANICAL": ["hood", "front_bumper"],
    "WATER/FLOOD": ["door", "front_bumper", "roof"],
}

SECTOR_SEVERITY_BASE: Dict[str, float] = {
    "FRONT END": 0.70,
    "REAR END": 0.65,
    "SIDE": 0.60,
    "ALL OVER": 0.90,
    "ROLLOVER": 0.95,
    "WATER/FLOOD": 0.85,
    "MECHANICAL": 0.75,
    "UNDERCARRIAGE": 0.70,
    "HAIL": 0.50,
    "MINOR DENT/SCRATCHES": 0.35,
}


def get_sector_cost_weight(sector: str) -> float:
    """Compute composite component labor/parts multiplier for damage sector."""
    comps = SECTOR_TO_COMPONENTS.get(sector, ["door"])
    weights = [COMPONENT_COST_WEIGHTS.get(c, 1.0) for c in comps]
    return float(np.mean(weights))


# ---------------------------------------------------------------------------
# Tabular Feature Engineering Pipeline
# ---------------------------------------------------------------------------
def engineer_features(
    df: pd.DataFrame,
    include_synthetic_visual: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Engineer comprehensive feature matrix for repair cost prediction.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe from copart_repair_cost_clean.csv
    include_synthetic_visual : bool
        If True, synthesizes coherent visual damage attributes (DSI, damage count,
        damaged area ratio, severity grade) matching CarDD/VehiDE empirical distributions.
    seed : int
        Random seed for reproducibility.

    Returns:
    --------
    pd.DataFrame with engineered tabular and visual-bridge features.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()

    # 1. Vehicle Age & Non-Linear Transformations
    current_year = 2026
    if "vehicle_age" not in out.columns:
        out["vehicle_age"] = (current_year - out["year"]).clip(0, 40)
    out["vehicle_age_sq"] = out["vehicle_age"] ** 2
    out["log_vehicle_age"] = np.log1p(out["vehicle_age"])

    # 2. Categorical Standardizations
    out["brand_tier"] = out["make"].apply(categorize_brand)
    out["body_style_clean"] = out["bodyStyle"].apply(normalize_body_style)
    out["drivetrain_clean"] = out["drivetrain"].apply(normalize_drivetrain)

    # 3. Operability Indicators
    runs_drives_str = out["runsDrives"].fillna("").astype(str).str.upper()
    out["is_operable"] = runs_drives_str.str.contains("RUN").astype(int)
    out["starts_only"] = runs_drives_str.str.contains("START").astype(int)

    # 4. Damage Sector Features
    out["primary_damage_clean"] = out["damageDescription"].fillna("UNKNOWN").astype(str)
    out["has_secondary_damage"] = out["secondaryDamage"].notna().astype(int)
    out["secondary_damage_clean"] = out["secondaryDamage"].fillna("NONE").astype(str)

    # Sector labor / parts baseline weight
    out["sector_cost_weight"] = out["primary_damage_clean"].apply(get_sector_cost_weight)
    out["base_sector_severity"] = out["primary_damage_clean"].apply(
        lambda s: SECTOR_SEVERITY_BASE.get(s, 0.50)
    )

    # 5. Visual Damage Feature Bridge (Multimodal Integration)
    if include_synthetic_visual:
        n_samples = len(out)
        
        # Simulate realistic visual damage attributes conditioned on damageDescription & cost
        # Minor sectors -> fewer instances, lower DSI
        # Front/Rear/All Over -> higher instances, higher DSI
        base_sev = out["base_sector_severity"].to_numpy()
        operable = out["is_operable"].to_numpy()
        
        # Damage count (Poisson-like, 1 to 8)
        expected_counts = np.clip(1.5 + base_sev * 3.0 + (1 - operable) * 1.5, 1.0, 7.0)
        damage_counts = rng.poisson(expected_counts).clip(1, 10)
        out["visual_damage_count"] = damage_counts

        # Damaged Area Ratio (log-normal, 0.01 to 0.45)
        mean_area = 0.05 + base_sev * 0.12 + (1 - operable) * 0.08
        noise_area = rng.normal(0, 0.03, size=n_samples)
        area_ratio = np.clip(mean_area + noise_area, 0.005, 0.50)
        out["visual_damaged_area_ratio"] = np.round(area_ratio, 4)

        # Damage Severity Index (DSI: 0 to 100)
        # Combines sector baseline, area ratio, and operational state
        raw_dsi = base_sev * 55.0 + np.sqrt(area_ratio) * 45.0 + (1 - operable) * 12.0
        dsi_noise = rng.normal(0, 6.0, size=n_samples)
        dsi = np.clip(raw_dsi + dsi_noise, 10.0, 98.0)
        out["visual_dsi"] = np.round(dsi, 1)

        # Discrete Severity Grade (0: Minor, 1: Moderate, 2: Severe)
        out["visual_severity_grade"] = np.where(
            out["visual_dsi"] < 38.0, 0,
            np.where(out["visual_dsi"] < 72.0, 1, 2)
        )

        # Most Critical Detected Component
        def pick_component(sector: str) -> str:
            comps = SECTOR_TO_COMPONENTS.get(sector, ["door"])
            return comps[0]

        out["visual_primary_component"] = out["primary_damage_clean"].apply(pick_component)

    return out


# ---------------------------------------------------------------------------
# Multimodal Inference Bridge (Visual Detections -> Feature Vector)
# ---------------------------------------------------------------------------
def extract_cost_features_from_visual_pipeline(
    image_detections: List[Dict[str, Any]],
    vehicle_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert raw visual pipeline detections into cost model input vector.

    Parameters:
    -----------
    image_detections : List of dicts
        Each dict has:
        - "damage_type": str ('scratch', 'dent', etc.)
        - "component": str ('front_bumper', 'hood', etc.)
        - "norm_area": float (bbox area / total image area)
        - "confidence": float (detection confidence)
        - "severity": optional str ('minor', 'moderate', 'severe')
    vehicle_metadata : Dict
        - "year": int
        - "make": str
        - "bodyStyle": str
        - "drivetrain": str
        - "runsDrives": str

    Returns:
    --------
    Dict containing the aligned feature vector ready for cost model prediction.
    """
    # 1. Vehicle attributes
    year = vehicle_metadata.get("year", 2018)
    age = max(0, 2026 - year)
    make = vehicle_metadata.get("make", "TOYOTA")
    body = vehicle_metadata.get("bodyStyle", "SEDAN")
    dt = vehicle_metadata.get("drivetrain", "FWD")
    runs = str(vehicle_metadata.get("runsDrives", "Run & Drive Verified")).upper()

    brand_tier = categorize_brand(make)
    body_style_clean = normalize_body_style(body)
    drivetrain_clean = normalize_drivetrain(dt)
    is_operable = 1 if "RUN" in runs else 0
    starts_only = 1 if "START" in runs else 0

    # 2. Aggregated visual damage features
    damage_count = len(image_detections)
    if damage_count == 0:
        # Undamaged / baseline
        primary_sector = "MINOR DENT/SCRATCHES"
        max_dsi = 15.0
        total_area = 0.0
        primary_comp = "door"
        sev_grade = 0
    else:
        dsi_scores = []
        areas = []
        components = []
        for det in image_detections:
            dmg_type = det.get("damage_type", "scratch")
            comp = det.get("component", "door")
            norm_area = det.get("norm_area", 0.05)
            conf = det.get("confidence", 0.85)

            dsi_res = compute_instance_dsi(dmg_type, comp, norm_area, conf)
            dsi_scores.append(dsi_res["dsi_score"])
            areas.append(norm_area)
            components.append(comp)

        max_dsi = float(np.max(dsi_scores))
        total_area = float(np.clip(np.sum(areas), 0.001, 1.0))
        # Primary component is the one with highest DSI
        idx_max = int(np.argmax(dsi_scores))
        primary_comp = components[idx_max]
        if max_dsi < 38.0 and total_area < 0.08:
            primary_sector = "MINOR DENT/SCRATCHES"
        else:
            primary_sector = COMPONENT_TO_COPART_SECTOR.get(primary_comp, "FRONT END")
        sev_grade = 0 if max_dsi < 38.0 else (1 if max_dsi < 72.0 else 2)

    sector_cost_weight = get_sector_cost_weight(primary_sector)
    base_sector_severity = SECTOR_SEVERITY_BASE.get(primary_sector, 0.50)

    return {
        "vehicle_age": age,
        "vehicle_age_sq": age ** 2,
        "log_vehicle_age": float(np.log1p(age)),
        "brand_tier": brand_tier,
        "body_style_clean": body_style_clean,
        "drivetrain_clean": drivetrain_clean,
        "is_operable": is_operable,
        "starts_only": starts_only,
        "primary_damage_clean": primary_sector,
        "has_secondary_damage": 1 if damage_count > 2 else 0,
        "sector_cost_weight": sector_cost_weight,
        "base_sector_severity": base_sector_severity,
        "visual_damage_count": max(1, damage_count),
        "visual_damaged_area_ratio": round(total_area, 4),
        "visual_dsi": round(max_dsi, 1),
        "visual_severity_grade": sev_grade,
        "visual_primary_component": primary_comp,
    }
