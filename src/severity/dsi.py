"""Phase 7 — Damage Severity Index (DSI) Formulations.

Implements academic and actuarial damage severity indices combining:
1. Normalized bounding box area ratio (damage footprint relative to vehicle)
2. Damage type intrinsic severity weighting
3. Vehicle component structural criticality weighting
"""

from typing import Dict, Union
import numpy as np

# Intrinsic damage severity ratings (0 to 1 scale)
DAMAGE_TYPE_SEVERITY: Dict[str, float] = {
    "scratch": 0.35,       # Superficial clearcoat/paint layer
    "dent": 0.65,          # Plastic deformation of substrate metal/plastic
    "crack": 0.75,         # Material fracture requiring repair/welding/replacement
    "lamp broken": 0.80,   # Component failure
    "glass shatter": 0.85, # Compromised safety & weatherproofing
    "tire flat": 0.90,     # Mobility-inhibiting failure
}

# Component structural criticality factors
COMPONENT_CRITICALITY: Dict[str, float] = {
    "roof": 1.50,
    "quarter_panel": 1.35,
    "hood": 1.20,
    "door": 1.15,
    "front_bumper": 1.05,
    "rear_bumper": 1.00,
    "fender": 0.95,
    "lamp": 0.85,
    "windshield_glass": 0.80,
    "wheel_tire": 0.75,
}


def compute_instance_dsi(
    damage_type: str,
    component: str,
    norm_area: float,
    confidence: float = 1.0,
) -> Dict[str, Union[float, str]]:
    """Compute Damage Severity Index (DSI) for a localized damage instance.

    DSI Score in [0, 100]:
    - [0, 35): Minor
    - [35, 70): Moderate
    - [70, 100]: Severe
    """
    type_weight = DAMAGE_TYPE_SEVERITY.get(damage_type, 0.50)
    comp_crit = COMPONENT_CRITICALITY.get(component, 1.00)

    # Area penalty (log-scaled to handle wide area distributions: 0.001 to 0.50)
    # Median damage box is ~0.08 (8% of image)
    area_factor = float(np.clip(np.sqrt(norm_area) * 2.5, 0.1, 1.5))

    raw_score = (type_weight * 0.45 + area_factor * 0.35 + (comp_crit - 0.7) * 0.20) * 100.0
    dsi_score = float(np.clip(raw_score * (0.85 + 0.15 * confidence), 5.0, 99.0))

    if dsi_score < 38.0:
        grade = "minor"
    elif dsi_score < 72.0:
        grade = "moderate"
    else:
        grade = "severe"

    return {
        "dsi_score": round(dsi_score, 1),
        "severity_grade": grade,
        "type_weight": type_weight,
        "component_criticality": comp_crit,
        "area_factor": round(area_factor, 3),
    }
