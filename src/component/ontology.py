"""Phase 6 — Automotive Component Ontology & Semantic Mappings.

Formalizes the 10-component automotive ontology addressing Gap 2:
Absence of Component-Level Annotations in Damage Datasets.
Bridges visual damage detection (CarDD/VehiDE) with actuarial repair records (Copart).
"""

from typing import Dict, List, Optional, Tuple

# 10 Standard Automotive Repair Components
VEHICLE_COMPONENTS = [
    "front_bumper",
    "rear_bumper",
    "hood",
    "door",
    "fender",
    "quarter_panel",
    "roof",
    "windshield_glass",
    "lamp",
    "wheel_tire",
]

COMPONENT_TO_IDX = {name: idx for idx, name in enumerate(VEHICLE_COMPONENTS)}
IDX_TO_COMPONENT = {idx: name for idx, name in enumerate(VEHICLE_COMPONENTS)}

# Deterministic mappings from damage types that imply specific components
DETERMINISTIC_DAMAGE_MAPPING: Dict[str, str] = {
    "tire flat": "wheel_tire",
    "lamp broken": "lamp",
    "glass shatter": "windshield_glass",
}

# Mapping from components to Copart primary damage sectors (for Phase 8/9 repair cost estimation)
COMPONENT_TO_COPART_SECTOR: Dict[str, str] = {
    "front_bumper": "FRONT END",
    "hood": "FRONT END",
    "lamp": "FRONT END",  # default to front lamp; rear taillight mapped if rear context
    "rear_bumper": "REAR END",
    "quarter_panel": "REAR END",
    "door": "SIDE",
    "fender": "SIDE",
    "wheel_tire": "SIDE",
    "roof": "ROLLOVER",
    "windshield_glass": "MINOR DENT/SCRATCHES",
}

# Average labor & parts replacement cost multipliers relative to a baseline door repair
# Derived from collision estimating industry standards (Mitchell / CCC ONE / Copart)
COMPONENT_COST_WEIGHTS: Dict[str, float] = {
    "front_bumper": 1.15,
    "rear_bumper": 1.10,
    "hood": 1.35,
    "door": 1.25,
    "fender": 0.95,
    "quarter_panel": 1.45,  # Structural weld-in panel, higher repair hours
    "roof": 1.80,           # High structural complexity
    "windshield_glass": 0.75,
    "lamp": 0.85,
    "wheel_tire": 0.65,
}
