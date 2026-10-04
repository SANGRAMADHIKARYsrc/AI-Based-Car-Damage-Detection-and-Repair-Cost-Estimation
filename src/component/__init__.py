"""Phase 6 — Vehicle Component Identification Module.

Provides:
- 10-class vehicle component ontology
- Spatial geometric prior modeling
- Component identification engine linking visual damage with actuarial repair records
"""

from .identifier import VehicleComponentIdentifier
from .ontology import (
    COMPONENT_COST_WEIGHTS,
    COMPONENT_TO_COPART_SECTOR,
    DETERMINISTIC_DAMAGE_MAPPING,
    VEHICLE_COMPONENTS,
)
from .spatial_geometry import SpatialComponentPrior

__all__ = [
    "VehicleComponentIdentifier",
    "SpatialComponentPrior",
    "VEHICLE_COMPONENTS",
    "COMPONENT_TO_COPART_SECTOR",
    "COMPONENT_COST_WEIGHTS",
    "DETERMINISTIC_DAMAGE_MAPPING",
]
