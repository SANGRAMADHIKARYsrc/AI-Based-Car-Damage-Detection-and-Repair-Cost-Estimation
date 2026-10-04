"""Phase 7 — Damage Severity Assessment Package.

Provides:
- 3-class severity dataset loader & transforms
- ResNet18 / EfficientNet-B0 severity classifiers
- Instance-level Damage Severity Index (DSI)
- Holistic vehicle severity assessor
"""

from .assessor import VehicleSeverityAssessor
from .dataset import (
    IDX_TO_SEVERITY,
    SEVERITY_CLASSES,
    SEVERITY_TO_IDX,
    SeverityDataset,
    get_severity_transforms,
    prepare_stratified_severity_splits,
)
from .dsi import compute_instance_dsi
from .model import DamageSeverityClassifier

__all__ = [
    "VehicleSeverityAssessor",
    "DamageSeverityClassifier",
    "SeverityDataset",
    "SEVERITY_CLASSES",
    "SEVERITY_TO_IDX",
    "IDX_TO_SEVERITY",
    "compute_instance_dsi",
    "get_severity_transforms",
    "prepare_stratified_severity_splits",
]
