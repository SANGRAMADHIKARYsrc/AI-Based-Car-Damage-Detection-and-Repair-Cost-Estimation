"""Phase 5 — Damage Classification Module.

Provides fine-grained classification for localized vehicle damage crops:
- Dataset and crop extraction utilities
- Multi-architecture classifier models (EfficientNet, ResNet, MobileNet)
- Training, evaluation, and two-stage detection+classification inference
"""

from .dataset import DamageCropDataset, extract_damage_crops, get_transforms
from .model import DamageClassifier, build_classifier

__all__ = [
    "DamageCropDataset",
    "extract_damage_crops",
    "get_transforms",
    "DamageClassifier",
    "build_classifier",
]
