"""Phase 6 — Visual Component Feature Model.

Extracts visual substrate representations (metal sheet, plastic bumper,
glass, lamp lens, tire tread) to distinguish ambiguous body panels.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .ontology import VEHICLE_COMPONENTS


class VehicleComponentVisualModel(nn.Module):
    """Deep visual context extractor for vehicle component discrimination."""

    def __init__(
        self,
        num_components: int = len(VEHICLE_COMPONENTS),
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v3_small(weights=weights)
        in_features = base.classifier[0].in_features
        base.classifier = nn.Identity()
        self.backbone = base

        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_components),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
