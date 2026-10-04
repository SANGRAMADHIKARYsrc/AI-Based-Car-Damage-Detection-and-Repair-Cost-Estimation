"""Phase 7 — Damage Severity Classifier Architectures.

Provides transfer learning models (ResNet18, EfficientNet-B0)
for 3-class damage severity assessment (minor, moderate, severe)
with bottleneck feature extraction for downstream tabular cost modeling.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .dataset import SEVERITY_CLASSES


class DamageSeverityClassifier(nn.Module):
    """3-class vehicle damage severity classifier."""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 3,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
        elif backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.backbone = base
        else:
            raise ValueError(f"Unsupported backbone: '{backbone_name}'")

        self.in_features = in_features

        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.7),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract visual bottleneck features for cost estimation."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        logits = self.head(feats)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
