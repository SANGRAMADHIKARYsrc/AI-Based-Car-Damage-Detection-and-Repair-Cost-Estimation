"""Phase 5 — Damage Classification Model Architectures.

Provides transfer learning backbones (EfficientNet-B0, MobileNetV3, ResNet)
with customized heads, feature extraction for downstream tabular cost modeling,
and class-imbalance-resilient loss functions (Weighted Cross-Entropy, Focal Loss).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

SUPPORTED_BACKBONES = ["efficientnet_b0", "mobilenet_v3_large", "resnet18", "resnet50"]


class FocalLoss(nn.Module):
    """Multi-class Focal Loss for handling severe class imbalance."""

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class DamageClassifier(nn.Module):
    """Fine-grained car damage classifier with feature extraction capabilities."""

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_classes: int = 6,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        hidden_dim: int = 256,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Build backbone and identify feature dimension
        if backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.backbone = base
        elif backbone_name == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v3_large(weights=weights)
            in_features = base.classifier[0].in_features
            base.classifier = nn.Identity()
            self.backbone = base
        elif backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
        else:
            raise ValueError(f"Unsupported backbone: '{backbone_name}'. Choose from {SUPPORTED_BACKBONES}")

        self.in_features = in_features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom MLP classification head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.7),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract bottleneck embeddings (used for downstream cost modeling)."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning raw logits."""
        feats = self.forward_features(x)
        logits = self.head(feats)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning softmax probability distributions."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


def build_classifier(
    backbone_name: str = "efficientnet_b0",
    num_classes: int = 6,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    hidden_dim: int = 256,
    freeze_backbone: bool = False,
    device: Optional[torch.device] = None,
) -> DamageClassifier:
    """Convenience factory function."""
    model = DamageClassifier(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        hidden_dim=hidden_dim,
        freeze_backbone=freeze_backbone,
    )
    if device is not None:
        model = model.to(device)
    return model
