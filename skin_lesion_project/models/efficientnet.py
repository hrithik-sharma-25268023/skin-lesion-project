"""
EfficientNet models for skin lesion classification.

This module provides EfficientNet-B0 and EfficientNet-B3 architectures
with optional ImageNet pretrained weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B3_Weights, efficientnet_b0, efficientnet_b3


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 classifier.

    Parameters
    ----------
    num_classes : int, default=8
        Number of output classes.
    pretrained : bool, default=True
        Whether to use ImageNet pretrained weights.
    """

    def __init__(self, num_classes: int = 8, pretrained: bool = True) -> None:

        super().__init__()
        weights = (EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freeze all backbone layers except classifier."""

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for parameter in self.backbone.classifier.parameters():
            parameter.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze the complete network."""

        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    @property
    def feature_dimension(self) -> int:
        """Returns feature dimension before classification."""
        return self.backbone.classifier[1].in_features

    def count_parameters(self) -> int:
        """Returns number of trainable parameters."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def summary(self) -> None:
        """Print model summary."""

        print("=" * 60)
        print("Model            : EfficientNet-B0")
        print(f"Output Classes   : {self.backbone.classifier[1].out_features}")
        print(f"Feature Dimension: {self.feature_dimension}")
        print(f"Trainable Params : {self.count_parameters():,}")
        print("=" * 60)


class EfficientNetB3(nn.Module):
    """
    EfficientNet-B3 classifier.

    Parameters
    ----------
    num_classes : int, default=8
        Number of output classes.
    pretrained : bool, default=True
        Whether to use ImageNet pretrained weights.
    """

    def __init__(self, num_classes: int = 8, pretrained: bool = True) -> None:

        super().__init__()
        weights = (EfficientNet_B3_Weights.DEFAULT if pretrained else None)
        self.backbone = efficientnet_b3(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freeze all backbone layers except classifier."""

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for parameter in self.backbone.classifier.parameters():
            parameter.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze the complete network."""

        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    @property
    def feature_dimension(self) -> int:
        """Returns feature dimension before classification."""
        return self.backbone.classifier[1].in_features

    def count_parameters(self) -> int:
        """Returns number of trainable parameters."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def summary(self) -> None:
        """Print model summary."""

        print("Model            : EfficientNet-B3")
        print(f"Output Classes   : {self.backbone.classifier[1].out_features}")
        print(f"Feature Dimension: {self.feature_dimension}")
        print(f"Trainable Params : {self.count_parameters():,}")


def get_efficientnet_b0(num_classes: int = 8, pretrained: bool = True) -> EfficientNetB0:
    """
    Returns an EfficientNet-B0 model.
    """

    return EfficientNetB0(num_classes=num_classes, pretrained=pretrained)


def get_efficientnet_b3(num_classes: int = 8, pretrained: bool = True) -> EfficientNetB3:
    """
    Returns an EfficientNet-B3 model.
    """

    return EfficientNetB3(num_classes=num_classes, pretrained=pretrained)
