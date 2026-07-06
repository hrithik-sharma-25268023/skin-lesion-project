"""
ConvNeXt-Tiny model for skin lesion classification.

This module provides a ConvNeXt-Tiny architecture with optional
ImageNet pretrained weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


class ConvNeXtTiny(nn.Module):
    """
    ConvNeXt-Tiny classifier.

    Parameters
    ----------
    num_classes : int, default=8
        Number of output classes.

    pretrained : bool, default=True
        Whether to use ImageNet pretrained weights.
    """

    def __init__(self, num_classes: int = 8, pretrained: bool = True) -> None:

        super().__init__()
        weights = (ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        self.backbone = convnext_tiny(weights=weights)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape
            (batch_size, 3, 224, 224)

        Returns
        -------
        torch.Tensor
            Classification logits.
        """

        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """
        Freeze all backbone layers except classifier.
        """

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for parameter in self.backbone.classifier.parameters():
            parameter.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze the complete network.
        """

        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    @property
    def feature_dimension(self) -> int:
        """
        Returns feature dimension before classification.
        """

        return self.backbone.classifier[2].in_features

    def count_parameters(self) -> int:
        """
        Returns the number of trainable parameters.
        """

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def summary(self) -> None:
        """
        Prints a model summary.
        """

        print("=" * 60)
        print("Model            : ConvNeXt-Tiny")
        print(f"Output Classes   : {self.backbone.classifier[2].out_features}")
        print(f"Feature Dimension: {self.feature_dimension}")
        print(f"Trainable Params : {self.count_parameters():,}")
        print("=" * 60)


def get_convnext_tiny(num_classes: int = 8, pretrained: bool = True) -> ConvNeXtTiny:
    """
    Returns a ConvNeXt-Tiny model.

    Parameters
    ----------
    num_classes : int
        Number of output classes.

    pretrained : bool
        Whether to use ImageNet pretrained weights.

    Returns
    -------
    ConvNeXtTiny
    """

    return ConvNeXtTiny(num_classes=num_classes, pretrained=pretrained)
