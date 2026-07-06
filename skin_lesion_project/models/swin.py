"""
Swin Transformer Tiny model for skin lesion classification.

This module provides a Swin Transformer Tiny architecture with optional
ImageNet pretrained weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchvision.models import (
    Swin_T_Weights,
    swin_t,
)


class SwinTransformer(nn.Module):
    """
    Swin Transformer Tiny classifier.

    Parameters
    ----------
    num_classes : int, default=8
        Number of output classes.

    pretrained : bool, default=True
        Whether to load ImageNet pretrained weights.
    """

    def __init__(self, num_classes: int = 8, pretrained: bool = True) -> None:

        super().__init__()

        weights = (Swin_T_Weights.DEFAULT if pretrained else None)
        self.backbone = swin_t(weights=weights)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, num_classes)

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
        Freeze all backbone layers except the classification head.
        """

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for parameter in self.backbone.head.parameters():
            parameter.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze the complete model.
        """

        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    @property
    def feature_dimension(self) -> int:
        """
        Returns the feature dimension before classification.
        """

        return self.backbone.head.in_features

    def count_parameters(self) -> int:
        """
        Returns the number of trainable parameters.
        """

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def summary(self) -> None:
        """
        Prints a simple model summary.
        """

        print("Model            : Swin Transformer Tiny")
        print(f"Output Classes   : {self.backbone.head.out_features}")
        print(f"Feature Dimension: {self.feature_dimension}")
        print(f"Trainable Params : {self.count_parameters():,}")


def get_swin_t(num_classes: int = 8, pretrained: bool = True) -> SwinTransformer:
    """
    Returns a Swin Transformer Tiny model.

    Parameters
    ----------
    num_classes : int
        Number of output classes.

    pretrained : bool
        Whether to use ImageNet pretrained weights.

    Returns
    -------
    SwinTransformer
    """

    return SwinTransformer(num_classes=num_classes, pretrained=pretrained)
