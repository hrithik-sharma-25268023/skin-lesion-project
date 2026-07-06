"""
ResNet-50 model for skin lesion classification.

This module provides a ResNet-50 architecture with optional ImageNet
pretrained weights. The final classification layer is replaced to
support an arbitrary number of output classes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50(nn.Module):
    """
    ResNet-50 classifier.

    Parameters
    ----------
    num_classes : int, default=8
        Number of output classes.

    pretrained : bool, default=True
        Whether to load ImageNet pretrained weights.
    """

    def __init__(self, num_classes: int = 8, pretrained: bool = True) -> None:
        """__init__ method"""

        super().__init__()
        weights = (ResNet50_Weights.DEFAULT if pretrained else None)
        self.backbone = resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape
            (batch_size, 3, 224, 224)

        Returns
        -------
        torch.Tensor
            Classification logits.
        """

        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """
        Freeze all backbone layers except the classifier.
        """

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze the entire network.
        """

        for param in self.backbone.parameters():
            param.requires_grad = True

    @property
    def feature_dimension(self) -> int:
        """
        Returns the feature dimension before classification.
        """

        return self.backbone.fc.in_features

    def count_parameters(self) -> int:
        """
        Returns the total number of trainable parameters.
        """

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def summary(self) -> None:
        """
        Prints a simple model summary.
        """

        print("=" * 60)
        print("Model            : ResNet-50")
        print(f"Output Classes   : {self.backbone.fc.out_features}")
        print(f"Feature Dimension: {self.feature_dimension}")
        print(f"Trainable Params : {self.count_parameters():,}")
        print("=" * 60)


def get_resnet50(
    num_classes: int = 8,
    pretrained: bool = True,
) -> ResNet50:
    """
    Returns a ResNet-50 model.

    Parameters
    ----------
    num_classes : int
        Number of output classes.

    pretrained : bool
        Whether to use ImageNet pretrained weights.

    Returns
    -------
    ResNet50
    """

    return ResNet50(
        num_classes=num_classes,
        pretrained=pretrained,
    )
