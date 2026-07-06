"""
Vision Transformer (ViT-B/16) model for skin lesion classification.

This module provides a Vision Transformer (ViT-B/16) architecture
with optional ImageNet pretrained weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchvision.models import ViT_B_16_Weights, vit_b_16


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT-B/16) classifier.

    Parameters
    ----------
    num_classes : int, default=8
        Number of output classes.

    pretrained : bool, default=True
        Whether to load ImageNet pretrained weights.
    """

    def __init__(self, num_classes: int = 8, pretrained: bool = True) -> None:

        super().__init__()
        weights = (ViT_B_16_Weights.DEFAULT if pretrained else None)
        self.backbone = vit_b_16(weights=weights)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Linear(in_features, num_classes)

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
        Freeze all backbone layers except classification head.
        """

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for parameter in self.backbone.heads.parameters():
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
        Returns feature dimension before classification.
        """

        return self.backbone.heads.head.in_features

    def count_parameters(self) -> int:
        """
        Returns number of trainable parameters.
        """

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def summary(self) -> None:
        """
        Prints model summary.
        """

        print("Model            : Vision Transformer (ViT-B/16)")
        print(f"Output Classes   : {self.backbone.heads.head.out_features}")
        print(f"Feature Dimension: {self.feature_dimension}")
        print(f"Trainable Params : {self.count_parameters():,}")

def get_vit_b16(num_classes: int = 8, pretrained: bool = True) -> VisionTransformer:
    """
    Returns a Vision Transformer (ViT-B/16).

    Parameters
    ----------
    num_classes : int
        Number of output classes.

    pretrained : bool
        Whether to use ImageNet pretrained weights.

    Returns
    -------
    VisionTransformer
    """

    return VisionTransformer(num_classes=num_classes, pretrained=pretrained)
