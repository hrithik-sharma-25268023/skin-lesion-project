"""
Swin Transformer Tiny model for skin lesion classification.

This module provides a Swin Transformer Tiny architecture with optional
ImageNet pretrained weights, plus staged unfreezing utilities that make
gradual fine-tuning straightforward.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torchvision.models import Swin_T_Weights, swin_t


class SwinTransformer(nn.Module):
    """
    Swin Transformer Tiny classifier.
    """

    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()

        weights = Swin_T_Weights.DEFAULT if pretrained else None

        # torchvision 0.19.x does not allow overriding
        # stochastic_depth_prob in swin_t()
        self.backbone = swin_t(weights=weights)

        self.in_features = self.backbone.head.in_features

        self.backbone.head = nn.Sequential(
            nn.Dropout(p=head_dropout),
            nn.Linear(self.in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    # ------------------------------------------------------------------
    # Freezing / unfreezing utilities
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all backbone layers except the classification head."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for parameter in self.backbone.head.parameters():
            parameter.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze the complete backbone."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def unfreeze_last_stages(self, num_stages: int = 1) -> None:
        """
        Unfreeze only the last Swin stages.
        """

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        feature_modules = list(self.backbone.features.children())
        num_to_unfreeze = min(num_stages * 2, len(feature_modules))

        for module in feature_modules[-num_to_unfreeze:]:
            for parameter in module.parameters():
                parameter.requires_grad = True

        for parameter in self.backbone.norm.parameters():
            parameter.requires_grad = True

        for parameter in self.backbone.head.parameters():
            parameter.requires_grad = True

    def trainable_parameter_names(self) -> Iterable[str]:
        return [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]

    def get_param_groups(
        self,
        head_lr: float,
        backbone_lr: float,
        weight_decay: float = 0.05,
    ):
        """
        Build optimizer parameter groups with discriminative learning rates.
        """

        head_decay = []
        head_no_decay = []
        backbone_decay = []
        backbone_no_decay = []

        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue

            is_head = name.startswith("backbone.head")
            no_decay = (
                parameter.ndim <= 1
                or "norm" in name
                or "bias" in name
            )

            if is_head:
                if no_decay:
                    head_no_decay.append(parameter)
                else:
                    head_decay.append(parameter)
            else:
                if no_decay:
                    backbone_no_decay.append(parameter)
                else:
                    backbone_decay.append(parameter)

        param_groups = []

        if head_decay:
            param_groups.append(
                {
                    "params": head_decay,
                    "lr": head_lr,
                    "weight_decay": weight_decay,
                }
            )

        if head_no_decay:
            param_groups.append(
                {
                    "params": head_no_decay,
                    "lr": head_lr,
                    "weight_decay": 0.0,
                }
            )

        if backbone_decay:
            param_groups.append(
                {
                    "params": backbone_decay,
                    "lr": backbone_lr,
                    "weight_decay": weight_decay,
                }
            )

        if backbone_no_decay:
            param_groups.append(
                {
                    "params": backbone_no_decay,
                    "lr": backbone_lr,
                    "weight_decay": 0.0,
                }
            )

        return param_groups

    @property
    def feature_dimension(self) -> int:
        return self.in_features

    def count_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def summary(self) -> None:
        print("Model            : Swin Transformer Tiny")
        print(f"Output Classes   : {self.backbone.head[-1].out_features}")
        print(f"Feature Dimension: {self.feature_dimension}")
        print(f"Trainable Params : {self.count_parameters():,}")


def get_swin_t(
    num_classes: int = 8,
    pretrained: bool = True,
    head_dropout: float = 0.2,
) -> SwinTransformer:
    """
    Returns a Swin Transformer Tiny model.
    """
    return SwinTransformer(
        num_classes=num_classes,
        pretrained=pretrained,
        head_dropout=head_dropout,
    )