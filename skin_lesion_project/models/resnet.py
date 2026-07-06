"""LeNet-5 implementation for multi-class skin lesion classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """
    LeNet-5 architecture.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    in_channels : int
        Number of image channels (default=3 for RGB).
    input_size : int
        Height/Width of the input image.
    """

    def __init__(
        self,
        num_classes: int = 8,
        in_channels: int = 3,
        input_size: int = 224,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=in_channels,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Automatically determine flattened feature size
        flattened_size = self._get_flattened_size(
            in_channels,
            input_size,
        )

        self.classifier = nn.Sequential(

            nn.Linear(flattened_size, 120),
            nn.ReLU(inplace=True),

            nn.Linear(120, 84),
            nn.ReLU(inplace=True),

            nn.Linear(84, num_classes),
        )

        self._initialize_weights()

    def _get_flattened_size(
        self,
        in_channels: int,
        input_size: int,
    ) -> int:
        """
        Computes the flattened feature dimension automatically.
        """

        with torch.no_grad():

            x = torch.zeros(
                1,
                in_channels,
                input_size,
                input_size,
            )

            x = self.features(x)

            return x.view(1, -1).size(1)

    def _initialize_weights(self) -> None:
        """
        Initializes model weights using Kaiming initialization.
        """

        for module in self.modules():

            if isinstance(module, nn.Conv2d):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):

                nn.init.kaiming_normal_(
                    module.weight,
                    nonlinearity="relu",
                )

                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape
            (batch_size, channels, height, width)

        Returns
        -------
        torch.Tensor
            Raw logits.
        """

        x = self.features(x)

        x = torch.flatten(x, start_dim=1)

        x = self.classifier(x)

        return x


def lenet5(
    num_classes: int = 8,
    in_channels: int = 3,
    input_size: int = 224,
) -> LeNet5:
    """
    Factory function for LeNet-5.

    Parameters
    ----------
    num_classes : int
        Number of output classes.

    in_channels : int
        Number of input channels.

    input_size : int
        Input image size.

    Returns
    -------
    LeNet5
    """

    return LeNet5(
        num_classes=num_classes,
        in_channels=in_channels,
        input_size=input_size,
    )


if __name__ == "__main__":

    model = lenet5()

    x = torch.randn(2, 3, 224, 224)

    y = model(x)

    print(model)

    print(f"Output Shape : {y.shape}")
    
