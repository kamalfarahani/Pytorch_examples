import torch
import torch.nn as nn


class TinyVGG(nn.Module):
    """
    TinyVGG model.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int
    ) -> None:
        super().__init__()
        
        # First convolutional block
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=16,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Second convolutional block
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=32 * 13 * 13,
                out_features=num_classes
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies forward propagation through the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after forward propagation.
        """
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return self.classifier(x)