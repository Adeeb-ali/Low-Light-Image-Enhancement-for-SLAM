import torch
import torch.nn as nn


class SpatialAttention(nn.Module):

    def __init__(
        self,
        kernel_size=7
    ):

        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(

            1,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False

        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x
    ):

        avg_out = torch.mean(

            x,
            dim=1,
            keepdim=True

        )

        attention = self.sigmoid(

            self.conv(avg_out)

        )

        return x * attention