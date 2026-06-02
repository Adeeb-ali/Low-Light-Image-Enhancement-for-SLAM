import torch
import torch.nn as nn


class ChannelAttention(nn.Module):

    def __init__(
        self,
        channels,
        reduction=16
    ):

        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.shared_mlp = nn.Sequential(

            nn.Conv2d(
                channels,
                channels // reduction,
                kernel_size=1,
                bias=False
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels // reduction,
                channels,
                kernel_size=1,
                bias=False
            )

        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x
    ):

        avg_out = self.shared_mlp(
            self.avg_pool(x)
        )

        attention = self.sigmoid(
            avg_out
        )

        return x * attention