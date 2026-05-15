import torch
import torch.nn as nn


class ChannelAttention(nn.Module):

    def __init__(
        self,
        channels=64,
        reduction=16
    ):

        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(
            channels,
            channels // reduction,
            kernel_size=1,
            bias=True
        )

        self.activation = nn.LeakyReLU(
            negative_slope=0.1,
            inplace=True
        )

        self.fc2 = nn.Conv2d(
            channels // reduction,
            channels,
            kernel_size=1,
            bias=True
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        attention = self.avg_pool(x)

        attention = self.fc1(attention)

        attention = self.activation(attention)

        attention = self.fc2(attention)

        attention = self.sigmoid(attention)

        out = x * attention

        return out
