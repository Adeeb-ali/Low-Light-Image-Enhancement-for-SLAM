import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(
        self,
        channels=64
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

        self.bn1 = nn.BatchNorm2d(
            channels
        )

        self.activation = nn.LeakyReLU(
            negative_slope=0.1,
            inplace=True
        )

        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

        self.bn2 = nn.BatchNorm2d(
            channels
        )

    def forward(self, x):

        identity = x

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.activation(out)

        out = self.conv2(out)

        out = self.bn2(out)

        out = out + identity

        return out