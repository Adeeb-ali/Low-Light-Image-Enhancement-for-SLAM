import torch
import torch.nn as nn

from phase2_extension.models.modules.cbam import CBAM


class DenseBlock(nn.Module):

    def __init__(

        self,
        channels,
        growth_channels=32

    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            growth_channels,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            channels + growth_channels,
            growth_channels,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            channels + growth_channels * 2,
            growth_channels,
            kernel_size=3,
            padding=1
        )

        self.conv4 = nn.Conv2d(
            channels + growth_channels * 3,
            growth_channels,
            kernel_size=3,
            padding=1
        )

        self.conv5 = nn.Conv2d(
            channels + growth_channels * 4,
            channels,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.LeakyReLU(
            0.2,
            inplace=True
        )

    def forward(

        self,
        x

    ):

        x1 = self.relu(
            self.conv1(x)
        )

        x2 = self.relu(
            self.conv2(
                torch.cat([x, x1], dim=1)
            )
        )

        x3 = self.relu(
            self.conv3(
                torch.cat([x, x1, x2], dim=1)
            )
        )

        x4 = self.relu(
            self.conv4(
                torch.cat([x, x1, x2, x3], dim=1)
            )
        )

        x5 = self.conv5(
            torch.cat([x, x1, x2, x3, x4], dim=1)
        )

        return x + x5 * 0.2


class RRDB(nn.Module):

    def __init__(

        self,
        channels

    ):

        super().__init__()

        self.db1 = DenseBlock(
            channels
        )

        self.db2 = DenseBlock(
            channels
        )

        self.db3 = DenseBlock(
            channels
        )

        self.attention = CBAM(
            channels
        )

    def forward(

        self,
        x

    ):

        out = self.db1(x)

        out = self.db2(out)

        out = self.db3(out)

        out = self.attention(out)

        return x + out * 0.2