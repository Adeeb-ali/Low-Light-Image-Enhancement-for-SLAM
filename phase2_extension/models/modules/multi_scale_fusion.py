import torch
import torch.nn as nn


class MultiScaleFusion(nn.Module):

    def __init__(

        self,
        channels

    ):

        super().__init__()

        self.conv3 = nn.Conv2d(

            channels,
            channels,
            kernel_size=3,
            padding=1

        )

        self.conv5 = nn.Conv2d(

            channels,
            channels,
            kernel_size=5,
            padding=2

        )

        self.conv7 = nn.Conv2d(

            channels,
            channels,
            kernel_size=7,
            padding=3

        )

        self.fusion = nn.Conv2d(

            channels * 3,
            channels,
            kernel_size=1

        )

        self.relu = nn.ReLU(inplace=True)

    def forward(

        self,
        x

    ):

        feat3 = self.relu(

            self.conv3(x)

        )

        feat5 = self.relu(

            self.conv5(x)

        )

        feat7 = self.relu(

            self.conv7(x)

        )

        concat = torch.cat(

            [feat3, feat5, feat7],
            dim=1

        )

        output = self.fusion(concat)

        return output