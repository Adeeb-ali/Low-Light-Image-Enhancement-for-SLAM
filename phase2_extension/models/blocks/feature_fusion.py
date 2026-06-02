import torch
import torch.nn as nn


class FeatureFusion(nn.Module):

    def __init__(
        self,
        channels=64
    ):

        super().__init__()

        self.fusion = nn.Sequential(

            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),

            nn.BatchNorm2d(
                channels
            ),

            nn.LeakyReLU(
                negative_slope=0.1,
                inplace=True
            )

        )

    def forward(self, x):

        out = self.fusion(x)

        return out