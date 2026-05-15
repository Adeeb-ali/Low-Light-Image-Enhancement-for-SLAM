import torch
import torch.nn as nn

from .residual_block import ResidualBlock


class RRDBBlock(nn.Module):

    def __init__(
        self,
        channels=64,
        num_blocks=3
    ):

        super().__init__()

        self.blocks = nn.ModuleList(

            [
                ResidualBlock(channels)
                for _ in range(num_blocks)
            ]

        )

        self.scale = 0.2

    def forward(self, x):

        identity = x

        out = x

        for block in self.blocks:

            out = block(out)

        out = identity + self.scale * out

        return out