import torch.nn as nn

from phase2_extension.models.blocks.rrdb_block import RRDB

from phase2_extension.models.modules.multi_scale_fusion import MultiScaleFusion


class DenoiseBackbone(nn.Module):

    def __init__(

        self,
        channels=64,
        num_rrdb_blocks=6

    ):

        super().__init__()

        self.initial = nn.Conv2d(
            3,
            channels,
            kernel_size=3,
            padding=1
        )

        self.rrdb_blocks = nn.Sequential(

            *[
                RRDB(channels)
                for _ in range(num_rrdb_blocks)
            ]

        )

        self.multi_scale = MultiScaleFusion(
            channels
        )

        self.final = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

    def forward(

        self,
        x

    ):

        feat = self.initial(x)

        out = self.rrdb_blocks(feat)

        out = self.multi_scale(out)

        out = self.final(out)

        return feat + out