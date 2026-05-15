import torch
import torch.nn as nn

from ..blocks.rrdb_block import RRDBBlock
from ..blocks.feature_fusion import FeatureFusion
from ..attention.channel_attention import ChannelAttention


class DenoiseBackbone(nn.Module):

    def __init__(
        self,
        channels=64,
        num_rrdb_blocks=4
    ):

        super().__init__()

        self.initial_conv = nn.Conv2d(
            in_channels=3,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

        self.rrdb_blocks = nn.Sequential(

            *[
                RRDBBlock(
                    channels=channels,
                    num_blocks=3
                )

                for _ in range(num_rrdb_blocks)
            ]

        )

        self.channel_attention = ChannelAttention(
            channels=channels,
            reduction=16
        )

        self.feature_fusion = FeatureFusion(
            channels=channels
        )

    def forward(self, x):

        shallow_features = self.initial_conv(x)

        deep_features = self.rrdb_blocks(
            shallow_features
        )

        attention_features = self.channel_attention(
            deep_features
        )

        fused_features = self.feature_fusion(
            attention_features
        )

        return fused_features
