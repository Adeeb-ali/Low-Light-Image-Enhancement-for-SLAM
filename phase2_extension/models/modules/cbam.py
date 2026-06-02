import torch.nn as nn

from phase2_extension.models.modules.channel_attention import ChannelAttention

from phase2_extension.models.modules.spatial_attention import SpatialAttention


class CBAM(nn.Module):

    def __init__(

        self,
        channels

    ):

        super().__init__()

        self.channel_attention = ChannelAttention(

            channels

        )

        self.spatial_attention = SpatialAttention()

    def forward(

        self,
        x

    ):

        x = self.channel_attention(x)

        x = self.spatial_attention(x)

        return x