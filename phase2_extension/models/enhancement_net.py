import torch
import torch.nn as nn

from .backbone.denoise_backbone import DenoiseBackbone


class EnhancementNet(nn.Module):

    def __init__(
        self,
        channels=64,
        num_rrdb_blocks=4
    ):

        super().__init__()

        self.backbone = DenoiseBackbone(
            channels=channels,
            num_rrdb_blocks=num_rrdb_blocks
        )

        self.reconstruction = nn.Conv2d(
            in_channels=channels,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

    def forward(self, x):

        features = self.backbone(x)

        predicted_noise = self.reconstruction(
            features
        )

        clean_image = x - predicted_noise

        clean_image = torch.clamp(
            clean_image,
            0.0,
            1.0
        )

        return clean_image