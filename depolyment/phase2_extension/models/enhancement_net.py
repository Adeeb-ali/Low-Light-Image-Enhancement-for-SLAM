import torch.nn as nn

from phase2_extension.models.backbone.denoise_backbone import DenoiseBackbone

from phase2_extension.models.modules.illumination_enhancer import IlluminationEnhancer


class EnhancementNet(nn.Module):

    def __init__(

        self,
        channels=64,
        num_rrdb_blocks=6

    ):

        super().__init__()

        self.backbone = DenoiseBackbone(
            channels=channels,
            num_rrdb_blocks=num_rrdb_blocks
        )

        self.illumination = IlluminationEnhancer(
            channels
        )

        self.reconstruction = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Conv2d(
                channels,
                3,
                kernel_size=3,
                padding=1
            )

        )

    def forward(

        self,
        x

    ):

        feat = self.backbone(x)

        feat = self.illumination(feat)

        out = self.reconstruction(feat)

        return out