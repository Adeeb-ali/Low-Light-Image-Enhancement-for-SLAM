import torch.nn as nn


class IlluminationEnhancer(nn.Module):

    def __init__(

        self,
        channels

    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(

                channels,
                channels,
                kernel_size=3,
                padding=1

            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(

                channels,
                channels,
                kernel_size=3,
                padding=1

            ),

            nn.Sigmoid()

        )

    def forward(

        self,
        x

    ):

        illumination_map = self.block(x)

        enhanced = x * illumination_map

        return enhanced