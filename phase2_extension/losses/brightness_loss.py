import torch
import torch.nn as nn


class BrightnessLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.l1 = nn.L1Loss()

    def forward(

        self,
        prediction,
        target

    ):

        prediction_gray = torch.mean(

            prediction,

            dim=1,

            keepdim=True

        )

        target_gray = torch.mean(

            target,

            dim=1,

            keepdim=True

        )

        loss = self.l1(

            prediction_gray,

            target_gray

        )

        return loss