import torch
import torch.nn as nn


class FrequencyLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.l1 = nn.L1Loss()

    def forward(

        self,
        prediction,
        target

    ):

        pred_fft = torch.fft.fft2(

            prediction

        )

        target_fft = torch.fft.fft2(

            target

        )

        pred_mag = torch.abs(

            pred_fft

        )

        target_mag = torch.abs(

            target_fft

        )

        loss = self.l1(

            pred_mag,
            target_mag

        )

        return loss