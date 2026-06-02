import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(
        self,
        prediction,
        target
    ):

        mse = F.mse_loss(
            prediction,
            target
        )

        ssim_approximation = 1.0 / (1.0 + mse)

        loss = 1.0 - ssim_approximation

        return loss