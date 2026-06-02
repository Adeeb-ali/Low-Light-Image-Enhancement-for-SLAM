import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):

    def __init__(
        self,
        epsilon=1e-3
    ):

        super().__init__()

        self.epsilon = epsilon

    def forward(
        self,
        prediction,
        target
    ):

        difference = prediction - target

        loss = torch.mean(

            torch.sqrt(
                (difference * difference)
                + (self.epsilon * self.epsilon)
            )

        )

        return loss