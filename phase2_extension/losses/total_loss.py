import torch
import torch.nn as nn

from .charbonnier_loss import CharbonnierLoss
from .ssim_loss import SSIMLoss
from .edge_loss import EdgeLoss


class TotalLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.charbonnier_loss = CharbonnierLoss()

        self.ssim_loss = SSIMLoss()

        self.edge_loss = EdgeLoss()

    def forward(
        self,
        prediction,
        target
    ):

        charbonnier = self.charbonnier_loss(
            prediction,
            target
        )

        ssim = self.ssim_loss(
            prediction,
            target
        )

        edge = self.edge_loss(
            prediction,
            target
        )

        total = (
            charbonnier
            +
            0.1 * ssim
            +
            0.05 * edge
        )

        return {
            "total_loss": total,
            "charbonnier_loss": charbonnier,
            "ssim_loss": ssim,
            "edge_loss": edge
        }