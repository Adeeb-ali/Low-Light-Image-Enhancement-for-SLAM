import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):

    def __init__(self):

        super().__init__()

        sobel_x = torch.tensor(

            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ],

            dtype=torch.float32

        ).unsqueeze(0).unsqueeze(0)

        sobel_y = torch.tensor(

            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ],

            dtype=torch.float32

        ).unsqueeze(0).unsqueeze(0)

        self.register_buffer(
            "sobel_x",
            sobel_x
        )

        self.register_buffer(
            "sobel_y",
            sobel_y
        )

    def compute_edges(
        self,
        image
    ):

        gray = torch.mean(
            image,
            dim=1,
            keepdim=True
        )

        edge_x = F.conv2d(
            gray,
            self.sobel_x,
            padding=1
        )

        edge_y = F.conv2d(
            gray,
            self.sobel_y,
            padding=1
        )

        edges = torch.sqrt(
            edge_x * edge_x
            +
            edge_y * edge_y
            +
            1e-6
        )

        return edges

    def forward(
        self,
        prediction,
        target
    ):

        pred_edges = self.compute_edges(
            prediction
        )

        target_edges = self.compute_edges(
            target
        )

        loss = F.l1_loss(
            pred_edges,
            target_edges
        )

        return loss