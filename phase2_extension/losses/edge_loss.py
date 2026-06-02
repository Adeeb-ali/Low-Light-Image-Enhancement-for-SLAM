import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):

    def __init__(self):

        super().__init__()

        # ============================================
        # SOBEL X KERNEL
        # ============================================

        sobel_x = torch.tensor(

            [

                [-1, 0, 1],

                [-2, 0, 2],

                [-1, 0, 1]

            ],

            dtype=torch.float32

        ).view(

            1,
            1,
            3,
            3

        )

        # ============================================
        # SOBEL Y KERNEL
        # ============================================

        sobel_y = torch.tensor(

            [

                [-1, -2, -1],

                [0, 0, 0],

                [1, 2, 1]

            ],

            dtype=torch.float32

        ).view(

            1,
            1,
            3,
            3

        )

        # ============================================
        # REGISTER BUFFERS
        # ============================================

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

        # ============================================
        # RGB TO GRAYSCALE
        # ============================================

        gray = (

            0.299 * image[:, 0:1]

            +

            0.587 * image[:, 1:2]

            +

            0.114 * image[:, 2:3]

        )

        # ============================================
        # MOVE SOBEL KERNELS TO SAME DEVICE
        # ============================================

        sobel_x = self.sobel_x.to(

            gray.device

        )

        sobel_y = self.sobel_y.to(

            gray.device

        )

        # ============================================
        # EDGE X
        # ============================================

        edge_x = F.conv2d(

            gray,

            sobel_x,

            padding=1

        )

        # ============================================
        # EDGE Y
        # ============================================

        edge_y = F.conv2d(

            gray,

            sobel_y,

            padding=1

        )

        # ============================================
        # EDGE MAGNITUDE
        # ============================================

        edges = torch.sqrt(

            edge_x ** 2

            +

            edge_y ** 2

            +

            1e-6

        )

        return edges

    def forward(

        self,
        prediction,
        target

    ):

        # ============================================
        # COMPUTE EDGES
        # ============================================

        pred_edges = self.compute_edges(

            prediction

        )

        target_edges = self.compute_edges(

            target

        )

        # ============================================
        # EDGE LOSS
        # ============================================

        loss = F.l1_loss(

            pred_edges,

            target_edges

        )

        return loss