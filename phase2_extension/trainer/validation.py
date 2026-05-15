import torch
import torch.nn.functional as F


class Validator:

    def __init__(
        self,
        model,
        dataloader,
        device
    ):

        self.model = model

        self.dataloader = dataloader

        self.device = device

    def compute_psnr(
        self,
        prediction,
        target
    ):

        mse = F.mse_loss(
            prediction,
            target
        )

        if mse == 0:
            return 100

        psnr = (
            20
            *
            torch.log10(
                1.0
                /
                torch.sqrt(mse)
            )
        )

        return psnr.item()

    def validate(self):

        self.model.eval()

        total_psnr = 0.0

        with torch.no_grad():

            for batch in self.dataloader:

                noisy = batch["noisy"].to(
                    self.device
                )

                clean = batch["clean"].to(
                    self.device
                )

                prediction = self.model(
                    noisy
                )

                psnr = self.compute_psnr(
                    prediction,
                    clean
                )

                total_psnr += psnr

        average_psnr = (
            total_psnr
            /
            len(self.dataloader)
        )

        return average_psnr