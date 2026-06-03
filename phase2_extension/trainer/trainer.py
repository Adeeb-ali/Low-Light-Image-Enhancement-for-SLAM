import torch

from tqdm import tqdm


class Trainer:

    def __init__(

        self,
        model,
        train_loader,
        optimizer,
        scheduler,
        loss_function,
        device

    ):

        self.model = model

        self.train_loader = train_loader

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.loss_function = loss_function

        self.device = device

    def train_one_epoch(

        self

    ):

        self.model.train()

        total_loss_sum = 0.0

        charbonnier_sum = 0.0

        ssim_sum = 0.0

        edge_sum = 0.0

        perceptual_sum = 0.0

        frequency_sum = 0.0

        brightness_sum = 0.0

        progress_bar = tqdm(

            self.train_loader,
            desc="Training"

        )

        for batch in progress_bar:

            noisy = batch["noisy"].to(

                self.device

            )

            clean = batch["clean"].to(

                self.device

            )

            self.optimizer.zero_grad()

            prediction = self.model(

                noisy

            )

            losses = self.loss_function(

                prediction,
                clean

            )

            total_loss = losses["total_loss"]

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(

                self.model.parameters(),
                max_norm=1.0

            )

            self.optimizer.step()

            total_loss_sum += losses["total_loss"].item()

            charbonnier_sum += losses["charbonnier_loss"].item()

            ssim_sum += losses["ssim_loss"].item()

            edge_sum += losses["edge_loss"].item()

            perceptual_sum += losses["perceptual_loss"].item()

            frequency_sum += losses["frequency_loss"].item()

            brightness_sum += losses["brightness_loss"].item()

            progress_bar.set_postfix(

                {

                    "Total": f"{losses['total_loss'].item():.4f}",

                    "Charbonnier": f"{losses['charbonnier_loss'].item():.4f}",

                    "SSIM": f"{losses['ssim_loss'].item():.4f}",

                    "Edge": f"{losses['edge_loss'].item():.4f}"

                }

            )

        num_batches = len(

            self.train_loader

        )

        return {

            "total_loss": total_loss_sum / num_batches,

            "charbonnier_loss": charbonnier_sum / num_batches,

            "ssim_loss": ssim_sum / num_batches,

            "edge_loss": edge_sum / num_batches,

            "perceptual_loss": perceptual_sum / num_batches,

            "frequency_loss": frequency_sum / num_batches,

            "brightness_loss": brightness_sum / num_batches

        }