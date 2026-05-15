import torch


class Trainer:

    def __init__(

        self,
        model,
        dataloader,
        loss_function,
        optimizer,
        device

    ):

        self.model = model

        self.dataloader = dataloader

        self.loss_function = loss_function

        self.optimizer = optimizer

        self.device = device

    def train_one_epoch(self):

        self.model.train()

        total_loss = 0.0

        total_batches = len(

            self.dataloader

        )

        for batch_index, batch in enumerate(

            self.dataloader

        ):

            noisy = batch["noisy"].to(

                self.device

            )

            clean = batch["clean"].to(

                self.device

            )

            # ============================================
            # ZERO GRADIENTS
            # ============================================

            self.optimizer.zero_grad()

            # ============================================
            # FORWARD PASS
            # ============================================

            output = self.model(

                noisy

            )

            # ============================================
            # LOSS CALCULATION
            # ============================================

            losses = self.loss_function(

                output,
                clean

            )

            total_loss_value = losses[

                "total_loss"

            ]

            # ============================================
            # BACKPROPAGATION
            # ============================================

            total_loss_value.backward()

            # ============================================
            # OPTIMIZER STEP
            # ============================================

            self.optimizer.step()

            # ============================================
            # LOSS ACCUMULATION
            # ============================================

            total_loss += total_loss_value.item()

        # ================================================
        # FINAL AVERAGE LOSS
        # ================================================

        average_loss = (

            total_loss / total_batches

        )

        return average_loss