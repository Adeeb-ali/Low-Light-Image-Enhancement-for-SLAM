import os

import torch

from torch.utils.data import DataLoader

from phase2_extension.data.sidd_dataset import SIDDDataset
from phase2_extension.models.enhancement_net import EnhancementNet
from phase2_extension.losses.total_loss import TotalLoss
from phase2_extension.trainer.trainer import Trainer
from phase2_extension.trainer.checkpoint import CheckpointManager


class Retrainer:

    def __init__(

        self,
        device

    ):

        self.device = device

    def retrain(

        self,
        architecture,
        epochs=200,
        resume=True

    ):

        dataset = SIDDDataset(

            root_dir="datasets/Data",

            patch_size=256,

            training=True

        )

        dataloader = DataLoader(

            dataset,

            batch_size=2,

            shuffle=True,

            num_workers=0

        )

        model = EnhancementNet(

            channels=architecture["channels"],

            num_rrdb_blocks=architecture[
                "num_rrdb_blocks"
            ]

        ).to(self.device)

        loss_function = TotalLoss()

        optimizer = torch.optim.Adam(

            model.parameters(),

            lr=1e-5

        )

        trainer = Trainer(

            model=model,

            dataloader=dataloader,

            loss_function=loss_function,

            optimizer=optimizer,

            device=self.device

        )

        checkpoint_dir = "outputs/checkpoints"

        os.makedirs(

            checkpoint_dir,

            exist_ok=True

        )

        checkpoint_manager = CheckpointManager(

            save_dir=checkpoint_dir

        )

        latest_checkpoint = os.path.join(

            checkpoint_dir,

            "latest_model.pth"

        )

        start_epoch = 0

        if resume and os.path.exists(

            latest_checkpoint

        ):

            print(

                "\n===================================="
            )

            print(

                "RESUMING TRAINING"
            )

            print(

                "====================================\n"
            )

            model, optimizer, start_epoch, _ = (

                checkpoint_manager.load_checkpoint(

                    model=model,

                    optimizer=optimizer,

                    checkpoint_path=latest_checkpoint

                )

            )

            start_epoch += 1

            print(

                f"Starting From Epoch : {start_epoch}\n"

            )

        total_params = sum(

            p.numel()

            for p in model.parameters()

        )

        trainable_params = sum(

            p.numel()

            for p in model.parameters()

            if p.requires_grad

        )

        print(

            "\n===================================="
        )

        print(

            "FINAL ARCHITECTURE"
        )

        print(

            "====================================\n"
        )

        print(

            f"Channels          : {architecture['channels']}"

        )

        print(

            f"RRDB Blocks       : {architecture['num_rrdb_blocks']}"

        )

        print(

            f"Total Parameters  : {total_params:,}"

        )

        print(

            f"Trainable Params  : {trainable_params:,}"

        )

        print(

            "\n====================================\n"
        )

        for epoch in range(

            start_epoch,

            epochs

        ):

            print(

                f"\nEpoch [{epoch + 1}/{epochs}]\n"

            )

            loss = trainer.train_one_epoch()

            checkpoint_manager.save_checkpoint(

                model=model,

                optimizer=optimizer,

                epoch=epoch,

                loss=loss,

                channels=architecture["channels"],

                num_rrdb_blocks=architecture[
                    "num_rrdb_blocks"
                ],

                filename="latest_model.pth"

            )

            print(

                f"\nAverage Loss : {loss:.6f}"

            )

            print(

                "Checkpoint Saved\n"

            )

        print(

            "\n===================================="
        )

        print(

            "RETRAINING COMPLETED"
        )

        print(

            "====================================\n"
        )

        return model