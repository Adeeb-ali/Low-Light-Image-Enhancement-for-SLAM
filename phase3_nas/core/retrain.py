# # phase3_nas/core/retrain.py

# import os

# import torch

# import torch.optim as optim

# from torch.optim.lr_scheduler import CosineAnnealingLR

# from torch.utils.data import DataLoader

# from phase2_extension.data.multi_dataset import MultiDataset

# from phase2_extension.models.enhancement_net import EnhancementNet

# from phase2_extension.losses.total_loss import (

#     ImprovedTotalLoss,
#     LossConfigs

# )

# from phase2_extension.trainer.trainer import Trainer


# class Retrainer:

#     def __init__(

#         self,
#         device

#     ):

#         self.device = device

#     def retrain(

#         self,
#         architecture,
#         epochs=200,
#         learning_rate=1e-4

#     ):

#         checkpoint_dir = "outputs/checkpoints"

#         os.makedirs(

#             checkpoint_dir,

#             exist_ok=True

#         )

#         checkpoint_path = os.path.join(

#             checkpoint_dir,

#             "best_model.pth"

#         )

#         dataset = MultiDataset(

#             sidd_root="datasets/Data",

#             lol_root="datasets/lol_dataset",

#             patch_size=256,

#             training=True

#         )

#         train_loader = DataLoader(

#             dataset,

#             batch_size=4,

#             shuffle=True,

#             num_workers=0,

#             pin_memory=False

#         )

#         print("\n====================================")
#         print(f"Total Training Samples : {len(dataset)}")
#         print("====================================\n")

#         model = EnhancementNet(

#             channels=architecture["channels"],

#             num_rrdb_blocks=architecture[
#                 "num_rrdb_blocks"
#             ]

#         ).to(

#             self.device

#         )

#         total_params = sum(

#             p.numel()

#             for p in model.parameters()

#         )

#         trainable_params = sum(

#             p.numel()

#             for p in model.parameters()

#             if p.requires_grad

#         )

#         print("\n====================================")
#         print("FINAL ARCHITECTURE")
#         print("====================================\n")

#         print(

#             f"Channels          : {architecture['channels']}"

#         )

#         print(

#             f"RRDB Blocks       : {architecture['num_rrdb_blocks']}"

#         )

#         print(

#             f"Total Parameters  : {total_params:,}"

#         )

#         print(

#             f"Trainable Params  : {trainable_params:,}"

#         )

#         print("\n====================================\n")

#         optimizer = optim.AdamW(

#             model.parameters(),

#             lr=learning_rate,

#             weight_decay=1e-4

#         )

#         scheduler = CosineAnnealingLR(

#             optimizer,

#             T_max=epochs,

#             eta_min=1e-6

#         )

#         loss_function = ImprovedTotalLoss(

#             use_perceptual=True,

#             loss_weights=LossConfigs.LOW_LIGHT_FOCUSED

#         )

#         trainer = Trainer(

#             model=model,

#             train_loader=train_loader,

#             optimizer=optimizer,

#             scheduler=scheduler,

#             loss_function=loss_function,

#             device=self.device

#         )

#         best_loss = float("inf")

#         start_epoch = 0

#         if os.path.exists(checkpoint_path):

#             print("\n====================================")
#             print("RESUMING BEST MODEL TRAINING")
#             print("====================================\n")

#             checkpoint = torch.load(

#                 checkpoint_path,

#                 map_location=self.device

#             )

#             model.load_state_dict(

#                 checkpoint["model_state_dict"]

#             )

#             optimizer.load_state_dict(

#                 checkpoint["optimizer_state_dict"]

#             )

#             scheduler.load_state_dict(

#                 checkpoint["scheduler_state_dict"]

#             )

#             start_epoch = checkpoint["epoch"] + 1

#             best_loss = checkpoint["loss"]

#             print("\n====================================")
#             print("CHECKPOINT LOADED")
#             print("====================================\n")

#             print(

#                 f"Epoch             : {start_epoch}"

#             )

#             print(

#                 f"Loss              : {best_loss:.6f}"

#             )

#             print(

#                 f"Channels          : {checkpoint['channels']}"

#             )

#             print(

#                 f"RRDB Blocks       : {checkpoint['num_rrdb_blocks']}"

#             )

#             print("\n====================================\n")

#         else:

#             print("\n====================================")
#             print("STARTING FRESH RETRAINING")
#             print("====================================\n")

#         for epoch in range(start_epoch, epochs):

#             print(

#                 f"\nEpoch [{epoch + 1}/{epochs}]\n"

#             )

#             losses = trainer.train_one_epoch()

#             average_loss = losses["total_loss"]

#             print(

#                 f"\nAverage Loss : {average_loss:.6f}"

#             )

#             print(

#                 f"Charbonnier  : {losses['charbonnier_loss']:.6f}"

#             )

#             print(

#                 f"SSIM Loss    : {losses['ssim_loss']:.6f}"

#             )

#             print(

#                 f"Edge Loss    : {losses['edge_loss']:.6f}"

#             )

#             print(

#                 f"Perceptual   : {losses['perceptual_loss']:.6f}"

#             )

#             print(

#                 f"Frequency    : {losses['frequency_loss']:.6f}"

#             )

#             print(

#                 f"Brightness   : {losses['brightness_loss']:.6f}"

#             )

#             if average_loss < best_loss:

#                 best_loss = average_loss

#                 torch.save(

#                     {

#                         "epoch": epoch,

#                         "model_state_dict": model.state_dict(),

#                         "optimizer_state_dict": optimizer.state_dict(),

#                         "scheduler_state_dict": scheduler.state_dict(),

#                         "loss": best_loss,

#                         "channels": architecture["channels"],

#                         "num_rrdb_blocks": architecture[
#                             "num_rrdb_blocks"
#                         ]

#                     },

#                     checkpoint_path

#                 )

#                 print(

#                     f"\nCheckpoint Saved : {checkpoint_path}"

#                 )

#                 print(

#                     f"\nNEW BEST MODEL SAVED (Loss: {best_loss:.6f})"

#                 )

#             else:

#                 print(

#                     "\nModel Not Improved"

#                 )

#             scheduler.step()

#         return model

# phase3_nas/core/retrain.py

import os

import torch

import torch.optim as optim

from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.data import DataLoader

from phase2_extension.data.multi_dataset import MultiDataset

from phase2_extension.models.enhancement_net import EnhancementNet

from phase2_extension.losses.total_loss import (

    ImprovedTotalLoss,
    LossConfigs

)

from phase2_extension.trainer.trainer import Trainer


class Retrainer:

    def __init__(

        self,
        device

    ):

        self.device = device

    def retrain(

        self,
        architecture,
        epochs=2,
        learning_rate=1e-4

    ):

        checkpoint_dir = "outputs/checkpoints"

        os.makedirs(

            checkpoint_dir,

            exist_ok=True

        )

        checkpoint_path = os.path.join(

            checkpoint_dir,

            "best_model.pth"

        )

        dataset = MultiDataset(

            sidd_root="datasets/Data",

            lol_root="datasets/lol_dataset",

            patch_size=256,

            training=True

        )

        train_loader = DataLoader(

            dataset,

            batch_size=4,

            shuffle=True,

            num_workers=0,

            pin_memory=False

        )

        print("\n====================================")
        print(f"Total Training Samples : {len(dataset)}")
        print("====================================\n")

        model = EnhancementNet(

            channels=architecture["channels"],

            num_rrdb_blocks=architecture[
                "num_rrdb_blocks"
            ]

        ).to(

            self.device

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

        print("\n====================================")
        print("FINAL ARCHITECTURE")
        print("====================================\n")

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

        print("\n====================================\n")

        optimizer = optim.AdamW(

            model.parameters(),

            lr=learning_rate,

            weight_decay=1e-4

        )

        scheduler = CosineAnnealingLR(

            optimizer,

            T_max=epochs,

            eta_min=1e-6

        )

        loss_function = ImprovedTotalLoss(

            use_perceptual=True,

            loss_weights=LossConfigs.LOW_LIGHT_FOCUSED

        )

        trainer = Trainer(

            model=model,

            train_loader=train_loader,

            optimizer=optimizer,

            scheduler=scheduler,

            loss_function=loss_function,

            device=self.device

        )

        best_loss = float("inf")

        start_epoch = 0

        if os.path.exists(checkpoint_path):

            print("\n====================================")
            print("RESUMING BEST MODEL TRAINING")
            print("====================================\n")

            checkpoint = torch.load(

                checkpoint_path,

                map_location=self.device

            )

            model.load_state_dict(

                checkpoint["model_state_dict"]

            )

            optimizer.load_state_dict(

                checkpoint["optimizer_state_dict"]

            )

            scheduler.load_state_dict(

                checkpoint["scheduler_state_dict"]

            )

            start_epoch = checkpoint["epoch"] + 1

            best_loss = checkpoint["loss"]

            print("\n====================================")
            print("CHECKPOINT LOADED")
            print("====================================\n")

            print(

                f"Epoch             : {start_epoch}"

            )

            print(

                f"Loss              : {best_loss:.6f}"

            )

            print(

                f"Channels          : {checkpoint['channels']}"

            )

            print(

                f"RRDB Blocks       : {checkpoint['num_rrdb_blocks']}"

            )

            print("\n====================================\n")

        else:

            print("\n====================================")
            print("STARTING FRESH RETRAINING")
            print("====================================\n")

        for epoch in range(start_epoch, epochs):

            print(

                f"\nEpoch [{epoch + 1}/{epochs}]\n"

            )

            losses = trainer.train_one_epoch()

            average_loss = losses["total_loss"]

            print(

                f"\nAverage Loss : {average_loss:.6f}"

            )

            print(

                f"Charbonnier  : {losses['charbonnier_loss']:.6f}"

            )

            print(

                f"SSIM Loss    : {losses['ssim_loss']:.6f}"

            )

            print(

                f"Edge Loss    : {losses['edge_loss']:.6f}"

            )

            print(

                f"Perceptual   : {losses['perceptual_loss']:.6f}"

            )

            print(

                f"Frequency    : {losses['frequency_loss']:.6f}"

            )

            print(

                f"Brightness   : {losses['brightness_loss']:.6f}"

            )

            if average_loss < best_loss:

                best_loss = average_loss

                torch.save(

                    {

                        "epoch": epoch,

                        "model_state_dict": model.state_dict(),

                        "optimizer_state_dict": optimizer.state_dict(),

                        "scheduler_state_dict": scheduler.state_dict(),

                        "loss": best_loss,

                        "channels": architecture["channels"],

                        "num_rrdb_blocks": architecture[
                            "num_rrdb_blocks"
                        ]

                    },

                    checkpoint_path

                )

                print(

                    f"\nCheckpoint Saved : {checkpoint_path}"

                )

                print(

                    f"\nNEW BEST MODEL SAVED (Loss: {best_loss:.6f})"

                )

            else:

                print(

                    "\nModel Not Improved"

                )

            scheduler.step()

        return model