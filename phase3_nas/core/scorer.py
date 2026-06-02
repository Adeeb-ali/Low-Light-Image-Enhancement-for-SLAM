# import gc

# import torch

# from torch.utils.data import DataLoader

# from phase2_extension.data.multi_dataset import MultiDataset

# from phase2_extension.models.enhancement_net import EnhancementNet

# from phase2_extension.losses.total_loss import (

#     ImprovedTotalLoss,
#     LossConfigs

# )

# from phase2_extension.trainer.trainer import Trainer


# class ArchitectureScorer:

#     def __init__(

#         self,
#         device

#     ):

#         self.device = device

#     def score(

#         self,
#         architecture

#     ):

#         torch.cuda.empty_cache()

#         gc.collect()

#         dataset = MultiDataset(

#             sidd_root="datasets/Data",

#             lol_root="datasets/lol_dataset",

#             patch_size=256,

#             training=True

#         )

#         train_loader = DataLoader(

#             dataset,

#             batch_size=2,

#             shuffle=True,

#             num_workers=0,

#             pin_memory=False

#         )

#         model = EnhancementNet(

#             channels=architecture["channels"],

#             num_rrdb_blocks=architecture[
#                 "num_rrdb_blocks"
#             ]

#         ).to(self.device)

#         loss_function = ImprovedTotalLoss(

#             use_perceptual=True,

#             loss_weights=LossConfigs.LOW_LIGHT_FOCUSED

#         )

#         optimizer = torch.optim.Adam(

#             model.parameters(),

#             lr=1e-4

#         )

#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

#             optimizer,

#             T_max=1,

#             eta_min=1e-6

#         )

#         trainer = Trainer(

#             model=model,

#             train_loader=train_loader,

#             optimizer=optimizer,

#             scheduler=scheduler,

#             loss_function=loss_function,

#             device=self.device

#         )

#         losses = trainer.train_one_epoch()

#         total_loss = losses["total_loss"]

#         score = 1.0 / (

#             total_loss + 1e-8

#         )

#         trainer = None

#         optimizer = None

#         scheduler = None

#         loss_function = None

#         model.cpu()

#         del model

#         del train_loader

#         del dataset

#         torch.cuda.empty_cache()

#         gc.collect()

#         return {

#             "architecture": architecture,

#             "loss": total_loss,

#             "charbonnier_loss": losses[
#                 "charbonnier_loss"
#             ],

#             "ssim_loss": losses[
#                 "ssim_loss"
#             ],

#             "edge_loss": losses[
#                 "edge_loss"
#             ],

#             "score": score

#         }

import gc
import torch
from torch.utils.data import DataLoader

from phase2_extension.data.multi_dataset import MultiDataset
from phase2_extension.models.enhancement_net import EnhancementNet
from phase2_extension.losses.total_loss import ImprovedTotalLoss, LossConfigs
from phase2_extension.trainer.trainer import Trainer


class ArchitectureScorer:
    def __init__(self, device):
        self.device = device

    def score(self, architecture):
        # Clear out any leftover GPU memory before building the next architecture
        torch.cuda.empty_cache()
        gc.collect()

        # 1. Initialize Dataset & Dataloader
        dataset = MultiDataset(
            sidd_root="datasets/Data",
            lol_root="datasets/lol_dataset",
            patch_size=256,
            training=True
        )
        
        train_loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

        # 2. Build the specific candidate network dynamically
        model = EnhancementNet(
            channels=architecture["channels"],
            num_rrdb_blocks=architecture["num_rrdb_blocks"]
        ).to(self.device)

        loss_function = ImprovedTotalLoss(
            use_perceptual=True,
            loss_weights=LossConfigs.LOW_LIGHT_FOCUSED
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4
        )

        # Updated T_max to 5 because we are now training for 5 epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=5,
            eta_min=1e-6
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function,
            device=self.device
        )

        print(f"\n>>> Evaluating candidate architecture: Channels={architecture['channels']}, Blocks={architecture['num_rrdb_blocks']}")
        
        # 3. Train for 5 epochs instead of 1 to allow metrics to stabilize
        num_eval_epochs = 1
        for epoch in range(1, num_eval_epochs + 1):
            losses = trainer.train_one_epoch()
            
            # Step the scheduler at the end of each epoch
            scheduler.step()
            
            print(f"    Epoch {epoch}/{num_eval_epochs} | Total Loss: {losses['total_loss']:.4f} | SSIM Loss: {losses['ssim_loss']:.4f}")

        # 4. Use the stabilized final epoch metrics for scoring
        total_loss = losses["total_loss"]
        score = 1.0 / (total_loss + 1e-8)

        # 5. Intensive cleanup to guarantee no memory leaks between NAS iterations
        trainer = None
        optimizer = None
        scheduler = None
        loss_function = None
        
        model.cpu()
        del model
        del train_loader
        del dataset
        
        torch.cuda.empty_cache()
        gc.collect()

        # Return comprehensive metrics dictionary back to nas_main.py
        return {
            "architecture": architecture,
            "loss": total_loss,
            "charbonnier_loss": losses["charbonnier_loss"],
            "ssim_loss": losses["ssim_loss"],
            "edge_loss": losses["edge_loss"],
            "score": score
        }