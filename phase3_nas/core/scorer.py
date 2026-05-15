import torch

from torch.utils.data import DataLoader

from phase2_extension.data.sidd_dataset import SIDDDataset

from phase2_extension.models.enhancement_net import EnhancementNet

from phase2_extension.losses.total_loss import TotalLoss

from phase2_extension.trainer.trainer import Trainer


class ArchitectureScorer:

    def __init__(

        self,

        device

    ):

        self.device = device

    def score(

        self,

        architecture

    ):

        dataset = SIDDDataset(

            root_dir="datasets/Data",

            patch_size=256,

            training=True

        )

        dataloader = DataLoader(

            dataset,

            batch_size=2,

            shuffle=True

        )

        model = EnhancementNet(

            channels=architecture["channels"],

            num_rrdb_blocks=architecture["num_rrdb_blocks"]

        ).to(self.device)

        loss_function = TotalLoss()

        optimizer = torch.optim.Adam(

            model.parameters(),

            lr=1e-4

        )

        trainer = Trainer(

            model=model,

            dataloader=dataloader,

            loss_function=loss_function,

            optimizer=optimizer,

            device=self.device

        )

        loss = trainer.train_one_epoch()

        score = 1.0 / loss

        return {

            "architecture": architecture,

            "loss": loss,

            "score": score

        }