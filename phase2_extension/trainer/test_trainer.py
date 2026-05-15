import torch

from torch.utils.data import DataLoader

from data.sidd_dataset import SIDDDataset

from models.enhancement_net import EnhancementNet

from losses.total_loss import TotalLoss

from trainer.trainer import Trainer


device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)


dataset = SIDDDataset(

    root_dir="../datasets/Data",

    patch_size=256,

    training=True

)


dataloader = DataLoader(

    dataset,

    batch_size=2,

    shuffle=True

)


model = EnhancementNet(

    channels=64,

    num_rrdb_blocks=4

).to(device)


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

    device=device

)


loss = trainer.train_one_epoch()

print(f"Average Training Loss : {loss}")