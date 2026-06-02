from torch.utils.data import DataLoader

from sidd_dataset import SIDDDataset


train_dataset = SIDDDataset(
    root_dir="../../datasets/Data",
    patch_size=256,
    training=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True
)


for batch in train_loader:

    noisy = batch["noisy"]
    clean = batch["clean"]

    print("Noisy Shape :", noisy.shape)
    print("Clean Shape :", clean.shape)

    break