import torch

from rrdb_block import RRDBBlock


model = RRDBBlock(
    channels=64,
    num_blocks=3
)

x = torch.randn(
    1,
    64,
    256,
    256
)

y = model(x)

print("Input Shape  :", x.shape)
print("Output Shape :", y.shape)