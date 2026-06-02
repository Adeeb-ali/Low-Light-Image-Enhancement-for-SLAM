import torch

from .enhancement_net import EnhancementNet


model = EnhancementNet(
    channels=64,
    num_rrdb_blocks=4
)

x = torch.randn(
    1,
    3,
    256,
    256
)

y = model(x)

print("Input Shape  :", x.shape)
print("Output Shape :", y.shape)