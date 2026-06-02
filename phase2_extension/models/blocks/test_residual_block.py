import torch

from residual_block import ResidualBlock


model = ResidualBlock(
    channels=64
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