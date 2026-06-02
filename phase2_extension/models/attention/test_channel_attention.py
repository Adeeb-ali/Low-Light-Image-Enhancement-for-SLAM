import torch

from channel_attention import ChannelAttention


model = ChannelAttention(
    channels=64,
    reduction=16
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
