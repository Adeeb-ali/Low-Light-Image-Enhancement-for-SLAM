import torch

from feature_fusion import FeatureFusion


model = FeatureFusion(
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