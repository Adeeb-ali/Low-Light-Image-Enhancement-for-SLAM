import torch

from .total_loss import TotalLoss


loss_function = TotalLoss()

prediction = torch.rand(
    2,
    3,
    256,
    256
)

target = torch.rand(
    2,
    3,
    256,
    256
)

losses = loss_function(
    prediction,
    target
)

print("Total Loss        :", losses["total_loss"].item())
print("Charbonnier Loss  :", losses["charbonnier_loss"].item())
print("SSIM Loss         :", losses["ssim_loss"].item())
print("Edge Loss         :", losses["edge_loss"].item())