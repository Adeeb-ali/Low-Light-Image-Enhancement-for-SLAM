import torch
import torch.nn as nn

from phase2_extension.losses.charbonnier_loss import CharbonnierLoss
from phase2_extension.losses.ssim_loss import SSIMLoss
from phase2_extension.losses.edge_loss import EdgeLoss
from phase2_extension.losses.perceptual_loss import PerceptualLoss
from phase2_extension.losses.frequency_loss import FrequencyLoss
from phase2_extension.losses.brightness_loss import BrightnessLoss


class ImprovedTotalLoss(nn.Module):

    def __init__(

        self,
        use_perceptual=True,
        temperature=1.0,
        loss_weights=None

    ):

        super().__init__()

        self.temperature = temperature
        self.use_perceptual = use_perceptual

        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.edge = EdgeLoss()
        self.frequency = FrequencyLoss()
        self.brightness = BrightnessLoss()

        if self.use_perceptual:

            self.perceptual = PerceptualLoss()

        else:

            self.perceptual = None

        default_weights = {

            "charbonnier": 0.32,
            "ssim": 0.23,
            "edge": 0.22,
            "perceptual": 0.13,
            "frequency": 0.05,
            "brightness": 0.05

        }

        if loss_weights is not None:

            default_weights.update(

                loss_weights

            )

        total_weight = sum(

            default_weights.values()

        )

        self.weights = {

            k: v / total_weight

            for k, v in default_weights.items()

        }

    def forward(

        self,
        prediction,
        target

    ):

        charbonnier_loss = self.charbonnier(

            prediction,
            target

        )

        ssim_loss = self.ssim(

            prediction,
            target

        )

        edge_loss = self.edge(

            prediction,
            target

        )

        frequency_loss = self.frequency(

            prediction,
            target

        )

        brightness_loss = self.brightness(

            prediction,
            target

        )

        if (

            self.use_perceptual

            and

            self.perceptual is not None

        ):

            perceptual_loss = self.perceptual(

                prediction,
                target

            )

        else:

            perceptual_loss = torch.tensor(

                0.0,
                device=prediction.device

            )

        charbonnier_loss = (

            charbonnier_loss / self.temperature

        )

        ssim_loss = (

            ssim_loss / self.temperature

        )

        edge_loss = (

            edge_loss / self.temperature

        )

        frequency_loss = (

            frequency_loss / self.temperature

        )

        brightness_loss = (

            brightness_loss / self.temperature

        )

        perceptual_loss = (

            perceptual_loss / self.temperature

        )

        total_loss = (

            self.weights["charbonnier"] * charbonnier_loss +

            self.weights["ssim"] * ssim_loss +

            self.weights["edge"] * edge_loss +

            self.weights["perceptual"] * perceptual_loss +

            self.weights["frequency"] * frequency_loss +

            self.weights["brightness"] * brightness_loss

        )

        return {

            "total_loss": total_loss,

            "charbonnier_loss": charbonnier_loss.detach(),

            "ssim_loss": ssim_loss.detach(),

            "edge_loss": edge_loss.detach(),

            "perceptual_loss": perceptual_loss.detach(),

            "frequency_loss": frequency_loss.detach(),

            "brightness_loss": brightness_loss.detach()

        }


class LossConfigs:

    BALANCED = {

        "charbonnier": 0.32,

        "ssim": 0.23,

        "edge": 0.22,

        "perceptual": 0.13,

        "frequency": 0.05,

        "brightness": 0.05

    }

    DETAIL_FOCUSED = {

        "charbonnier": 0.30,

        "ssim": 0.22,

        "edge": 0.25,

        "perceptual": 0.13,

        "frequency": 0.07,

        "brightness": 0.03

    }

    LOW_LIGHT_FOCUSED = {

        "charbonnier": 0.28,

        "ssim": 0.24,

        "edge": 0.20,

        "perceptual": 0.15,

        "frequency": 0.05,

        "brightness": 0.08

    }

    PERCEPTUAL_FOCUSED = {

        "charbonnier": 0.25,

        "ssim": 0.20,

        "edge": 0.18,

        "perceptual": 0.25,

        "frequency": 0.07,

        "brightness": 0.05

    }