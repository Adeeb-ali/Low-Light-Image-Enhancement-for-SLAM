import torch


def normalize_image(image_tensor):

    image_tensor = torch.clamp(
        image_tensor,
        0.0,
        1.0
    )

    return image_tensor