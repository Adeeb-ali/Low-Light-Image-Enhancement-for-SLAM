import random
from PIL import Image


def apply_augmentations(noisy_img, clean_img):

    if random.random() > 0.5:

        noisy_img = noisy_img.transpose(
            Image.FLIP_LEFT_RIGHT
        )

        clean_img = clean_img.transpose(
            Image.FLIP_LEFT_RIGHT
        )

    if random.random() > 0.5:

        noisy_img = noisy_img.transpose(
            Image.FLIP_TOP_BOTTOM
        )

        clean_img = clean_img.transpose(
            Image.FLIP_TOP_BOTTOM
        )

    if random.random() > 0.5:

        noisy_img = noisy_img.rotate(90)
        clean_img = clean_img.rotate(90)

    return noisy_img, clean_img