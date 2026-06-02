import random

from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance


def apply_augmentations(

    noisy_img,
    clean_img

):

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

    if random.random() > 0.5:

        radius = random.uniform(0.3, 1.5)

        noisy_img = noisy_img.filter(
            ImageFilter.GaussianBlur(radius)
        )

    if random.random() > 0.5:

        brightness_factor = random.uniform(
            0.6,
            1.4
        )

        noisy_img = ImageEnhance.Brightness(
            noisy_img
        ).enhance(
            brightness_factor
        )

    if random.random() > 0.5:

        contrast_factor = random.uniform(
            0.7,
            1.5
        )

        noisy_img = ImageEnhance.Contrast(
            noisy_img
        ).enhance(
            contrast_factor
        )

    return noisy_img, clean_img