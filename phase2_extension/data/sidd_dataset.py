import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .augmentations import apply_augmentations
from .preprocessing import normalize_image


class SIDDDataset(Dataset):

    def __init__(
        self,
        root_dir,
        patch_size=256,
        training=True
    ):

        self.root_dir = root_dir

        self.patch_size = patch_size

        self.training = training

        self.scene_folders = sorted(

            [

                folder

                for folder in os.listdir(root_dir)

                if os.path.isdir(

                    os.path.join(
                        root_dir,
                        folder
                    )

                )

            ]

        )

        self.to_tensor = transforms.ToTensor()

    def __len__(self):

        return len(self.scene_folders)

    def random_crop(
        self,
        noisy_img,
        clean_img
    ):

        width, height = noisy_img.size

        crop_size = self.patch_size

        x = random.randint(
            0,
            width - crop_size
        )

        y = random.randint(
            0,
            height - crop_size
        )

        noisy_crop = noisy_img.crop(

            (
                x,
                y,
                x + crop_size,
                y + crop_size
            )

        )

        clean_crop = clean_img.crop(

            (
                x,
                y,
                x + crop_size,
                y + crop_size
            )

        )

        return noisy_crop, clean_crop

    def __getitem__(self, index):

        scene_name = self.scene_folders[index]

        scene_path = os.path.join(
            self.root_dir,
            scene_name
        )

        noisy_path = os.path.join(
            scene_path,
            "NOISY_SRGB_010.PNG"
        )

        clean_path = os.path.join(
            scene_path,
            "GT_SRGB_010.PNG"
        )

        noisy_img = Image.open(
            noisy_path
        ).convert("RGB")

        clean_img = Image.open(
            clean_path
        ).convert("RGB")

        if self.training:

            noisy_img, clean_img = self.random_crop(
                noisy_img,
                clean_img
            )

            noisy_img, clean_img = apply_augmentations(
                noisy_img,
                clean_img
            )

        noisy_tensor = self.to_tensor(
            noisy_img
        )

        clean_tensor = self.to_tensor(
            clean_img
        )

        noisy_tensor = normalize_image(
            noisy_tensor
        )

        clean_tensor = normalize_image(
            clean_tensor
        )

        return {

            "noisy": noisy_tensor,

            "clean": clean_tensor

        }