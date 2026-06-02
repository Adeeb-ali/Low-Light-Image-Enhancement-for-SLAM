import os

import random

from PIL import Image

from torch.utils.data import Dataset

from torchvision import transforms

from phase2_extension.data.augmentations import apply_augmentations


class MultiDataset(Dataset):

    def __init__(

        self,
        sidd_root,
        lol_root,
        patch_size=256,
        training=True

    ):

        self.patch_size = patch_size

        self.training = training

        self.transform = transforms.ToTensor()

        self.samples = []

        # =====================================================
        # SIDD DATASET
        # =====================================================

        sidd_folders = sorted(

            os.listdir(sidd_root)

        )

        for folder_name in sidd_folders:

            folder_path = os.path.join(

                sidd_root,
                folder_name

            )

            if not os.path.isdir(folder_path):

                continue

            noisy_path = os.path.join(

                folder_path,
                "NOISY_SRGB_010.PNG"

            )

            clean_path = os.path.join(

                folder_path,
                "GT_SRGB_010.PNG"

            )

            if (

                os.path.exists(noisy_path)

                and

                os.path.exists(clean_path)

            ):

                self.samples.append(

                    {

                        "type": "sidd",

                        "noisy": noisy_path,

                        "clean": clean_path

                    }

                )

        # =====================================================
        # LOL DATASET
        # =====================================================

        lol_subfolders = [

            "our485"

        ]

        for subfolder in lol_subfolders:

            low_dir = os.path.join(

                lol_root,
                subfolder,
                "low"

            )

            high_dir = os.path.join(

                lol_root,
                subfolder,
                "high"

            )

            image_names = sorted(

                os.listdir(low_dir)

            )

            for image_name in image_names:

                low_path = os.path.join(

                    low_dir,
                    image_name

                )

                high_path = os.path.join(

                    high_dir,
                    image_name

                )

                if (

                    os.path.exists(low_path)

                    and

                    os.path.exists(high_path)

                ):

                    self.samples.append(

                        {

                            "type": "lol",

                            "noisy": low_path,

                            "clean": high_path

                        }

                    )

        print(

            f"\nTotal Training Samples : {len(self.samples)}\n"

        )

    def __len__(self):

        return len(self.samples)

    def random_crop(

        self,
        noisy,
        clean

    ):

        width, height = noisy.size

        crop_size = self.patch_size

        if (

            width < crop_size

            or

            height < crop_size

        ):

            noisy = noisy.resize(

                (crop_size, crop_size)

            )

            clean = clean.resize(

                (crop_size, crop_size)

            )

            return noisy, clean

        x = random.randint(

            0,
            width - crop_size

        )

        y = random.randint(

            0,
            height - crop_size

        )

        noisy = noisy.crop(

            (

                x,
                y,
                x + crop_size,
                y + crop_size

            )

        )

        clean = clean.crop(

            (

                x,
                y,
                x + crop_size,
                y + crop_size

            )

        )

        return noisy, clean

    def __getitem__(

        self,
        index

    ):

        sample = self.samples[index]

        noisy = Image.open(

            sample["noisy"]

        ).convert("RGB")

        clean = Image.open(

            sample["clean"]

        ).convert("RGB")

        if self.training:

            noisy, clean = self.random_crop(

                noisy,
                clean

            )

            noisy, clean = apply_augmentations(

                noisy,
                clean

            )

        noisy_tensor = self.transform(

            noisy

        )

        clean_tensor = self.transform(

            clean

        )

        return {

            "noisy": noisy_tensor,

            "clean": clean_tensor

        }