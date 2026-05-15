import os

import cv2

import torch

import numpy as np

from PIL import Image

from torchvision import transforms

from skimage.metrics import peak_signal_noise_ratio

from skimage.metrics import structural_similarity

from phase2_extension.models.enhancement_net import EnhancementNet


device = torch.device("cpu")


CHANNELS = 48

NUM_RRDB_BLOCKS = 8


MODEL_PATH = (

    "outputs/checkpoints/latest_model.pth"

)


DATASET_TYPE = "salt_pepper"

# "salt_pepper"
# "sidd"


if DATASET_TYPE == "salt_pepper":

    TEST_DATASET_PATH = (

        "test_images/salt_pepper"

    )

else:

    TEST_DATASET_PATH = (

        "test_images/salt_pepper"

    )


OUTPUT_DIR = (

    "testing_model/results"

)

os.makedirs(

    OUTPUT_DIR,

    exist_ok=True

)


model = EnhancementNet(

    channels=CHANNELS,

    num_rrdb_blocks=NUM_RRDB_BLOCKS

).to(device)

checkpoint = torch.load(

    MODEL_PATH,

    map_location=device

)

model.load_state_dict(

    checkpoint["model_state_dict"]

)

model.eval()

print(

    "\nModel Loaded Successfully\n"

)


transform = transforms.Compose([

    transforms.ToTensor()

])


total_noisy_psnr = 0.0

total_denoised_psnr = 0.0

total_ssim = 0.0

image_count = 0


# ============================================================
# SALT PEPPER DATASET
# ============================================================

if DATASET_TYPE == "salt_pepper":

    noisy_dir = os.path.join(

        TEST_DATASET_PATH,

        "Noisy_folder"

    )

    gt_dir = os.path.join(

        TEST_DATASET_PATH,

        "Ground_truth"

    )

    noisy_images = sorted(

        os.listdir(noisy_dir)

    )

    print(

        f"Testing On {len(noisy_images)} Images\n"

    )

    with torch.inference_mode():

        for noisy_name in noisy_images:

            gt_name = noisy_name.replace(

                "noisy_",

                ""

            )

            noisy_path = os.path.join(

                noisy_dir,

                noisy_name

            )

            gt_path = os.path.join(

                gt_dir,

                gt_name

            )

            if not os.path.exists(gt_path):

                continue

            print(

                f"Processing : {noisy_name}"

            )

            noisy_image = Image.open(

                noisy_path

            ).convert("RGB")

            gt_image = Image.open(

                gt_path

            ).convert("RGB")

            noisy_image = noisy_image.resize(

                (512, 512)

            )

            gt_image = gt_image.resize(

                (512, 512)

            )

            noisy_tensor = transform(

                noisy_image

            ).unsqueeze(0).to(device)

            output_tensor = model(

                noisy_tensor

            )

            output_tensor = torch.clamp(

                output_tensor,

                0.0,

                1.0

            )

            noisy_np = np.array(

                noisy_image

            ).astype(np.float32) / 255.0

            gt_np = np.array(

                gt_image

            ).astype(np.float32) / 255.0

            output_np = output_tensor.squeeze(

                0

            ).permute(

                1,
                2,
                0

            ).cpu().numpy()

            noisy_psnr = peak_signal_noise_ratio(

                gt_np,

                noisy_np,

                data_range=1.0

            )

            denoised_psnr = peak_signal_noise_ratio(

                gt_np,

                output_np,

                data_range=1.0

            )

            ssim_value = structural_similarity(

                gt_np,

                output_np,

                channel_axis=2,

                data_range=1.0

            )

            psnr_gain = (

                denoised_psnr - noisy_psnr

            )

            print(

                f"Noisy PSNR      : {noisy_psnr:.4f}"

            )

            print(

                f"Denoised PSNR   : {denoised_psnr:.4f}"

            )

            print(

                f"PSNR Gain       : {psnr_gain:.4f}"

            )

            print(

                f"SSIM            : {ssim_value:.4f}\n"

            )

            gt_save = (

                gt_np * 255.0

            ).astype(np.uint8)

            noisy_save = (

                noisy_np * 255.0

            ).astype(np.uint8)

            output_save = (

                output_np * 255.0

            ).astype(np.uint8)

            comparison_image = np.concatenate(

                [

                    gt_save,

                    noisy_save,

                    output_save

                ],

                axis=1

            )

            comparison_image = cv2.cvtColor(

                comparison_image,

                cv2.COLOR_RGB2BGR

            )

            save_path = os.path.join(

                OUTPUT_DIR,

                f"{gt_name}"

            )

            cv2.imwrite(

                save_path,

                comparison_image

            )

            total_noisy_psnr += noisy_psnr

            total_denoised_psnr += denoised_psnr

            total_ssim += ssim_value

            image_count += 1


# ============================================================
# SIDD DATASET
# ============================================================

else:

    folders = sorted(

        os.listdir(

            TEST_DATASET_PATH

        )

    )

    print(

        f"Testing On {len(folders)} Images\n"

    )

    with torch.inference_mode():

        for folder_name in folders:

            folder_path = os.path.join(

                TEST_DATASET_PATH,

                folder_name

            )

            if not os.path.isdir(

                folder_path

            ):

                continue

            noisy_path = os.path.join(

                folder_path,

                "NOISY_SRGB_010.PNG"

            )

            gt_path = os.path.join(

                folder_path,

                "GT_SRGB_010.PNG"

            )

            if (

                not os.path.exists(noisy_path)

                or

                not os.path.exists(gt_path)

            ):

                continue

            print(

                f"Processing : {folder_name}"

            )

            noisy_image = Image.open(

                noisy_path

            ).convert("RGB")

            gt_image = Image.open(

                gt_path

            ).convert("RGB")

            noisy_image = noisy_image.resize(

                (512, 512)

            )

            gt_image = gt_image.resize(

                (512, 512)

            )

            noisy_tensor = transform(

                noisy_image

            ).unsqueeze(0).to(device)

            output_tensor = model(

                noisy_tensor

            )

            output_tensor = torch.clamp(

                output_tensor,

                0.0,

                1.0

            )

            noisy_np = np.array(

                noisy_image

            ).astype(np.float32) / 255.0

            gt_np = np.array(

                gt_image

            ).astype(np.float32) / 255.0

            output_np = output_tensor.squeeze(

                0

            ).permute(

                1,
                2,
                0

            ).cpu().numpy()

            noisy_psnr = peak_signal_noise_ratio(

                gt_np,

                noisy_np,

                data_range=1.0

            )

            denoised_psnr = peak_signal_noise_ratio(

                gt_np,

                output_np,

                data_range=1.0

            )

            ssim_value = structural_similarity(

                gt_np,

                output_np,

                channel_axis=2,

                data_range=1.0

            )

            psnr_gain = (

                denoised_psnr - noisy_psnr

            )

            print(

                f"Noisy PSNR      : {noisy_psnr:.4f}"

            )

            print(

                f"Denoised PSNR   : {denoised_psnr:.4f}"

            )

            print(

                f"PSNR Gain       : {psnr_gain:.4f}"

            )

            print(

                f"SSIM            : {ssim_value:.4f}\n"

            )

            gt_save = (

                gt_np * 255.0

            ).astype(np.uint8)

            noisy_save = (

                noisy_np * 255.0

            ).astype(np.uint8)

            output_save = (

                output_np * 255.0

            ).astype(np.uint8)

            comparison_image = np.concatenate(

                [

                    gt_save,

                    noisy_save,

                    output_save

                ],

                axis=1

            )

            comparison_image = cv2.cvtColor(

                comparison_image,

                cv2.COLOR_RGB2BGR

            )

            save_path = os.path.join(

                OUTPUT_DIR,

                f"{folder_name}.png"

            )

            cv2.imwrite(

                save_path,

                comparison_image

            )

            total_noisy_psnr += noisy_psnr

            total_denoised_psnr += denoised_psnr

            total_ssim += ssim_value

            image_count += 1


average_noisy_psnr = (

    total_noisy_psnr / image_count

)

average_denoised_psnr = (

    total_denoised_psnr / image_count

)

average_ssim = (

    total_ssim / image_count

)

average_gain = (

    average_denoised_psnr

    -

    average_noisy_psnr

)

print(

    "=============================="

)

print(

    f"Average Noisy PSNR      : {average_noisy_psnr:.4f}"

)

print(

    f"Average Denoised PSNR   : {average_denoised_psnr:.4f}"

)

print(

    f"Average PSNR Gain       : {average_gain:.4f}"

)

print(

    f"Average SSIM            : {average_ssim:.4f}"

)

print(

    "=============================="

)

print(

    "\nTesting Completed Successfully\n"

)