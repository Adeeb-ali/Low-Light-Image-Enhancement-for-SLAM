
#its code given the results in comprision form 

# import os
# import cv2
# import torch
# import numpy as np

# from PIL import Image
# from torchvision import transforms

# from phase2_extension.models.enhancement_net import EnhancementNet


# # ============================================================
# # DEVICE
# # ============================================================

# device = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )

# print("\n====================================")
# print("STARTING REAL WORLD INFERENCE")
# print("====================================")
# print("DEVICE :", device)

# if torch.cuda.is_available():
#     print("GPU :", torch.cuda.get_device_name(0))


# # ============================================================
# # CONFIGURATION
# # ============================================================

# CHANNELS = 48
# NUM_RRDB_BLOCKS = 3

# MODEL_PATH = "outputs/checkpoints/best_model.pth"

# INPUT_DIR = "test_images/real_world/blur/frames"
# OUTPUT_DIR = "testing_model/results/real_world/blur"

# # ------------------------------------------------------------
# # Resize images before inference
# # ------------------------------------------------------------

# MAX_IMAGE_SIZE = 512
# # use 512 for faster testing
# # use 1024 for better quality

# os.makedirs(OUTPUT_DIR, exist_ok=True)


# # ============================================================
# # MODEL
# # ============================================================

# model = EnhancementNet(
#     channels=CHANNELS,
#     num_rrdb_blocks=NUM_RRDB_BLOCKS
# ).to(device)

# checkpoint = torch.load(
#     MODEL_PATH,
#     map_location=device
# )

# model.load_state_dict(
#     checkpoint["model_state_dict"],
#     strict=True
# )

# model.eval()

# print("\n====================================")
# print("MODEL LOADED SUCCESSFULLY")
# print("====================================")


# # ============================================================
# # TRANSFORM
# # ============================================================

# transform = transforms.Compose([
#     transforms.ToTensor()
# ])


# # ============================================================
# # IMAGE LIST
# # ============================================================

# image_list = sorted([
#     f for f in os.listdir(INPUT_DIR)
#     if f.lower().endswith(
#         (".png", ".jpg", ".jpeg")
#     )
# ])

# print(f"\nImages Found : {len(image_list)}")

# if len(image_list) == 0:
#     raise RuntimeError(
#         "No images found in real_world folder."
#     )


# # ============================================================
# # INFERENCE LOOP
# # ============================================================

# success_count = 0
# fail_count = 0

# for image_name in image_list:

#     input_path = os.path.join(
#         INPUT_DIR,
#         image_name
#     )

#     try:

#         # ----------------------------------------------------
#         # Load Image
#         # ----------------------------------------------------

#         input_image = Image.open(
#             input_path
#         ).convert("RGB")

#         original_w, original_h = input_image.size

#         # ----------------------------------------------------
#         # Resize Large Images
#         # ----------------------------------------------------

#         if max(original_w, original_h) > MAX_IMAGE_SIZE:

#             scale = (
#                 MAX_IMAGE_SIZE /
#                 max(original_w, original_h)
#             )

#             new_w = int(original_w * scale)
#             new_h = int(original_h * scale)

#             print(
#                 f"\nResizing : {image_name}"
#             )
#             print(
#                 f"{original_w}x{original_h}"
#                 f" -> "
#                 f"{new_w}x{new_h}"
#             )

#             input_image = input_image.resize(
#                 (new_w, new_h),
#                 Image.BICUBIC
#             )

#         print(
#             f"Inference Size : "
#             f"{input_image.size[0]}x"
#             f"{input_image.size[1]}"
#         )

#         # ----------------------------------------------------
#         # Tensor
#         # ----------------------------------------------------

#         input_tensor = (
#             transform(input_image)
#             .unsqueeze(0)
#             .to(device)
#         )

#         # ----------------------------------------------------
#         # Inference
#         # ----------------------------------------------------

#         with torch.inference_mode():
#             output_tensor = model(input_tensor)

#         output_tensor = torch.clamp(
#             output_tensor,
#             0.0,
#             1.0
#         )

#         # ----------------------------------------------------
#         # Convert Output
#         # ----------------------------------------------------

#         output_np = (
#             output_tensor
#             .squeeze(0)
#             .permute(1, 2, 0)
#             .cpu()
#             .numpy()
#         )

#         output_np = np.clip(
#             output_np,
#             0.0,
#             1.0
#         )

#         # ----------------------------------------------------
#         # Save Images
#         # ----------------------------------------------------

#         input_np = np.array(input_image)

#         output_save = (
#             output_np * 255
#         ).astype(np.uint8)

#         input_bgr = cv2.cvtColor(
#             input_np,
#             cv2.COLOR_RGB2BGR
#         )

#         output_bgr = cv2.cvtColor(
#             output_save,
#             cv2.COLOR_RGB2BGR
#         )

#         comparison = np.concatenate(
#             [
#                 input_bgr,
#                 output_bgr
#             ],
#             axis=1
#         )

#         save_path = os.path.join(
#             OUTPUT_DIR,
#             image_name
#         )

#         cv2.imwrite(
#             save_path,
#             comparison
#         )

#         success_count += 1

#         print(
#             f"[{success_count}] Done : "
#             f"{image_name}"
#         )

#     except Exception as e:

#         fail_count += 1

#         print(
#             f"Failed : {image_name}"
#         )

#         print(e)


# # ============================================================
# # SUMMARY
# # ============================================================

# print("\n====================================")
# print("REAL WORLD INFERENCE COMPLETE")
# print("====================================")

# print(f"Processed : {success_count}")
# print(f"Failed    : {fail_count}")
# print(f"Saved To  : {OUTPUT_DIR}")

# print(
#     "\nNo metrics calculated "
#     "(No Ground Truth)\n"
# )

import os
import cv2
import torch
import numpy as np

from PIL import Image
from torchvision import transforms

from phase2_extension.models.enhancement_net import EnhancementNet


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n====================================")
print("STARTING REAL WORLD INFERENCE")
print("====================================")
print("DEVICE :", device)

if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))


# ============================================================
# CONFIGURATION
# ============================================================

CHANNELS = 48
NUM_RRDB_BLOCKS = 3

MODEL_PATH = "outputs/checkpoints/best_model.pth"

INPUT_DIR = "test_images/real_world/blur/frames"
OUTPUT_DIR = "testing_model/results/real_world/blur"

MAX_IMAGE_SIZE = 512

# FORCE OUTPUT SIZE
TARGET_W = 752
TARGET_H = 480

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# MODEL
# ============================================================

model = EnhancementNet(
    channels=CHANNELS,
    num_rrdb_blocks=NUM_RRDB_BLOCKS
).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)

model.load_state_dict(checkpoint["model_state_dict"], strict=True)

model.eval()

print("\n====================================")
print("MODEL LOADED SUCCESSFULLY")
print("====================================")


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor()
])


# ============================================================
# IMAGE LIST
# ============================================================

image_list = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

print(f"\nImages Found : {len(image_list)}")

if len(image_list) == 0:
    raise RuntimeError("No images found in input folder.")


# ============================================================
# INFERENCE LOOP
# ============================================================

success_count = 0
fail_count = 0

for image_name in image_list:

    input_path = os.path.join(INPUT_DIR, image_name)

    try:

        # Load image
        input_image = Image.open(input_path).convert("RGB")

        original_w, original_h = input_image.size

        # Resize large images (input control)
        if max(original_w, original_h) > MAX_IMAGE_SIZE:

            scale = MAX_IMAGE_SIZE / max(original_w, original_h)

            new_w = int(original_w * scale)
            new_h = int(original_h * scale)

            print(f"\nResizing : {image_name}")
            print(f"{original_w}x{original_h} -> {new_w}x{new_h}")

            input_image = input_image.resize((new_w, new_h), Image.BICUBIC)

        print(f"Inference Size : {input_image.size[0]}x{input_image.size[1]}")

        # Tensor
        input_tensor = transform(input_image).unsqueeze(0).to(device)

        # Inference
        with torch.inference_mode():
            output_tensor = model(input_tensor)

        output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

        # Convert to numpy
        output_np = (
            output_tensor.squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        output_np = np.clip(output_np, 0.0, 1.0)

        # To uint8
        output_save = (output_np * 255).astype(np.uint8)

        # FORCE SIZE = 752x480
        output_resized = cv2.resize(
            output_save,
            (TARGET_W, TARGET_H),
            interpolation=cv2.INTER_CUBIC
        )

        # Save
        output_bgr = cv2.cvtColor(output_resized, cv2.COLOR_RGB2BGR)

        save_path = os.path.join(OUTPUT_DIR, image_name)

        cv2.imwrite(save_path, output_bgr)

        success_count += 1

        print(f"[{success_count}] Done : {image_name}")

    except Exception as e:
        fail_count += 1
        print(f"Failed : {image_name}")
        print(e)


# ============================================================
# SUMMARY
# ============================================================

print("\n====================================")
print("REAL WORLD INFERENCE COMPLETE")
print("====================================")

print(f"Processed : {success_count}")
print(f"Failed    : {fail_count}")
print(f"Saved To  : {OUTPUT_DIR}")