import os
import cv2
import csv
import time
import torch
import argparse
import numpy as np

from PIL import Image
from torchvision import transforms

from phase2_extension.models.enhancement_net import EnhancementNet


# ============================================================
# CLI ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    required=True,
    help="Input image folder"
)

parser.add_argument(
    "--output",
    required=True,
    help="Output image folder"
)

parser.add_argument(
    "--model",
    default="best_model.pth",
    help="Model path"
)

parser.add_argument(
    "--width",
    default=752,
    type=int,
    help="Output width"
)

parser.add_argument(
    "--height",
    default=480,
    type=int,
    help="Output height"
)

parser.add_argument(
    "--max-size",
    default=None,
    type=int,
    help="Maximum image size limit. Leave empty to process any original size."
)

args = parser.parse_args()


# ============================================================
# CONFIGURATION
# ============================================================

CHANNELS = 48
NUM_RRDB_BLOCKS = 3

INPUT_DIR = args.input
OUTPUT_DIR = args.output

MODEL_PATH = args.model

TARGET_W = args.width
TARGET_H = args.height

MAX_IMAGE_SIZE = args.max_size

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    device = torch.device("cuda")

elif torch.backends.mps.is_available():

    device = torch.device("mps")

else:

    device = torch.device("cpu")

print("\n====================================")
print("ENHANCEMENTNET DEPLOYMENT")
print("====================================")
print("DEVICE :", device)
print("INPUT  :", INPUT_DIR)
print("OUTPUT :", OUTPUT_DIR)
print("MODEL  :", MODEL_PATH)

if torch.cuda.is_available():

    print(
        "GPU :",
        torch.cuda.get_device_name(0)
    )

print("====================================\n")


# ============================================================
# MODEL
# ============================================================

model = EnhancementNet(
    channels=CHANNELS,
    num_rrdb_blocks=NUM_RRDB_BLOCKS
).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

model.eval()

print("MODEL LOADED SUCCESSFULLY\n")


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
    if f.lower().endswith(
        (".png", ".jpg", ".jpeg")
    )
])

print(
    f"Images Found : {len(image_list)}"
)

if len(image_list) == 0:

    raise RuntimeError(
        "No images found in input folder."
    )


# ============================================================
# CSV METRICS
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "latency.csv"
)

csv_file = open(
    csv_path,
    "w",
    newline=""
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "image_name",
    "latency_ms"
])


# ============================================================
# INFERENCE LOOP
# ============================================================

success_count = 0
fail_count = 0

total_latency = 0.0
total_images = len(image_list)

for image_name in image_list:

    input_path = os.path.join(
        INPUT_DIR,
        image_name
    )

    try:
        input_image = Image.open(
            input_path
        ).convert("RGB")

        original_w, original_h = input_image.size
        
        # Calculate current dynamic item number out of total 
        current_index = success_count + fail_count + 1
        
        # Enhanced Progress Counter & Docker friendly stream flushing
        print(
            f"[{current_index}/{total_images}] Processing: {image_name} | Size: {original_w}x{original_h}",
            flush=True
        )
        
        print(
            f"   Using Device: {device}",
            flush=True
        )

        if MAX_IMAGE_SIZE is not None and max(original_w, original_h) > MAX_IMAGE_SIZE:

            scale = (
                MAX_IMAGE_SIZE /
                max(original_w, original_h)
            )

            new_w = int(
                original_w * scale
            )

            new_h = int(
                original_h * scale
            )

            print(
                f"   [Resize] Capping size down to: {new_w}x{new_h}",
                flush=True
            )
            
            input_image = input_image.resize(
                (new_w, new_h),
                Image.BICUBIC
            )

        input_tensor = transform(
            input_image
        ).unsqueeze(0).to(device)

        start_time = time.time()

        with torch.inference_mode():

            output_tensor = model(
                input_tensor
            )

        latency_ms = (
            time.time() - start_time
        ) * 1000.0

        total_latency += latency_ms

        csv_writer.writerow([
            image_name,
            f"{latency_ms:.3f}"
        ])

        output_tensor = torch.clamp(
            output_tensor,
            0.0,
            1.0
        )

        output_np = (
            output_tensor
            .squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        output_np = np.clip(
            output_np,
            0.0,
            1.0
        )

        output_save = (
            output_np * 255
        ).astype(np.uint8)

        output_resized = cv2.resize(
            output_save,
            (TARGET_W, TARGET_H),
            interpolation=cv2.INTER_CUBIC
        )

        output_bgr = cv2.cvtColor(
            output_resized,
            cv2.COLOR_RGB2BGR
        )

        save_path = os.path.join(
            OUTPUT_DIR,
            image_name
        )

        # FIX: Corrected image array passing to restore saving operations
        cv2.imwrite(
            save_path,
            output_bgr
        )

        success_count += 1

        print(
            f"   [Done] Finished in {latency_ms:.2f} ms\n",
            flush=True
        )

    except Exception as e:

        fail_count += 1

        print(
            f"   FAILED : {image_name}",
            flush=True
        )

        print(e, flush=True)


# ============================================================
# SUMMARY
# ============================================================

csv_file.close()

avg_latency = 0.0

if success_count > 0:

    avg_latency = (
        total_latency /
        success_count
    )

summary_path = os.path.join(
    OUTPUT_DIR,
    "summary.csv"
)

with open(
    summary_path,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "processed_images",
        "failed_images",
        "average_latency_ms"
    ])

    writer.writerow([
        success_count,
        fail_count,
        f"{avg_latency:.3f}"
    ])

print("\n====================================")
print("DEPLOYMENT COMPLETE")
print("====================================")
print("Processed :", success_count)
print("Failed    :", fail_count)
print("Avg Latency (ms) :", avg_latency)
print("Saved To  :", OUTPUT_DIR)
print("====================================\n", flush=True)