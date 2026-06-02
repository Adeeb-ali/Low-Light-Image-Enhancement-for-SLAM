import os
import cv2
import time
import torch
import numpy as np
import csv

from PIL import Image
from torchvision import transforms

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from phase2_extension.models.enhancement_net import EnhancementNet


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n====================================")
print("STARTING EVALUATION PIPELINE")
print("LOL + SALT & PEPPER DATASETS")
print("====================================")
print("DEVICE :", device)

if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))


# ============================================================
# MODEL CONFIG
# ============================================================

CHANNELS        = 48
NUM_RRDB_BLOCKS = 3
MODEL_PATH      = "outputs/checkpoints/best_model.pth"


# ============================================================
# DATASET CONFIGS  (both have ground truth)
# ============================================================

DATASETS = [
    {
        "name"      : "LOL eval15",
        "low_dir"   : "test_images/lol_dataset/eval15/low",
        "high_dir"  : "test_images/lol_dataset/eval15/high",
        "output_dir": "testing_model/results/lol_eval15",
    },
    {
        "name"      : "LOL our485",
        "low_dir"   : "test_images/lol_dataset/our485/low",
        "high_dir"  : "test_images/lol_dataset/our485/high",
        "output_dir": "testing_model/results/lol_our485",
    },
    {
        "name"      : "Salt and Pepper",
        "low_dir"   : "test_images/salt_pepper/Noisy_folder",
        "high_dir"  : "test_images/salt_pepper/Ground_truth",
        "output_dir": "testing_model/results/salt_pepper",
    }
]


# ============================================================
# MODEL INITIALIZATION
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
# PROCESS ONE DATASET (Wrapped into a functional block)
# ============================================================

def process_dataset(dataset):
    name       = dataset["name"]
    low_dir    = dataset["low_dir"]
    high_dir   = dataset["high_dir"]
    output_dir = dataset["output_dir"]

    print(f"\n{'='*52}")
    print(f"  DATASET : {name}")
    print(f"{'='*52}")

    if not os.path.isdir(low_dir):
        print(f"  ERROR : Low folder not found -> {low_dir}")
        return None

    if not os.path.isdir(high_dir):
        print(f"  ERROR : GT folder not found -> {high_dir}")
        return None

    os.makedirs(output_dir, exist_ok=True)

    # ======================================================
    # CSV FILE SETUP
    # ======================================================
    csv_path = os.path.join(output_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "Image",
        "Noisy_PSNR",
        "Output_PSNR",
        "PSNR_Gain",
        "Noisy_SSIM",
        "Output_SSIM",
        "SSIM_Gain",
        "Latency_ms"
    ])

    image_list = sorted([
        f for f in os.listdir(low_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    print(f"  Images Found : {len(image_list)}")

    if len(image_list) == 0:
        csv_file.close()
        return None

    total_noisy_psnr  = 0.0
    total_output_psnr = 0.0
    total_noisy_ssim  = 0.0
    total_output_ssim = 0.0
    total_latency_ms  = 0.0
    count             = 0

    for image_name in image_list:
        low_path = os.path.join(low_dir, image_name)
        gt_name = image_name

        if name == "Salt and Pepper":
            gt_name = gt_name.replace("noisy_", "")

        high_path = os.path.join(high_dir, gt_name)

        if not os.path.isfile(high_path):
            print(f"Skipping (GT missing) : {image_name}")
            continue

        try:
            noisy_image = Image.open(low_path).convert("RGB")
            gt_image    = Image.open(high_path).convert("RGB")

            noisy_tensor = transform(noisy_image).unsqueeze(0).to(device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            with torch.inference_mode():
                output_tensor = model(noisy_tensor)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000.0
            output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

            noisy_np = np.array(noisy_image).astype(np.float32) / 255.0
            gt_np    = np.array(gt_image).astype(np.float32) / 255.0

            output_np = (
                output_tensor
                .squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )

            output_np = np.clip(output_np, 0.0, 1.0)

            noisy_psnr = peak_signal_noise_ratio(gt_np, noisy_np, data_range=1.0)
            output_psnr = peak_signal_noise_ratio(gt_np, output_np, data_range=1.0)

            noisy_ssim = structural_similarity(gt_np, noisy_np, channel_axis=2, data_range=1.0)
            output_ssim = structural_similarity(gt_np, output_np, channel_axis=2, data_range=1.0)

            psnr_gain = output_psnr - noisy_psnr
            ssim_gain = output_ssim - noisy_ssim

            # ==================================================
            # WRITE CSV ROW
            # ==================================================
            csv_writer.writerow([
                image_name,
                f"{noisy_psnr:.4f}",
                f"{output_psnr:.4f}",
                f"{psnr_gain:.4f}",
                f"{noisy_ssim:.4f}",
                f"{output_ssim:.4f}",
                f"{ssim_gain:.4f}",
                f"{latency_ms:.2f}"
            ])

            # Process visualization layout: Ground Truth | Noisy Input | Enhanced Output
            gt_save     = (gt_np * 255).astype(np.uint8)
            noisy_save  = (noisy_np * 255).astype(np.uint8)
            output_save = (output_np * 255).astype(np.uint8)

            comparison = np.concatenate([gt_save, noisy_save, output_save], axis=1)
            comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(output_dir, image_name), comparison_bgr)

            total_noisy_psnr  += noisy_psnr
            total_output_psnr += output_psnr
            total_noisy_ssim  += noisy_ssim
            total_output_ssim += output_ssim
            total_latency_ms  += latency_ms
            count += 1

        except Exception as e:
            print(f"Failed : {image_name}")
            print(e)

    if count == 0:
        csv_file.close()
        return None

    avg_noisy_psnr  = total_noisy_psnr / count
    avg_output_psnr = total_output_psnr / count
    avg_noisy_ssim  = total_noisy_ssim / count
    avg_output_ssim = total_output_ssim / count
    avg_latency_ms  = total_latency_ms / count

    # ==================================================
    # WRITE SUMMARY TO CSV
    # ==================================================
    csv_writer.writerow([])
    csv_writer.writerow(["SUMMARY STATISTICS"])
    csv_writer.writerow(["Metric Description", "Value"])
    csv_writer.writerow(["Total Images Processed", count])
    csv_writer.writerow(["Avg Noisy PSNR", f"{avg_noisy_psnr:.4f}"])
    csv_writer.writerow(["Avg Output PSNR", f"{avg_output_psnr:.4f}"])
    csv_writer.writerow(["Avg PSNR Gain", f"{(avg_output_psnr - avg_noisy_psnr):.4f}"])
    csv_writer.writerow(["Avg Noisy SSIM", f"{avg_noisy_ssim:.4f}"])
    csv_writer.writerow(["Avg Output SSIM", f"{avg_output_ssim:.4f}"])
    csv_writer.writerow(["Avg SSIM Gain", f"{(avg_output_ssim - avg_noisy_ssim):.4f}"])
    csv_writer.writerow(["Avg Latency (ms)", f"{avg_latency_ms:.2f}"])

    csv_file.close()

    return {
        "name": name,
        "count": count,
        "avg_noisy_psnr": avg_noisy_psnr,
        "avg_output_psnr": avg_output_psnr,
        "avg_psnr_gain": avg_output_psnr - avg_noisy_psnr,
        "avg_noisy_ssim": avg_noisy_ssim,
        "avg_output_ssim": avg_output_ssim,
        "avg_ssim_gain": avg_output_ssim - avg_noisy_ssim,
        "avg_latency_ms": avg_latency_ms,
    }


# ============================================================
# RUN ALL DATASETS
# ============================================================

all_results = []

for dataset in DATASETS:
    result = process_dataset(dataset)
    if result is not None:
        all_results.append(result)


# ============================================================
# FINAL SUMMARY PRINT
# ============================================================

print(f"\n{'='*52}")
print("FINAL SUMMARY")
print(f"{'='*52}")

for r in all_results:
    print(f"\n[ {r['name']} ]")
    print(f"  Images Processed   : {r['count']}")
    print(f"  Avg Noisy  PSNR    : {r['avg_noisy_psnr']:.4f}")
    print(f"  Avg Output PSNR    : {r['avg_output_psnr']:.4f}")
    print(f"  Avg PSNR Gain      : {r['avg_psnr_gain']:.4f}")
    print(f"  Avg Noisy  SSIM    : {r['avg_noisy_ssim']:.4f}")
    print(f"  Avg Output SSIM    : {r['avg_output_ssim']:.4f}")
    print(f"  Avg SSIM Gain      : {r['avg_ssim_gain']:.4f}")
    print(f"  Avg Latency        : {r['avg_latency_ms']:.2f} ms")

print("\nEvaluation Completed Successfully\n")