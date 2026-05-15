import torch
import os
import numpy as np
from pytorch_nndct.apis import torch_quantizer
from model_builder import build_model
from utils import load_best_config
from PIL import Image
import torchvision.transforms as transforms


def load_calibration_images(calib_dir, size=(256, 256), max_images=50):

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    images = []

    files = os.listdir(calib_dir)

    for file in files:
        if len(images) >= max_images:
            break

        path = os.path.join(calib_dir, file)

        try:
            img = Image.open(path).convert("RGB")

            # -------------------------
            # 🔥 APPLY DEGRADATION (IMPORTANT)
            # -------------------------
            arr = np.array(img) / 255.0

            # Gaussian noise
            noise = np.random.normal(0, 0.05, arr.shape)
            arr = np.clip(arr + noise, 0, 1)

            img = Image.fromarray((arr * 255).astype("uint8"))

            # Transform
            img = transform(img)

            images.append(img)

        except Exception as e:
            print(f"⚠️ Skipping {file}: {e}")
            continue

    if len(images) == 0:
        raise RuntimeError("❌ No valid calibration images found")

    print(f"✅ Loaded {len(images)} calibration images")

    return torch.stack(images)


# -------------------------
# MAIN QUANTIZATION
# -------------------------
def run_quantization(project_root="."):

    print("=" * 60)
    print("🔹 STARTING QUANTIZATION (PTQ)")
    print("=" * 60)

    # -------------------------
    # Load model
    # -------------------------
    config, model_path = load_best_config(project_root)

    model = build_model(config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print(f"✅ Loaded model: {model_path}")

    dummy_input = torch.randn(1, 3, 256, 256)

    # -------------------------
    # CALIBRATION STEP
    # -------------------------
    print("🔹 Creating quantizer (CALIB mode)")

    quantizer = torch_quantizer(
        quant_mode="calib",
        module=model,
        input_args=(dummy_input,)
    )

    quant_model = quantizer.quant_model

    # 🔥 Use mounted dataset path
    calib_dir = "/workspace/phase2_multidegradation/train_images/DIV2K_train_HR"

    print(f"🔹 Loading calibration data from:\n{calib_dir}")

    calib_data = load_calibration_images(calib_dir)

    print("🔹 Running calibration...")

    with torch.no_grad():
        for img in calib_data:
            quant_model(img.unsqueeze(0))

    os.makedirs("output", exist_ok=True)

    print("🔹 Exporting quant config...")
    quantizer.export_quant_config()

    # -------------------------
    # EXPORT XMODEL
    # -------------------------
    print("🔹 Switching to TEST mode")

    quantizer = torch_quantizer(
        quant_mode="test",
        module=model,
        input_args=(dummy_input,)
    )

    print("🔹 Exporting XMODEL...")

    quantizer.export_xmodel("output")

    print("✅ XMODEL exported → output/")
    print("=" * 60)