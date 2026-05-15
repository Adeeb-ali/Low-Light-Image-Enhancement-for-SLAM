import json
import os


def load_best_config(project_root="."):

    project_root = os.path.abspath(project_root)

    json_path = os.path.join(
        project_root,
        "phase3_nas",
        "storage",
        "best_arch.json"
    )

    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"❌ best_arch.json not found:\n{json_path}")

    # -------------------------
    # Load JSON
    # -------------------------
    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("❌ best_arch.json is empty or invalid")

    best = data[0]

    if "config" not in best:
        raise KeyError("❌ Missing 'config' in best_arch.json")

    config = best["config"]

    # -------------------------
    # Model directory
    # -------------------------
    models_dir = os.path.join(
        project_root,
        "phase3_nas",
        "storage",
        "models"
    )

    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"❌ Models directory not found:\n{models_dir}")

    # -------------------------
    # Try JSON model_path
    # -------------------------
    model_path = None

    if "model_path" in best and best["model_path"]:
        candidate = os.path.join(models_dir, best["model_path"])

        if os.path.isfile(candidate):
            model_path = candidate
            print(f"✅ Using model from JSON: {model_path}")
        else:
            print(f"⚠️ JSON model_path not found → {candidate}")

    # -------------------------
    # Fallback: latest .pth
    # -------------------------
    if model_path is None:
        pth_files = [
            os.path.join(models_dir, f)
            for f in os.listdir(models_dir)
            if f.endswith(".pth")
        ]

        if not pth_files:
            raise FileNotFoundError("❌ No .pth files found in models directory")

        # Pick latest modified file
        pth_files.sort(key=os.path.getmtime, reverse=True)
        model_path = pth_files[0]

        print(f"⚠️ Using latest .pth: {model_path}")

    if not os.path.isfile(model_path):
        raise RuntimeError(f"❌ Invalid model path:\n{model_path}")

    return config, model_path