import os
import json
import torch
from .core.retrain import Retrainer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n====================================")
    print(f"DEVICE : {device}")

    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"CUDA Devices : {torch.cuda.device_count()}")
    print("====================================\n")

    config_path = "best_architecture.json"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Could not find '{config_path}'. You must run nas_main.py first to export the winner!"
        )

    print(f"Loading search-space winner from {config_path}...")
    with open(config_path, "r") as f:
        architecture = json.load(f)

    print(f"Retraining Target: Channels={architecture['channels']}, Blocks={architecture['num_rrdb_blocks']}")
    print("====================================\n")

    retrainer = Retrainer(device=device)
    retrainer.retrain(
        architecture=architecture,
        epochs=200,
        learning_rate=1e-4
    )


if __name__ == "__main__":
    main()
