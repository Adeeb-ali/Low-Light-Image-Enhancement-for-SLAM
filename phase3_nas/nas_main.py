import json
import torch
from .core.search_space import SEARCH_SPACE
from .core.scorer import ArchitectureScorer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n====================================")
    print("NEURAL ARCHITECTURE SEARCH")
    print("====================================\n")
    print(f"DEVICE : {device}")

    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"CUDA Devices : {torch.cuda.device_count()}")
    print("\n====================================\n")

    scorer = ArchitectureScorer(device=device)
    best_score = -1
    best_architecture = None

    for channels in SEARCH_SPACE["channels"]:
        for num_rrdb_blocks in SEARCH_SPACE["num_rrdb_blocks"]:
            architecture = {
                "channels": channels,
                "num_rrdb_blocks": num_rrdb_blocks
            }

            print("\n====================================")
            print("TESTING ARCHITECTURE")
            print("====================================\n")
            print(f"Channels        : {channels}")
            print(f"RRDB Blocks     : {num_rrdb_blocks}")

            result = scorer.score(architecture)
            score = result["score"]

            print(f"\nArchitecture Score : {score:.6f}")
            print(f"Loss               : {result['loss']:.6f}")
            print(f"Charbonnier Loss   : {result['charbonnier_loss']:.6f}")
            print(f"SSIM Loss          : {result['ssim_loss']:.6f}")
            print(f"Edge Loss          : {result['edge_loss']:.6f}")

            if score > best_score:
                best_score = score
                best_architecture = architecture
                print("\nNEW BEST ARCHITECTURE FOUND")

    print("\n====================================")
    print("BEST ARCHITECTURE FOUND BY NAS")
    print("====================================\n")
    print(f"Channels        : {best_architecture['channels']}")
    print(f"RRDB Blocks     : {best_architecture['num_rrdb_blocks']}")
    print(f"Best Score      : {best_score:.6f}\n")

    # --- FIX: Save the winner to disk so retraining phase can load it ---
    output_path = "best_architecture.json"
    with open(output_path, "w") as f:
        json.dump(best_architecture, f, indent=4)
    print(f"Successfully serialized winner to {output_path}")
    print("====================================\n")


if __name__ == "__main__":
    main()