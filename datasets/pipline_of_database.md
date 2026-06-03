# Dataset Configuration & Pipeline

This project utilizes the **LOL (Low-Light) Dataset** for both training and evaluation. Below is the detailed directory structure and data processing pipeline.

---

## Directory Structure

Please organize your `datasets/` directory as shown below.

```text
datasets/
├── _ReadMe.txt
├── Scene_Instances.txt
├── Data/
└── lol_dataset/
    ├── our485/
    │   ├── low/      # Low-light training inputs
    │   └── high/     # Ground-truth normal-light training images
    └── eval15/
        ├── low/      # Low-light evaluation inputs
        └── high/     # Ground-truth normal-light evaluation images

```

---

## Dataset Splits

| Split | Path | Image Pairs | Description |
| --- | --- | --- | --- |
| **Training** | `datasets/lol_dataset/our485/` | 485 | Paired low-light (`low/`) and ground-truth normal-light (`high/`) images used for model optimization. |
| **Evaluation** | `datasets/lol_dataset/eval15/` | 15 | Paired low-light (`low/`) and ground-truth (`high/`) images used for benchmarking performance. |

---

## Workflow Pipelines

### 1. Training Stage

The training pipeline passes the low-light image through the network, generates a prediction, and calculates the loss against the high-light ground truth to update the weights.

```text
datasets/lol_dataset/our485/low/ (Input)
               │
               ▼
        EnhancementNet
               │
               ▼
          Prediction
               │
               ▼
        Loss Calculation ◄────────► datasets/lol_dataset/our485/high/ (GT)

```

### 2. Evaluation Stage

The evaluation pipeline generates enhanced outputs and runs them through a series of quantitative and feature-based analyses.

```text
datasets/lol_dataset/eval15/low/ (Input)
               │
               ▼
        EnhancementNet
               │
               ▼
        Enhanced Output
               │
               ├──► Brightness Analysis
               ├──► ORB Feature Analysis
               ├──► ORB Matching Analysis
               └──► Visual Comparison (vs. datasets/lol_dataset/eval15/high/)

```

---

## Additional Information

* **Auxiliary Data:** The repository includes `Scene_Instances.txt`, `_ReadMe.txt`, and the `Data/` folder, which contain auxiliary metadata utilized during specific experimentation phases.
* **Path Configuration:** Dataset paths are currently configured manually inside the training and inference scripts. If your local setup differs, please update the path variables accordingly.
* **Storage Note:** Large dataset files are **not** tracked by Git. You must download the LOL Dataset separately and place it into the `datasets/` directory to reproduce the experiments.
