# EnhancementNet Setup Guide

This document explains how to set up and run the EnhancementNet training, evaluation, and Neural Architecture Search (NAS) pipeline used in this project.

## System Requirements

### Python Version
This project was developed and tested using:
* **Python 3.10**

Check your Python version:
```bash
python --version
# or
python3 --version

```

---

## Clone Repository

```bash
git clone [https://github.com/Adeeb-ali/fpga-image-enhancement-pipeline.git](https://github.com/Adeeb-ali/fpga-image-enhancement-pipeline.git)
cd fpga-image-enhancement-pipeline

```

---

## Create Virtual Environment

### Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate

```

### Windows:

```bash
python -m venv venv
venv\Scripts\activate

```

### Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt

```

Verify installation:

```bash
pip list

```


---

## Model Weights

The repository includes:

* **`best_model.pth`**: This checkpoint contains the trained EnhancementNet model used in the reported baseline experiments.

---

## Phase 2: Standard Training & Inference

### Training

Training scripts are located inside `phase2_extension/`. Run training via:

```bash
python phase2_extension/train.py

```

Adjust training configurations (learning rate, epochs, etc.) directly inside the source files when required.

### Inference

Run inference using:

```bash
python phase2_extension/inference.py

```

Input and output paths are configured manually inside the script.

---

## Phase 3: Neural Architecture Search (NAS) & Retraining

Phase 3 introduces automated search capabilities to find hardware-efficient sub-networks.

### 1. Run Architecture Search

To kick off the supernet training and search space exploration, run:

```bash
python -m phase3_nas.nas_main

```

> 💡 **Important Note on Search Spaces:** > By default, the configuration is set to a **weak architecture/smaller search space** for rapid testing and verification. To search for a robust, high-performance model, you must manually increase the search space configurations directly inside the `phase3_nas/nas_main.py` source file before running.

### 2. Retrain the Discovered Architecture

Once the search is complete and you have selected your optimal architecture, initiate the standalone training process using:

```bash
python -m phase3_nas.retrain_only

```

---

## Evaluation

Evaluation scripts are provided for:

* Brightness Analysis
* ORB Feature Analysis
* ORB Matching Analysis
* Visual Comparison

Run the corresponding evaluation script directly from the project root directory.

---

## Experimental Environment

### Development Environment:

* Python 3.10
* PyTorch 2.2.0
* Torchvision 0.17.0
* OpenCV 4.9.0
* NumPy 1.26.4
* *Optional (for NAS profiling):* `thop` / `torchinfo`

### Hardware Used:

* Apple Silicon (M-Series)
* NVIDIA GPU Compatible
* CPU Compatible

---

## Notes

* This repository contains the original research and development code used during experimentation.
* Dataset locations, checkpoints, search space dimensions, and training configurations are intentionally configured manually inside scripts to keep execution straightforward.
* Results reported in the project were generated using the included model architecture, NAS configurations, and evaluation pipeline.

## Additional Resources

Results, evaluation outputs, videos, and additional materials are available through the links provided in the project **README.md**.
