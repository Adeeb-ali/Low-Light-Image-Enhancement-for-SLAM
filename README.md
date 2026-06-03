# EnhancementNet: Low-Light Image Enhancement for Robotics, SLAM, and Embedded Vision

EnhancementNet is a deep learning-based low-light image enhancement and denoising framework developed for robotics, autonomous navigation, drone vision, visual SLAM, and embedded AI applications.

The primary objective of this project is to improve visual perception in challenging illumination environments where conventional cameras struggle to capture useful features for downstream computer vision algorithms. The model enhances image brightness, suppresses noise, improves edge visibility, and significantly increases feature detectability for feature-based vision systems such as ORB-SLAM.

---

## Motivation

Visual systems deployed in drones, robots, surveillance systems, and autonomous platforms frequently operate in low-light conditions. Under these conditions:

* **Feature detectors fail** to identify sufficient keypoints.
* **Feature matching** becomes highly unreliable.
* **Localization accuracy** degrades sharply.
* **SLAM systems** struggle to maintain tracking.
* **Object visibility** decreases significantly.

EnhancementNet addresses these challenges by processing and stabilizing image quality before downstream perception tasks take over.

---

## Architecture & Hardware Optimization (NAS)

EnhancementNet balances representational capacity with deployment efficiency. It natively supports standard hand-crafted designs as well as automated, hardware-aware sub-networks.

### Baseline Architecture
* **Core Components:** Residual in Residual Dense Blocks (RRDB), Multi-Scale Feature Fusion, Illumination Enhancement Modules, and Residual Reconstruction Layers.
* **Channels:** 48  
* **RRDB Blocks:** 3  
* **Framework:** PyTorch

### Phase 3: Neural Architecture Search (NAS)
To support resource-constrained embedded targets, the framework includes a Hardware-Aware Neural Architecture Search (NAS) pipeline. This allows the system to explore macro/micro architecture spaces to discover optimized, low-latency configurations tailored specifically for edge AI deployment.

---

## Experimental Results

The model was evaluated on real-world low-light video sequences to measure its direct impact on feature-based vision pipelines.

### Quantitative Evaluation

| Metric | Raw Images | Enhanced Images | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Average Brightness** | 20.13 | 96.40 | **~4.8×** |
| **ORB Features Detected** | 93.68 | 589.81 | **~6.3×** |
| **Valid ORB Matches** | 62.18 | 372.99 | **~6.0×** |

These results demonstrate that EnhancementNet significantly stabilizes the visual information matrix required by feature-based localization algorithms.

### Observed Signal Enhancement Performance
* **Moderate Low-Light Conditions:** Typical visual enhancement gain of `+5 dB to +8 dB`
* **Extreme Low-Light & High Noise:** Typical visual enhancement gain of `+2 dB to +5 dB`

> ⚠️ *Note: These figures represent qualitatively observed visual enhancement gains across multiple test conditions. Performance varies based on sensor characteristics, ambient noise profiles, scene illumination, and motion blur levels.*

---

## Visual Results & Datasets

For comprehensive visual comparisons, video demonstrations, datasets, and detailed evaluation outputs, visit the external repository:

📂 **[Google Drive Results Repository](https://drive.google.com/drive/folders/1vh3wC2cMafbKwcCKSe0bAA3U7khxMKpd?usp=drive_link)**

### Contents Available:
* Input vs. Enhanced Video Roll
* Side-by-Side ORB Feature Maps
* Real-time Feature Matching Comparisons
* Raw Quantitative Evaluation CSVs

---

## Quick Start (Docker Deployment)

The fastest way to run inference using EnhancementNet is via the prebuilt Docker image. This eliminates the need for manual environment configuration or dependency installation.

### 1. Pull the Docker Image
```bash
docker pull adeebali521/lowlight-denoise-enhancer:latest

```

### 2. Run Inference

Mount your local input and output directories to the container using volume flags (`-v`):

```bash
docker run --rm \
  -v "/path/to/input:/input" \
  -v "/path/to/output:/output" \
  adeebali521/lowlight-denoise-enhancer:latest \
  --input /input \
  --output /output

```

#### Running Example:

```bash
docker run --rm \
  -v "/home/user/images:/input" \
  -v "/home/user/results:/output" \
  adeebali521/lowlight-denoise-enhancer:latest \
  --input /input \
  --output /output

```

### Container Arguments

| Argument | Description |
| --- | --- |
| `--input` | Path to the directory containing input images. |
| `--output` | Path to the directory where results will be saved. |
| `--width` | Target output image width. |
| `--height` | Target output image height. |
| `--max-size` | *Optional.* Upper bound limit for image dimensions. |
| `--model` | *Optional.* Path to a custom model checkpoint. |

### Generated Outputs

The container automatically structures your target output directory as follows:

```text
output/
├── enhanced_image_001.jpg
├── enhanced_image_002.jpg
├── latency.csv             # Per-image inference tracking
└── summary.csv             # General run performance statistics

```

---

## Repository Structure & Local Setup

For manual configuration, local training, or extending the Neural Architecture Search (NAS) pipeline, please refer to the dedicated setup document:

📖 **See [SETUP.md](https://www.google.com/search?q=./SETUP.md) for local installation, training, and NAS execution details.**

```text
├── phase2_extension/       # Core training, inference, and evaluation scripts
├── phase3_nas/             # Supernet spaces, exploration, and retraining engines
├── requirements.txt         # Package dependencies
├── SETUP.md                # Environment setup guide
└── README.md               # Main documentation

```

---

## FPGA Deployment Status

The EnhancementNet architecture was fundamentally engineered with hardware synthesis and parallelization boundaries in mind (e.g., highly predictable memory strides, data reuse configurations).

* **Architecture Development:** Complete
* **Training & NAS Evaluation:** Complete
* **Docker Portability Layer:** Complete
* **Hardware Synthesis & FPGA Validation:** `Pending`

The model is considered **FPGA-deployment ready** from a structural and mathematical perspective; however, physical hardware verification on targeted silicon blocks has not yet been finalized.

*If you are an FPGA developer, hardware acceleration engineer, or embedded AI researcher interested in benchmarking EnhancementNet on physical hardware boards, please reach out via the contact information below.*

---

## Future Work

* Full downstream integration with **ORB-SLAM3**.
* High-speed visual localization and absolute mapping stability evaluations.
* **ROS / ROS2** node packaging for robotic middleware ecosystems.
* FPGA register-transfer level (RTL) implementation and real-time hardware acceleration prototyping.

---

## Author

**Adeeb Ali** *Research Interests:* Computer Vision • Embedded AI • FPGA Acceleration • Robotics • Visual SLAM • Autonomous Systems

---

## License

This project is provided exclusively for research and educational purposes.


