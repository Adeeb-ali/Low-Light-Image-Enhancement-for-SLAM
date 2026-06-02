# EnhancementNet: Deep Learning Low-Light Image Enhancement

`EnhancementNet` is a containerized deep learning pipeline optimized for low-light image enhancement in robotics, drone vision, and SLAM applications.

---

## Quick Start

### 1. Pull Image

```bash
docker pull adeebali521/enhancementnet:latest

```

### 2. Run Container

Replace `/path/to/input` and `/path/to/output` with your local directory paths.

**CPU Mode:**

```bash
docker run --rm -v "/path/to/input:/input" -v "/path/to/output:/output" adeebali521/enhancementnet:latest --input /input --output /output

```

**NVIDIA GPU Mode:**

```bash
docker run --rm --gpus all -v "/path/to/input:/input" -v "/path/to/output:/output" adeebali521/enhancementnet:latest --input /input --output /output

```

**Advanced Example (Resizing):**

```bash
docker run --rm -v "/path/to/input:/input" -v "/path/to/output:/output" adeebali521/enhancementnet:latest --input /input --output /output --width 752 --height 480

```

---

## Configuration CLI Parameters

| Argument | Description |
| --- | --- |
| `--input` | **Required.** Path to input image directory inside container. |
| `--output` | **Required.** Path to output directory inside container. |
| `--width` | Target output image width (optional). |
| `--height` | Target output image height (optional). |
| `--max-size` | Maximum image dimension constraint (optional). |

---

## Output Structure

Supported inputs: `.jpg`, `.jpeg`, `.png`.

```text
output/
├── frame_000001.jpg   # Enhanced image
├── frame_000002.jpg
├── latency.csv        # Per-image inference time
└── summary.csv        # Performance statistics

```

---

## Features & Architecture

* **Network:** RRDB (Residual in Residual Dense Blocks) + Illumination Enhancement Module.
* **Hardware:** CPU and CUDA-compatible deployment. Native MPS support is available when running outside Docker on Apple Silicon.
* **Use Cases:** Visual SLAM feature-matching optimization, autonomous navigation, and edge AI.

---

## License & Contact

* **License:** Research and educational purposes.
* **Maintainer:** Adeeb (Computer Vision & Embedded AI)