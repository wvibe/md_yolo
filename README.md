# MD YOLO

A project for developing and experimenting with YOLO (You Only Look Once) models and their variations for object detection.

## Project Overview

This repository contains experiments, implementations, and comparisons of various YOLO architectures including:

- YOLOv3
- YOLOv4
- YOLOv5
- YOLOv6
- YOLOv7
- YOLOv8
- YOLO-NAS

## Project Structure

```
md_yolo/
│
├── data/               # Dataset storage and processing scripts
│   ├── raw/            # Raw datasets
│   ├── processed/      # Processed datasets
│   └── augmented/      # Augmented datasets for training
│
├── models/             # Saved model checkpoints and configurations
│   ├── yolov3/
│   ├── yolov5/
│   ├── yolov8/
│   └── ...
│
├── notebooks/          # Jupyter notebooks for exploration and visualization
│
├── src/                # Source code
│   ├── models/         # Model implementations
│   ├── utils/          # Utility functions
│   ├── train.py        # Training script
│   ├── eval.py         # Evaluation script
│   └── infer.py        # Inference script
│
└── tests/              # Unit tests
```

## Setup

```bash
# Create virtual environment using uv
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows

# Install dependencies
uv pip install -r requirements.txt
```

## Usage

Details on how to use the project will be added as the project evolves.

## License

MIT 