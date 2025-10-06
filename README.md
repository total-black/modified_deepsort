# Modified DeepSORT: Multi-Object Tracking with Person Re-identification

A multi-object tracking system combining object detection, person re-identification (ReID), and DeepSORT tracking algorithm.

## Installation

```bash
pip install numpy torch torchvision ultralytics torchreid opencv-python-headless matplotlib tqdm scipy opencv-python Pillow git+https://github.com/cheind/py-motmetrics.git
```

## Core Scripts

### 1. Full Tracking Pipeline (`run_deepsort_hota.py`)

Runs complete tracking pipeline: detection → ReID → tracking → evaluation.

**Usage:**
```bash
python scripts/run_deepsort_hota.py \
    --sequence_dir data/KITTI-17 \
    --detector yolov5 \
    --reid_model osnet_x0_25 \
    --device cuda
```

**Key Parameters:**
- `--sequence_dir`: Path to dataset sequence (must contain `img1/` folder)
- `--detector`: `yolov5` (default), `fasterrcnn`, `retinanet`
- `--reid_model`: `osnet_x0_25` (default), `osnet_x0_5`, `osnet_x1_0`, `resnet50`, `densenet121`
- `--conf_threshold`: Detection confidence (default: 0.5)
- `--max_cosine_distance`: ReID matching threshold (default: 0.2)
- `--max_age`: Track deletion age (default: 30)
- `--n_init`: Frames to confirm track (default: 3)
- `--device`: `cpu`, `cuda`, `mps`

**Output:** MOT format tracking results + HOTA evaluation (if ground truth available)

### 2. Hyperparameter Optimization (`grid_search.py`)

Automatically finds optimal parameters by testing different combinations.

**Usage:**
```bash
python scripts/grid_search.py \
    --sequence_dir data/KITTI-17 \
    --detector yolov5 \
    --reid_model osnet_x0_25
```

**What it optimizes:**
- Detection confidence threshold (0.3, 0.4, 0.5, 0.6, 0.7)
- ReID cosine distance (0.1, 0.2, 0.3, 0.4)
- Track age (20, 30, 40, 50)
- NMS IoU threshold (0.3, 0.4, 0.5, 0.6)

**Output:** Best parameter combination with HOTA score

## Dataset Setup

Create a `data/` folder in the project root and place your datasets there:

```
project_root/
├── data/                    # Dataset folder (create this)
│   ├── KITTI-17/           # Example sequence
│   │   ├── img1/           # Image frames (000001.jpg, 000002.jpg, ...)
│   │   ├── gt/             # Ground truth (optional, for evaluation)
│   │   │   └── gt.txt
│   │   └── det/            # Pre-computed detections (optional)
│   │       └── det.txt
│   └── MOT16-09/           # Another sequence
│       └── ...
└── modified_deepsort/      # Project code
    └── ...
```

**Required structure for each sequence:**
- `img1/` folder with numbered image files (000001.jpg, 000002.jpg, etc.)
- `gt/gt.txt` (optional, for HOTA evaluation)
- `det/det.txt` (optional, for pre-computed detections)

## Quick Examples

**Basic tracking:**
```bash
python scripts/run_deepsort_hota.py --sequence_dir data/KITTI-17
```

**High accuracy setup:**
```bash
python scripts/run_deepsort_hota.py \
    --sequence_dir data/KITTI-17 \
    --detector fasterrcnn \
    --reid_model osnet_x1_0 \
    --conf_threshold 0.6 \
    --max_cosine_distance 0.15
```

**Find best parameters:**
```bash
python scripts/grid_search.py --sequence_dir data/KITTI-17 --detector yolov5
```
