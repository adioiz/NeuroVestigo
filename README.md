# SAM2 Neuron Fine-Tuning

LoRA+ fine-tuning pipeline for SAM2 on neuron microscopy data.

## Overview

This repository provides a training pipeline to fine-tune SAM2 (Segment Anything Model 2) for neuron segmentation using LoRA+ (Low-Rank Adaptation with asymmetric learning rates). Fine-tuning improves segmentation accuracy for your specific microscopy domain.

**For labeling/annotation**, use the companion [neurons-labeling](../neurons-labeling/) repository.

## Features

- **LoRA+ fine-tuning**: Efficient training (~0.5% of parameters) with asymmetric A/B learning rates
- **Feature precomputation**: 10x faster training by caching frozen image encoder output
- **In-memory caching**: All frames cached in RAM, eliminating disk I/O
- **cbDice loss**: Skeleton topology-preserving loss with boundary weighting for tubular structures
- **Supervoxel loss**: Connectivity-preserving loss for split/merge errors
- **Strong augmentations**: Elastic deformations, blur, noise (full pipeline without precomputation); feature-level flips (with precomputation)
- **Early stopping**: Automatic stopping when metrics plateau
- **MPS & CUDA support**: Works on Apple Silicon and NVIDIA GPUs

## Repository Connection

This repo is **symlinked** to `neurons-labeling` for seamless data sharing:

```
labeled_masks/       -> ../neurons-labeling/labeled_masks/
labeled_coordinates/ -> ../neurons-labeling/labeled_coordinates/
```

The labeling repo also symlinks back here:

```
neurons-labeling/checkpoints/ -> ../Fine-tune-SAM2-model-for-Neurons-segmentation/checkpoints/
```

**No manual copying needed** - label in one repo, train in the other.

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install SAM2
pip install git+https://github.com/facebookresearch/sam2.git
```

## Quick Start

### Step 1: Create Data Manifest

Create a `data_manifest.json` file listing your TIF stacks. Use relative paths via symlinks:

```json
[
    {
        "image_tif": "/path/to/neuron_video_1.tif",
        "mask_tif": "labeled_masks/video_1_masks.tif",
        "name": "video_1"
    },
    {
        "image_tif": "/path/to/neuron_video_2.tif",
        "mask_tif": "labeled_masks/video_2_masks.tif",
        "name": "video_2"
    }
]
```

**Note:** `mask_tif` uses relative path `labeled_masks/` which resolves via symlink to `../neurons-labeling/labeled_masks/`.

**Requirements:**
- `image_tif`: Grayscale microscopy TIF stack (T, H, W)
- `mask_tif`: Integer-labeled mask TIF stack (T, H, W) where each neuron has a unique ID
- At least 5-10 videos recommended for good generalization

### Step 2: Prepare Data

Convert TIF stacks to PNG training format:

```bash
python cli.py prepare --manifest data_manifest.json --output-dir training_data
```

This creates:
```
training_data/
├── train/
│   ├── video_1/
│   │   ├── images/
│   │   │   ├── 000.png
│   │   │   ├── 001.png
│   │   │   └── ...
│   │   └── masks/
│   │       ├── 000.png
│   │       ├── 001.png
│   │       └── ...
│   └── video_2/
│       └── ...
├── val/
│   └── ...
└── stats.json  # Normalization statistics
```

### Step 3: Train

```bash
# Basic training (recommended — uses feature precomputation for 10x speedup)
python cli.py train --epochs 30

# With cbDice for thin structures (requires more memory)
python cli.py train --epochs 30 --batch-size 2

# For Apple Silicon (MPS) if OOM occurs
python cli.py train --epochs 30 --batch-size 1 --no-cbdice

# Full customization
python cli.py train \
    --epochs 30 \
    --batch-size 2 \
    --lora-rank 8 \
    --lora-lr 1e-4 \
    --patience 10 \
    --val-every 2

# Disable precomputation for full augmentation pipeline (slower but more diverse)
python cli.py train --epochs 30 --no-precompute
```

Training outputs:
```
=== Precomputing Image Encoder Features ===
  12 videos, 876 frames total
  Processed 876/876 frames...
  Feature precomputation done in 240.3s (6912 MB cached)

=== Training (60 epochs, 82 steps/epoch (precomputed features)) ===
  Using precomputed features (encoder skipped during training)
  Epoch 1/60, Step 10/82, Loss: 1.2340, LR: 3.33e-05, 7.8s/step
Epoch 1/60: train_loss=1.1200, time=640.2s, 7.8s/step
  Validating...
  val_loss=0.987, val_iou=0.543, val_dice=0.654
  New best IoU: 0.543 - saved checkpoint
```

### Step 4: Use Fine-Tuned Model

Thanks to the symlink, the weights are automatically available in `neurons-labeling`:

1. Open `neurons-labeling/` and run `python main.py`
2. Check **"Use fine-tuned model"** in the GUI
3. Click "SAM2: Propagate from frame 0"

No manual copying needed - the `neurons-labeling/checkpoints/` symlink points here.

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 60 | Maximum training epochs |
| `--batch-size` | 2 | Batch size (1 if OOM) |
| `--lora-rank` | 8 | LoRA rank (8-16 typical) |
| `--lora-alpha` | 16.0 | LoRA alpha (usually 2x rank) |
| `--lora-lr` | 1e-4 | LoRA A matrix learning rate |
| `--lora-b-lr-ratio` | 16.0 | LoRA+ B/A learning rate ratio |
| `--base-lr` | 1e-5 | Learning rate for non-LoRA params |
| `--warmup-epochs` | 1 | Warmup epochs before cosine decay |
| `--num-frames` | 8 | Frames per training clip |
| `--patience` | 10 | Early stopping patience |
| `--val-every` | 2 | Validation frequency (epochs) |
| `--val-samples` | 0 | Max validation batches (0=all, >0=faster) |
| `--no-cbdice` | False | Disable cbDice loss to save memory |
| `--supervoxel-weight` | 1.0 | Supervoxel connectivity loss weight |
| `--no-supervoxel` | False | Disable supervoxel loss |
| `--no-topology` | False | Disable BOTH cbDice and supervoxel (fast CPU/MPS training) |

**Performance parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--clip-stride` | 4 | Stride between clip start frames (1=all clips, reduces overlap) |
| `--no-precompute` | False | Disable feature precomputation (slower but full augmentation) |
| `--no-cache` | False | Disable in-memory frame caching (saves RAM but slower) |

## Key Metrics

| Metric | Good Range | Meaning |
|--------|------------|---------|
| `val_iou` | 0.65-0.85 | Intersection over Union (higher = better) |
| `val_dice` | 0.75-0.90 | Dice coefficient (higher = better) |
| Train/Val gap | < 0.1 | Low gap = not overfitting |

## Troubleshooting

### Training Too Slow

Feature precomputation is enabled by default and gives a **10x speedup**. If training is still slow:

```bash
# 1. Ensure precomputation is enabled (default)
python cli.py train
# Look for: "Using precomputed features (encoder skipped during training)"

# 2. Increase clip stride to reduce steps per epoch
python cli.py train --clip-stride 8

# 3. Reduce frames per clip
python cli.py train --num-frames 4
```

| Optimization | Effect |
|-------------|--------|
| Feature precomputation (default) | 10x faster per step |
| `--clip-stride 4` (default) | 4x fewer steps/epoch |
| `--clip-stride 8` | 8x fewer steps/epoch |
| `--num-frames 4` | ~2x faster per step |

### Out of Memory (OOM)
```bash
# Reduce batch size
python cli.py train --batch-size 1

# Disable ALL topology losses (fastest for CPU/MPS)
python cli.py train --no-topology

# Or disable only cbDice
python cli.py train --no-cbdice

# Disable in-memory frame caching (saves ~2 GB RAM)
python cli.py train --no-cache

# Or reduce frames per clip
python cli.py train --num-frames 4
```

### Broken Neuron Branches
If thin structures break:
1. Enable cbDice (don't use `--no-cbdice`)
2. Enable supervoxel loss (default: enabled)
3. Increase weights for connectivity losses:
   ```bash
   python cli.py train --supervoxel-weight 2.0
   ```
   Or edit `src/config.py`: `loss_cbdice_weight = 2.0`, `loss_supervoxel_weight = 2.0`

### Poor Generalization
1. Add more training data (5-10 videos minimum)
2. Reduce LoRA rank: `--lora-rank 4`
3. Use full augmentation: `--no-precompute` (slower but more augmentation diversity)
4. Check for data leakage (same video in train and val)

### Validation Too Slow
```bash
# Limit to 10 validation batches
python cli.py train --val-samples 10
```

## Project Structure

```
Fine-tune-SAM2-model-for-Neurons-segmentation/
├── cli.py                   # Training CLI
├── data_manifest.json       # Your data manifest file
├── requirements.txt
├── README.md
├── CODE_EXPLAINED.md        # Detailed code documentation
├── FINE_TUNING_GUIDE.md     # Complete training guide
├── labeled_masks/           # Symlink -> ../neurons-labeling/labeled_masks
├── labeled_coordinates/     # Symlink -> ../neurons-labeling/labeled_coordinates
├── checkpoints/             # Saved model weights (source of truth)
│   └── lora_weights.pt
├── training_data/           # Prepared PNG data
│   └── stats.json           # Normalization statistics
└── src/
    ├── config.py            # Hyperparameters
    ├── lora.py              # LoRA+ implementation
    ├── trainer.py           # Training loop
    ├── evaluate.py          # Validation metrics
    ├── neuron_dataset.py    # PyTorch Dataset
    ├── augmentations.py     # Video augmentation
    ├── loss_fns.py          # Loss functions (Focal, Dice, cbDice, Supervoxel)
    ├── prepare_data.py      # TIF to PNG conversion
    └── data_utils.py        # Data containers
```

## Hardware Requirements

With feature precomputation enabled (default):

| Hardware | Batch Size | Training Time (5 epochs) | Training Time (30 epochs) |
|----------|------------|--------------------------|---------------------------|
| NVIDIA GPU (8GB+) | 2 | ~55 min | ~5-6 hours |
| Apple M1/M2 (MPS) | 1-2 | ~2-3 hours | ~12-18 hours |
| CPU | 1 | Not recommended | Not recommended |

Without feature precomputation (`--no-precompute`), training is ~10x slower.

## Documentation

- `FINE_TUNING_GUIDE.md` - Complete training guide with troubleshooting
- `CODE_EXPLAINED.md` - Detailed code documentation

## References

- [SAM 2: Segment Anything in Images and Videos](https://github.com/facebookresearch/sam2)
- [LoRA+: Efficient Low Rank Adaptation](https://arxiv.org/abs/2402.12354) (Hayou et al., ICML 2024)
- [cbDice: centerline boundary Dice loss](https://github.com/PengchengShi1220/cbDice) (Shi et al., MICCAI 2024)
- [Supervoxel Loss: Connectivity-Preserving Segmentation](https://arxiv.org/abs/2501.01022) (Grim et al., AAAI 2025)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) (Hu et al., ICLR 2022)
