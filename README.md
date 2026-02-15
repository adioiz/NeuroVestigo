# SAM2 Neuron Fine-Tuning

LoRA fine-tuning pipeline for SAM2 on neuron microscopy data.

## Overview

This repository provides a training pipeline to fine-tune SAM2 (Segment Anything Model 2) for neuron segmentation using LoRA (Low-Rank Adaptation). Fine-tuning improves segmentation accuracy for your specific microscopy domain.

**For labeling/annotation**, use the companion [neurons-labeling](../neurons-labeling/) repository.

## Features

- **LoRA fine-tuning**: Efficient training (~0.5% of parameters)
- **clDice loss**: Skeleton topology-preserving loss for tubular structures
- **Supervoxel loss**: Connectivity-preserving loss for split/merge errors (NEW!)
- **Strong augmentations**: Elastic deformations, blur, noise
- **Early stopping**: Automatic stopping when metrics plateau
- **MPS support**: Works on Apple Silicon

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

### Step 1: Create Manifest

Create a `manifest.json` file listing your TIF stacks. Use relative paths via symlinks:

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
python train.py prepare --manifest manifest.json --output-dir training_data
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
# Basic training (recommended for GPU)
python train.py train --epochs 30

# With clDice for thin structures (requires more memory)
python train.py train --epochs 30 --batch-size 2

# For Apple Silicon (MPS) if OOM occurs
python train.py train --epochs 30 --batch-size 1 --no-cldice

# Full customization
python train.py train \
    --epochs 30 \
    --batch-size 2 \
    --lora-rank 8 \
    --lora-lr 1e-4 \
    --patience 10 \
    --val-every 2
```

Training outputs:
```
Epoch 1/30: 100%|██████| 50/50 [02:30<00:00]
  train_loss: 1.234  loss_mask: 0.456  loss_dice: 0.234

Validation (epoch 2):
  val_loss: 0.987  val_iou: 0.654  val_dice: 0.754

Best model saved! IoU: 0.654
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
| `--lora-lr` | 1e-4 | LoRA learning rate |
| `--base-lr` | 1e-5 | Learning rate for non-LoRA params |
| `--warmup-epochs` | 1 | Warmup epochs before cosine decay |
| `--num-frames` | 8 | Frames per training clip |
| `--patience` | 10 | Early stopping patience |
| `--val-every` | 2 | Validation frequency (epochs) |
| `--val-samples` | 0 | Max validation batches (0=all, >0=faster) |
| `--no-cldice` | False | Disable clDice loss to save memory |
| `--supervoxel-weight` | 1.0 | Supervoxel connectivity loss weight |
| `--no-supervoxel` | False | Disable supervoxel loss |
| `--no-topology` | False | Disable BOTH clDice and supervoxel (fast CPU/MPS training) |

## Key Metrics

| Metric | Good Range | Meaning |
|--------|------------|---------|
| `val_iou` | 0.65-0.85 | Intersection over Union (higher = better) |
| `val_dice` | 0.75-0.90 | Dice coefficient (higher = better) |
| Train/Val gap | < 0.1 | Low gap = not overfitting |

## Troubleshooting

### Out of Memory (OOM) or Slow Training
```bash
# Reduce batch size
python train.py train --batch-size 1

# Disable ALL topology losses (fastest for CPU/MPS)
python train.py train --no-topology

# Or disable only clDice
python train.py train --no-cldice

# Or reduce frames per clip
python train.py train --num-frames 4
```

### Broken Neuron Branches
If thin structures break:
1. Enable clDice (don't use `--no-cldice`)
2. Enable supervoxel loss (default: enabled)
3. Increase weights for connectivity losses:
   ```bash
   python train.py train --supervoxel-weight 2.0
   ```
   Or edit `src/training/config.py`: `loss_cldice_weight = 2.0`, `loss_supervoxel_weight = 2.0`

### Validation Too Slow
```bash
# Limit to 10 validation batches
python train.py train --val-samples 10
```

## Project Structure

```
Fine-tune-SAM2-model-for-Neurons-segmentation/
├── train.py                 # Training CLI
├── requirements.txt
├── README.md
├── CODE_EXPLAINED.md        # Detailed code documentation
├── FINE_TUNING_GUIDE.md     # Complete training guide
├── manifest.json            # Your manifest file
├── labeled_masks/           # Symlink -> ../neurons-labeling/labeled_masks
├── labeled_coordinates/     # Symlink -> ../neurons-labeling/labeled_coordinates
├── checkpoints/             # Saved model weights (source of truth)
│   └── lora_weights.pt
├── training_data/           # Prepared PNG data
└── src/
    └── training/
        ├── config.py        # Hyperparameters
        ├── lora.py          # LoRA implementation
        ├── train.py         # Training loop
        ├── evaluate.py      # Validation metrics
        ├── neuron_dataset.py # PyTorch Dataset
        ├── augmentations.py # Video augmentation
        ├── loss_fns.py      # Loss functions (Focal, Dice, clDice, Supervoxel)
        ├── prepare_data.py  # TIF to PNG conversion
        └── data_utils.py    # Data containers
```

## Hardware Requirements

| Hardware | Batch Size | Training Time (30 epochs) |
|----------|------------|---------------------------|
| NVIDIA GPU (8GB+) | 2 | ~1-2 hours |
| Apple M1/M2 (MPS) | 1-2 | ~2-4 hours |
| CPU | 1 | Not recommended |

## Documentation

- `FINE_TUNING_GUIDE.md` - Complete training guide with troubleshooting
- `CODE_EXPLAINED.md` - Detailed code documentation

## References

- [SAM 2: Segment Anything in Images and Videos](https://github.com/facebookresearch/sam2)
- [clDice: A Novel Topology-Preserving Loss Function](https://arxiv.org/abs/2003.07311) (Shit et al., CVPR 2021)
- [Supervoxel Loss: Connectivity-Preserving Segmentation](https://arxiv.org/abs/2501.01022) (Grim et al., AAAI 2025)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
