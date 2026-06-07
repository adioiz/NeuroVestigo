# SAM2 Neuron Fine-Tuning

LoRA+ fine-tuning pipeline for SAM2 on neuron microscopy data, tuned for
NVIDIA B200 (Blackwell) and small datasets (~20 videos).

## Overview

This repository fine-tunes SAM2 (Segment Anything Model 2) for neuron segmentation using **LoRA+** (Low-Rank Adaptation with asymmetric A/B learning rates) plus topology-preserving losses tailored to tree-like neurite structures.

The shipping configuration is **Run 7**: 20 videos (15 train / 5 val), dual-rank LoRA (rank 16 on decoder + memory_attention, rank 8 on the last 2 Hiera trunk blocks), the full memory_attention forward pass ("Path B"), and a 6-component loss including cbDice and supervoxel. Run 7 reaches **val IoU 0.8401 / Dice 0.9117** at epoch 88.

**For labeling/annotation**, use the companion [neurons-labeling](../neurons-labeling/) repository.

## Features

- **LoRA+ fine-tuning** — 13.9% of parameters trained (5.56M of 39.88M), asymmetric A/B learning rates for faster convergence.
- **Dual-rank LoRA** — rank 16 on 68 Linear layers in mask_decoder + memory_attention, rank 8 on 8 Linear layers in the last 2 Hiera trunk blocks. Inference loaders classify each tensor by regex to apply the right rank.
- **Path B forward pass** — uses SAM 2's real `track_step` API so the memory_attention adapters actually train (Runs 1–4 silently skipped them).
- **Topology-aware loss** — cbDice (centerline boundary Dice) emphasises thin neurite branches; supervoxel loss directly penalises split/merge connectivity errors.
- **B200-optimised** — bf16 + channels-last + cuDNN FlashAttention-3 + fused AdamW + (optional) `torch.compile`.
- **Diagnostics built into the loop** — per-component loss, per-frame IoU, per-video IoU, worst-case prediction PNG dumps.
- **Resumable** — `train_state.pt` written every epoch lets a single logical run span multiple SLURM jobs under a 4-hour wall-clock limit.
- **Safe defaults from `config.py`** — CLI flags only override when explicitly passed; nothing silently changes between runs.

## Repository Connection

This repo is **symlinked** to `neurons-labeling` for seamless data sharing:

```
labeled_masks/        -> ../neurons-labeling/labeled_masks/
labeled_coordinates/  -> ../neurons-labeling/labeled_coordinates/
```

The labeling repo also symlinks back here:

```
neurons-labeling/checkpoints/ -> ../Fine-tune-SAM2-model-for-Neurons-segmentation/checkpoints/
```

**No manual copying needed** — label in one repo, train in the other.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/sam2.git
```

## Quick Start

### Step 1: Create the data manifest

```json
[
    {"image_tif": "/path/to/video_1.tif", "mask_tif": "labeled_masks/video_1_masks.tif", "name": "video_1"},
    {"image_tif": "/path/to/video_2.tif", "mask_tif": "labeled_masks/video_2_masks.tif", "name": "video_2"}
]
```

`mask_tif` paths use the symlink to `../neurons-labeling/labeled_masks/`.

### Step 2: Prepare data (TIF stacks → PNGs)

```bash
python cli.py prepare --manifest data_manifest.json --output-dir training_data
```

### Step 3: Train

The simplest command — uses everything in `src/config.py`:

```bash
python cli.py train
```

For maximum throughput on B200, add the optional compile flags:

```bash
python cli.py train --compile-encoder --compile-decoder
```

To pin a specific validation split (e.g. for val-swap experiments):

```bash
python cli.py train --val-videos A91_C1_2,A31_C1,A52_C2,A22_C2,A42_C1
```

The five videos above are the Run 7 default val split.

### Step 4: Use the fine-tuned weights

The `lora_weights.pt` checkpoint is automatically available in `neurons-labeling` via symlink. Open the labeling app, check **"Use fine-tuned model"**, click **"SAM2: Propagate from frame 0"**.

## Config — `src/config.py` is the source of truth

CLI flags default to `None` and only override config when you pass them explicitly. So the most reproducible workflow is to edit `src/config.py` and run `python cli.py train` with no flags. The startup banner echoes the resolved config so the SLURM log makes it obvious which values ran.

### Current defaults

**LoRA+ (`LoRAConfig`)**

| field | value | notes |
|---|---|---|
| `rank` | 16 | LoRA rank for decoder + memory_attention layers |
| `alpha` | 32.0 | scale = α/r = 2.0 |
| `dropout` | 0.10 | adapter dropout |
| `hiera_last_stage` | True | **Run 6 addition** — also LoRA-adapt the last 2 Hiera trunk blocks (deepest encoder features). 8 additional Linear layers wrapped. |
| `hiera_rank` | 8 | smaller rank for the wider 768-dim Hiera layers |
| `hiera_alpha` | 16.0 | scale = α/r = 2.0 (consistent) |

This is a **dual-rank checkpoint**: the main group is rank 16, the Hiera group is rank 8. Inference loaders must regex-classify each saved tensor against `SAM2_LORA_TARGETS` vs `HIERA_LAST_STAGE_LORA_TARGETS` to pick the matching rank when rebuilding the adapters. Together: 5,557,250 trainable parameters of 39,876,930 (13.9%).

**Optimizer (`OptimizerConfig`)**

| field | value | notes |
|---|---|---|
| `lora_lr` | 1e-4 | LoRA-A peak LR |
| `lora_b_lr_ratio` | 16.0 | LoRA-B peak LR = 1.6e-3 (LoRA+ asymmetric) |
| `base_lr` | 1e-5 | unfrozen embeddings (10× lower — protects pretrained weights) |
| `weight_decay` | 0.05 | AdamW decoupled weight decay |
| `max_grad_norm` | 1.0 | gradient clipping |
| `warmup_epochs` | 2 | linear warmup before cosine decay |
| `use_cosine_schedule` | True | GPT-3-style cosine annealing |
| `min_lr_ratio` | 0.05 | LR floor at 5% of peak |

**Loss (`LossConfig`)**

| field | value | notes |
|---|---|---|
| `loss_mask_weight` (focal) | 5.0 | pixel-level loss |
| `loss_dice_weight` | 2.0 | region overlap |
| `loss_cbdice_weight` | 5.0 | skeleton topology — strengthened in Run 5 to push connectivity in tree-like neurons |
| `use_supervoxel` | True | **re-enabled in Run 5** at small weight (0.3) once Path B made the connectivity signal actually learnable |
| `loss_supervoxel_weight` | 0.3 | small weight; diagnostics confirmed the per-component value actually moves under Path B |
| `loss_iou_weight` | 1.0 | IoU prediction calibration |
| `loss_class_weight` | 1.0 | object-presence classifier |
| `topology_resolution` | 256 | cbDice skeleton + supervoxel connectivity are computed at 256×256 (16× fewer pixels than 1024); ~2000× faster than full-res with minimal topology quality loss |

**Data (`DataConfig`)**

| field | value | notes |
|---|---|---|
| `image_size` | 1024 | SAM2 native resolution |
| `num_frames` | 8 | clip length (B×T = 16, GPU-friendly) |
| `max_objects` | 3 | per clip |
| `batch_size` | 2 | small for generalization on a small dataset |
| `val_fraction` | 0.25 | 5 of 20 videos held out (stable mean) |
| `val_videos` | None | when set, overrides val_fraction with explicit list. Run 7 default: `A91_C1_2, A31_C1, A52_C2, A22_C2, A42_C1` |
| `precompute_features` | False | full augmentation pipeline (rotation, shear, brightness, contrast) is enabled |
| `clip_stride` | 4 | dataset stride between clip starts |
| `cache_in_memory` | True | all frames cached at startup |

**Training (`TrainingConfig`)**

| field | value | notes |
|---|---|---|
| `num_epochs` | 90 | Run 7: 20 videos, dual-rank LoRA, 5% LR floor — ~6.5 h on a single B200 |
| `use_memory_attention_path` | True | use SAM 2's real `track_step` API for temporal propagation (Path B). Runs 1–4 silently skipped memory_attention; Run 5 turned this on and jumped val IoU from ~0.7755 to ~0.8295 |

**Forward-pass architecture**

The training wrapper defaults to SAM 2's full memory-attention pipeline (**Path B**, introduced in Run 5). On each frame `t`, the wrapper calls `model.track_step(...)` which:
1. Conditions current-frame features on past memory entries via `memory_attention` — this is where the 40 LoRA adapters on memory_attention finally receive gradients.
2. Runs the SAM mask decoder (GT mask as dense prompt on `t=0`, no prompt on `t≥1` — memory provides the context).
3. Encodes the prediction back into the memory bank via `memory_encoder` for future frames.

The simplified previous-mask-as-prompt path (used in Runs 1–4) is retained as `_forward_simple()` and selectable via `--no-memory-attention` for A/B comparison.

**Critical footgun: the obj-score nuke.** When `use_memory_path=True`, the wrapper sets `self.model.pred_obj_scores = False` and `self.model.use_mask_input_as_output_without_sam = False`. Without this, `SAM2Base._forward_sam_heads` replaces masks with `NO_OBJ_SCORE` (effectively `-inf`) whenever its untrained obj-score head outputs <0. With random near-zero scores at init, predictions silently went to all-zero, dice → 1.0 trivially, val IoU = 0.0, and no gradient flowed. This bug masqueraded as training succeeding for several runs before Run 5 diagnosed and fixed it. Anyone touching the wrapper should preserve those two flags.

**Performance (auto-applied on CUDA)**

- Mixed precision: bfloat16 (no GradScaler — bf16 doesn't need it).
- Channels-last memory format on the encoder + decoder.
- Pinned memory + non_blocking H2D copies.
- cuDNN SDPA backend enabled — FlashAttention-3 on B200.
- Fused AdamW kernel.
- Per-step `.item()` syncs deferred to logging boundaries (every 10 steps).
- cbDice scipy distance transforms parallelized via `ThreadPoolExecutor`.

### CLI flags

| flag | default | what it does |
|---|---|---|
| `--epochs N` | None (config) | override num_epochs |
| `--batch-size N` | None (config) | override batch_size |
| `--lora-rank N` | None (config) | override LoRA rank |
| `--lora-alpha F` | None (config) | override LoRA alpha |
| `--lora-lr F` | None (config) | override LoRA-A learning rate |
| `--lora-b-lr-ratio F` | None (config) | override LoRA+ B/A ratio |
| `--base-lr F` | None (config) | override embeddings learning rate |
| `--warmup-epochs N` | None (config) | override warmup duration |
| `--num-frames N` | None (config) | override clip length |
| `--data-dir DIR` | None (config) | override training_data dir |
| `--checkpoint-dir DIR` | None (config) | override checkpoint output dir |
| `--val-every N` | None (config) | override val_every_n_epochs |
| `--val-samples N` | None (config) | max val batches (0 = all) |
| `--val-videos A,B,C` | None (config) | **explicit val split** — overrides val_fraction |
| `--patience N` | None (config) | override early stopping patience |
| `--clip-stride N` | None (config) | override clip stride |
| `--no-precompute` | False | force precompute_features=False |
| `--no-cache` | False | force cache_in_memory=False |
| `--no-cbdice` | False | disable cbDice loss |
| `--no-supervoxel` | False | disable supervoxel loss (already off by default) |
| `--no-topology` | False | disable BOTH cbDice and supervoxel |
| `--supervoxel-weight F` | None (config) | override supervoxel weight (only if you re-enable it) |
| `--use-imagenet-stats` | False | use ImageNet mean/std instead of dataset stats.json |
| `--compile-encoder` | False | `torch.compile` the image encoder |
| `--compile-decoder` | False | `torch.compile` the SAM mask decoder |
| `--no-memory-attention` | False | fall back to legacy simplified propagation (A/B comparison only) |
| `--no-hiera-last-stage` | False | disable LoRA on the Hiera last stage (A/B vs encoder-frozen baseline) |
| `--resume PATH` | None | resume from a `train_state.pt` checkpoint (chain SLURM jobs under a wall-clock limit) |

## Chained training across SLURM jobs

Single-job training is capped by the cluster's wall-clock limit (e.g. 4 hours on the `B200-4h` partition). At Run 7 step times (~260 s/epoch on B200), a 90-epoch run is ~6.5 h and so needs **2 chained jobs**. The `--resume` flag lets you split one logical training run across multiple jobs.

How it works:
- At the end of every epoch the trainer writes `checkpoints/train_state.pt` (model + optimiser + scheduler + counters).
- Submitting a job with `--resume checkpoints/train_state.pt` restores everything and continues from `saved_epoch + 1`.
- `lora_weights.pt` (best-IoU inference checkpoint) is unchanged and unaffected.

Minimal sbatch pattern (auto-detects whether to resume or start fresh):

```bash
STATE_FILE=checkpoints/train_state.pt
if [ -f "$STATE_FILE" ]; then
    python -u cli.py train --epochs 90 --resume "$STATE_FILE"
else
    python -u cli.py train --epochs 90
fi
```

Submit the script repeatedly until `num_epochs` is reached (or early stopping triggers). To begin a **new** training run later, delete `train_state.pt` first — otherwise the script silently resumes the old one.

## What gets logged

Each epoch prints (Run 7 best-epoch numbers, epoch 88):

```
Epoch 88/90: train_loss=..., time=~260s, ~3.4s/step
  Validating...
  val_loss=..., val_iou=0.8401, val_dice=0.9117
  val_loss_components: mask=0.009, dice=0.096, cbdice=0.004, supervoxel=0.064, ...
  per_frame_iou: t0=0.946, t1=0.837, t2=0.825, t3=0.828, t4=0.827, t5=0.819, t6=0.818, t7=0.822
  per_video_iou: A91_C1_2=0.793, A31_C1=0.873, A52_C2=0.844, A22_C2=0.840, A42_C1=0.850
```

Plus, every validation epoch saves the **6 worst-IoU prediction PNGs** to:

```
checkpoints/val_samples/epoch_NNN/<tag>_rank00_iou0.42_<video>_t6_f47.png
```

Each PNG is `image | GT | prediction | overlay (red=pred, green=GT, yellow=both)`. Open them to see *what's actually failing*, not just an aggregate metric.

## Key metrics — what to look for

| metric | good range | Run 7 best | meaning |
|---|---|---|---|
| `val_iou` | 0.80–0.85 | 0.8401 | aggregate intersection-over-union across all val frames |
| `val_dice` | 0.88–0.92 | 0.9117 | aggregate Dice across all val frames |
| per-frame `t0` IoU | 0.90–0.95 | 0.946 | prompt frame — should be high |
| per-frame `t7` IoU | 0.78–0.85 | 0.822 | last frame after propagation chain |
| `t0 - t7` drop | < 0.15 | 0.124 | how much IoU you lose along the chain |
| train/val loss gap | small or closed | — | open gap = overfitting; closed gap can also mean under-fitting |

## Project structure

```
Fine-tune-SAM2-model-for-Neurons-segmentation/
├── cli.py                   # Training CLI
├── data_manifest.json
├── requirements.txt
├── README.md                # ← you are here
├── METHODS_CHEAT_SHEET.md   # quick technical reference
├── FINE_TUNING_GUIDE.md     # full pedagogical guide
├── CODE_EXPLAINED.md        # file-by-file walkthrough
├── labeled_masks/           # symlink → ../neurons-labeling/labeled_masks
├── labeled_coordinates/     # symlink → ../neurons-labeling/labeled_coordinates
├── checkpoints/
│   ├── lora_weights.pt
│   ├── training_log.json
│   └── val_samples/epoch_NNN/    # worst-IoU prediction dumps
├── training_data/
│   └── stats.json
└── src/
    ├── config.py            # ← single source of truth for hyperparameters
    ├── lora.py              # LoRA+ adapter implementation
    ├── trainer.py           # training loop
    ├── evaluate.py          # validation + per-frame/per-video metrics
    ├── neuron_dataset.py    # PyTorch Dataset
    ├── augmentations.py     # video augmentation
    ├── loss_fns.py          # focal, dice, cbDice, supervoxel
    ├── prepare_data.py      # TIF → PNG conversion
    └── data_utils.py        # batched data containers
```

## Hardware

The current config targets a single B200 (or any modern NVIDIA GPU with ≥8 GB):

| GPU | epoch time (Run 7, no compile) | 90-epoch wall clock |
|---|---|---|
| B200 | ~260 s/epoch | ~6.5 h → 2 chained 4-hour SLURM jobs |
| A100 | ~2× slower | not benchmarked end-to-end at Run 7 scale |
| Apple M-series (MPS) | seconds-to-tens-of-seconds per step | not realistic for the full Path B forward |

`--compile-encoder --compile-decoder` shaves ~10–25% off step time after the first-call compile cost (30–90 s).

## Documentation

- `METHODS_CHEAT_SHEET.md` — concise technical reference, current defaults
- `FINE_TUNING_GUIDE.md` — full pedagogical guide
- `CODE_EXPLAINED.md` — file-by-file walkthrough

## References

- [SAM 2: Segment Anything in Images and Videos](https://github.com/facebookresearch/sam2)
- [LoRA+: Efficient Low Rank Adaptation](https://arxiv.org/abs/2402.12354) (Hayou et al., ICML 2024)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) (Hu et al., ICLR 2022)
- [cbDice: centerline boundary Dice loss](https://github.com/PengchengShi1220/cbDice) (Shi et al., MICCAI 2024)
- [Focal Loss](https://arxiv.org/abs/1708.02002) (Lin et al., ICCV 2017)
- [FlashAttention-3](https://tridao.me/blog/2024/flash3/) (Shah et al., 2024)
