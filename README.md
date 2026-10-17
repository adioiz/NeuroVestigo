# NeuroVestigo

Topology-preserving segmentation and tracking of single neurons in label-free
phase-contrast time-lapse microscopy.

**NeuroVestigo** is a parameter-efficient adaptation of SAM 2 — dual-rank LoRA+
updating 13.9% of the weights (5.56M of 39.9M) — trained with a
connectivity-preserving objective. Given one annotated mask on frame 0 it
segments an entire multi-day recording, maintaining a single identity
throughout.

This repository holds the training pipeline and the full benchmark. The model
itself, and the napari plug-in that runs it, are in the companion repository
[neurons-labeling](https://github.com/adioiz/neurons-labeling).

## Overview

This repository fine-tunes SAM2 (Segment Anything Model 2) for neuron segmentation using **LoRA+** (Low-Rank Adaptation with asymmetric A/B learning rates) plus topology-preserving losses tailored to tree-like neurite structures.

**The released model is `checkpoints-23rec/lora_weights_topo.pt`** — run 1 of
the paper's three-split repeated hold-out (`src/folds.py`), at its topology
tie-break checkpoint. The same file ships in
[neurons-labeling](https://github.com/adioiz/neurons-labeling) as
`checkpoints-16-frames/lora_weights.pt` (the directory name is historical).
32 labeled recordings, of which the model sees 28 (23 train / 5 val) and never
sees the 4 held-out test, dual-rank LoRA (rank 16 on the
decoder and memory_attention, rank 8 on the last 2 Hiera trunk blocks), the
full memory_attention forward pass with the memory bank left attached to the
graph, and a 4-term loss — focal 3.0, Dice 2.0, cbDice 0.3, supervoxel 0.3 — at
`topology_resolution` 512, over clips of **16** frames, for the full 110
epochs. `checkpoints/lora_weights.pt` is an earlier run on the same split (best
epoch 90, val IoU 0.8399), not the released model.

> **The table in [Results](#results) predates the paper's evaluation.** It
> scores an earlier run — the same configuration over 20 train / 5 val
> recordings, local `checkpoints-16-frames/lora_weights.pt`, best epoch 100 at
> val IoU 0.8377 — on run 1's four test recordings. The paper reports mean ±
> SD over the three splits: those numbers are in `benchmark/results/folds/`
> (regenerate with `benchmark/aggregate_folds.py`), and the released model's
> own run-1 test scores are in `benchmark/results/test/ours_23rec_topo.csv`.

> **Training now distils the prompt pathway.** Frame 0 receives the GT
> mask as a dense prompt and later frames do not, which costs 12% IoU and ~9×
> the Betti-0 error. Each step re-decodes a few frames with their own GT as a
> prompt and trains the ordinary prediction toward that teacher, and frame 0's
> prompt is withheld 40% of the time. Both are **on by default** — see
> [Methods §3b](METHODS_CHEAT_SHEET.md). Checkpoints trained before this,
> the Results table's included, need `use_prompt_distill = False` and
> `prompt_dropout = 0.0` in `src/config.py` to reproduce.

> **`gpu-checkpoints/` no longer exists.** It held the earlier 8-frame,
> `topology_resolution` 256 run. `benchmark/data.py` used to read the
> authoritative split from its `training_config.json` and now reads
> `checkpoints/training_config.json` instead.

> **Read [Correctness fixes](#correctness-fixes) before interpreting any run before Run D.** Three defects — an inert cbDice term, a resampling artefact inside cbDice, and an elastic augmentation that misaligned images from their labels — were found and fixed after most of the run history below was recorded. They do not invalidate reported validation numbers (validation was never augmented), but they do explain why sweeping the loss weights produced differences too small to call.

## Results

Held-out test set: 4 recordings never seen during training or validation, and
removed from the pool *before* the train/val split. Every method is scored by
the same code (`benchmark/score.py` → `benchmark/eval_masks.py`) against the
same ground truth, skipping frame 0 because prompted methods are given it.

| method | prompt | IoU | Dice | clDice | cbDice |
|---|---|---:|---:|---:|---:|
| **This work** (SAM 2 + LoRA+) | mask, frame 0 | 0.7599 | 0.8612 | 0.9432 | 0.8505 |
| SAM 2 zero-shot | mask, frame 0 | 0.6392 | 0.7705 | 0.8301 | 0.7578 |
| Cellpose-SAM | automatic | 0.5629 | 0.7078 | 0.7976 | 0.7245 |
| micro-SAM | box, every frame | 0.5615 | 0.7017 | 0.8041 | 0.7672 |
| SAM 3.1 | 10 points, every frame | 0.4712 | 0.5953 | 0.5921 | 0.6018 |
| NeuroQuantify | automatic | 0.4461 | 0.5941 | 0.6801 | 0.6019 |
| DeepSea | automatic | 0.3350 | 0.4800 | 0.5649 | 0.4313 |
| Omnipose | automatic | 0.2791 | 0.4284 | 0.6492 | 0.5600 |
| SAM 3.1 | text "cells" | 0.2002 | 0.2779 | 0.2901 | 0.2600 |
| Trackastra + DeepSea | automatic | 0.0290 | 0.0462 | 0.0551 | 0.0742 |
| Trackastra + Omnipose | automatic | 0.0111 | 0.0205 | 0.0362 | 0.0702 |

Numbers above are read from `benchmark/results/test/summary.csv`, which is
version-controlled — the per-recording breakdown for each method sits beside it
in the same directory, and `benchmark/results/val/` holds the validation split
used to choose every method's configuration.

**Reading the table.** The prompt column matters more than the ranking. Methods
given a mask on frame 0 start from the exact target; methods given points or
boxes must infer the shape, and a neuron's thin branching arbour is mostly empty
space inside its own bounding box. SAM 3.1 needed ground-truth points on *every*
frame to reach its score — given a single prompt on frame 0 it scores 0.0000,
which is the like-for-like comparison against this work.

Competitor configurations were each chosen on validation and applied unchanged
to test; the sweeps are recorded in `benchmark/sweep_*.py` and, for SAM 3.1, in
[`benchmark/SAM3_RUNBOOK.md`](benchmark/SAM3_RUNBOOK.md).

## Reproducing the benchmark

Every method follows the same three steps: **run → score → read**. Predictions
are written as one `uint8` TIF stack per recording; `score.py` is the only thing
that ever computes a metric, so no method can be scored by a different rule.

```bash
# 1. run a method on the held-out test recordings
#    (run_ours.py defaults to the val split, so name the test recordings)
python benchmark/run_ours.py --videos A71_C1 A81_C1 A12_C3_100k A12_C1_100k_2

# 2. score whatever it wrote (same command for every method)
python benchmark/score.py --pred-dir benchmark/predictions/ours \
    --model ours --split test --cbdice --skip-first-frame

# 3. read the table
column -t -s, benchmark/results/test/summary.csv
```

`--skip-first-frame` is not optional for a fair comparison: prompted methods
are handed frame 0, so scoring it would flatter them over the automatic ones.

**Choosing a configuration.** No method is tuned on test. Each one is swept on
the validation split (`benchmark/sweep_*.py`), the best setting is recorded, and
that setting is applied unchanged to test. `benchmark/results/val/summary.csv`
holds those sweeps, so the choice is auditable rather than asserted.

**Competitors live in isolated environments.** Cellpose, Omnipose, micro-SAM,
DeepSea and SAM 3.1 have mutually incompatible dependencies (SAM 3.1 needs
`numpy<2`; the training stack uses `numpy>=2`). Each runner names the
environment it expects in its docstring — read it before installing anything,
and never install into the training environment.

**For labeling/annotation**, use the companion [neurons-labeling](../neurons-labeling/) repository.

## Features

- **LoRA+ fine-tuning** — 13.9% of parameters trained (5.56M of 39.88M), asymmetric A/B learning rates for faster convergence.
- **Dual-rank LoRA** — rank 16 on 68 Linear layers in mask_decoder + memory_attention, rank 8 on 8 Linear layers in the last 2 Hiera trunk blocks. Inference loaders classify each tensor by regex to apply the right rank.
- **Path B forward pass** — uses SAM 2's real `track_step` API so the memory_attention adapters actually train (Runs 1–4 silently skipped them).
- **Topology-aware loss** — cbDice (centerline boundary Dice) emphasises thin neurite branches; supervoxel loss directly penalises split/merge connectivity errors, concentrating ~4.9M× more gradient on the few pixels that would reconnect a severed branch than on an ordinary pixel. See [what each loss term actually catches](#what-each-loss-term-actually-catches) — the terms cover different failure modes and none sees everything.
- **B200-optimised** — bf16 + channels-last + cuDNN FlashAttention-3 + fused AdamW + (optional) `torch.compile`.
- **Topology metrics, not just overlap** — validation reports Betti-number error (β0 = component count, β1 = holes) alongside IoU/Dice. IoU is blind to a severed neurite; β0 is not.
- **Held-out test set** — 4 recordings are removed from the pool *before* the train/val split and can never leak in, even under the random split.
- **Diagnostics built into the loop** — per-component loss, per-frame IoU, per-video IoU, Betti error, worst-case prediction PNG dumps.
- **Resumable** — `train_state.pt` written every epoch lets a single logical run span multiple SLURM jobs under a 4-hour wall-clock limit.
- **Safe defaults from `config.py`** — CLI flags only override when explicitly passed; nothing silently changes between runs. Every augmentation and loss parameter is echoed in the startup banner.
- **Regression-tested loss and augmentation** — `tests/` pins the properties whose violation caused the three historical defects, and holds a whole-frame reference implementation of the supervoxel detector so future optimisations must prove bit-equality.
- **~0.9 s/step on a B200** — the connectivity detector was quietly consuming about two-thirds of every training step in *every* run before `ba13ff7`; boxing it cut a 110-epoch run from ~29 h to ~5 h with provably identical gradients.

## Repository Connection

This repo is **symlinked** to `neurons-labeling` for seamless data sharing:

```
labeled_masks/        -> ../neurons-labeling/labeled_masks/
labeled_coordinates/  -> ../neurons-labeling/labeled_coordinates/
```

**No manual copying needed** — label in one repo, train in the other.

Training writes its adapters to `checkpoints/lora_weights.pt` (~22 MB: the
LoRA matrices only, not a full SAM 2 checkpoint). The released adapters are run 1's
`checkpoints-23rec/lora_weights_topo.pt`, shipped in
[neurons-labeling](https://github.com/adioiz/neurons-labeling) as
`checkpoints-16-frames/lora_weights.pt`. Loading them requires the base SAM 2 weights, which are
downloaded separately — see
[Step 4](#step-4-use-the-fine-tuned-weights).

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

`neurons-labeling` ships the released model and loads it by default — unless
`../NeuroVestigo/checkpoints/lora_weights.pt` exists, which it prefers as the
most recent training run. To use the released model from this repository, set
`SAM2_LORA_WEIGHTS` to `checkpoints-23rec/lora_weights_topo.pt`. Open the labeling app, check **"Use fine-tuned model"**, click **"SAM2: Propagate from frame 0"**.

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
| `loss_mask_weight` (focal) | 3.0 | pixel-level loss. Saturates by ~epoch 3 (focal down-weights well-classified pixels by design) and contributes ~0.02 of the total thereafter — 3 vs 5 is not a meaningful lever. |
| `loss_dice_weight` | 2.0 | region overlap — carries most of the gradient |
| `loss_cbdice_weight` | 0.3 | skeleton topology. Was 5.0 while the term was inert; after the fixes it is ~20× larger in magnitude, so the weight was reduced. See [Correctness fixes](#correctness-fixes). |
| `use_supervoxel` | True | **re-enabled in Run 5** at small weight (0.3) once Path B made the connectivity signal actually learnable |
| `loss_supervoxel_weight` | 0.3 | small weight. Run 8 tried 2.0 → val IoU fell to 0.8229; the component sits at a structural floor and weight-pumping it does nothing. |
| `loss_iou_weight` | 1.0 | IoU prediction calibration |
| `loss_class_weight` | 1.0 | object-presence classifier |
| `topology_resolution` | 512 | cbDice skeleton + supervoxel connectivity grid. Raised 256 → 512 for finer thin-neurite detail (the loss receives 1024², so 256 is a 4× downsample). Since `ba13ff7` the only real cost here is cbDice (225 → 896 ms/forward at B·T·N=16); the supervoxel side is ~0.02 s/step. Don't lower it for speed — there is nothing left to reclaim. |

**Data (`DataConfig`)**

| field | value | notes |
|---|---|---|
| `image_size` | 1024 | SAM2 native resolution |
| `num_frames` | 8 | clip length (B×T = 16, GPU-friendly) |
| `max_objects` | 3 | per clip |
| `batch_size` | 2 | small for generalization on a small dataset |
| `val_fraction` | 0.25 | fraction held out when `val_videos` is not set |
| `val_videos` | None | when set, overrides val_fraction with an explicit list. **Pin this** — comparing runs against a shifting val set makes every delta meaningless. Standard set: `A91_C1_2, A31_C1, A52_C2, A22_C2, A42_C1` |
| `test_videos` | `A71_C1, A81_C1, A12_C3_100k, A12_C1_100k_2` | held-out **test** set, removed from the pool *before* the train/val split so it cannot leak in even under the random split. Pinning one of these as val raises an error. Reserved for final benchmarking only. |
| `precompute_features` | False | full augmentation pipeline enabled (see `AugmentationConfig`) |
| `clip_stride` | 4 | stride between clip **start** frames. With `num_frames=8` and ~72-frame videos this gives ~17 clips/video and covers every frame with overlap. Stride 2 roughly doubles clips/epoch but the added clips share 6 of 8 frames — redundancy, not new information. |
| `cache_in_memory` | True | all frames + masks cached at startup. Note the printed MB figure counts frames only; masks are stored `int64` and add ~65% more. |

**Augmentation (`AugmentationConfig`)**

All spatial parameters are sampled **once per clip** and applied identically to
every frame and every mask — temporal consistency is the whole point, since
Path B is learning frame-to-frame propagation. Validation uses
`VideoValidationTransform` (resize + normalize only, **no augmentation**), so
validation metrics are never affected by anything in this table.

| field | value | notes |
|---|---|---|
| `horizontal_flip_p` / `vertical_flip_p` | 0.5 / 0.5 | |
| `rotation_degrees` | 25.0 | |
| `shear` | 20.0 | |
| `scale_limit` | (0.85, 1.15) | random zoom |
| `brightness` / `contrast` | 0.15 / 0.15 | |
| `gaussian_blur_p` | 0.3 | sigma ∈ (0.1, 2.0) |
| `gaussian_noise_p` | 0.2 | std ∈ (0.01, 0.05) |
| `elastic_transform_p` | 0.3 | alpha 50.0, sigma 5.0. Set to **0.0** to disable. |

Before commit `76240c8` these eight fields did not exist in the config: they ran
at hardcoded `VideoAugmentation` defaults, were not passed by the trainer, and
never appeared in the banner — which is how a broken elastic warp went unnoticed
for ~10 runs. They are now logged on startup and overridable.

**Training (`TrainingConfig`)**

| field | value | notes |
|---|---|---|
| `num_epochs` | 90 | Runs since Run 7 use `--epochs 110`. Peak val IoU consistently lands around **epoch 72–84**, with the LR-floor tail (epoch ~95+) contributing nothing — early stopping cuts it. |
| `early_stopping_patience` | 10 | validation checks without improvement |
| `val_every_n_epochs` | 2 | |
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
| `--no-cbdice` | False | disable cbDice loss only |
| `--no-supervoxel` | False | disable supervoxel loss only (it is **on** by default at weight 0.3) |
| `--no-topology` | False | disable BOTH cbDice and supervoxel — don't use this for a cbDice ablation, it changes two variables |
| `--supervoxel-weight F` | None (config) | override supervoxel weight |
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

Each epoch prints (Run D best-epoch numbers, epoch 84):

```
Epoch 84/110: train_loss=0.3474, time=~337s, ~1.4s/step
  Validating...
  val_loss=0.2522, val_iou=0.8520, val_dice=0.9188
  val_betti_err: b0=0.476, b1=0.235
  val_loss_components: mask=0.007, dice=0.089, cbdice=0.117, supervoxel=0.059, ...
  per_frame_iou: t0=0.950, t1=0.848, t2=0.835, t3=0.843, t4=0.840, t5=0.832, t6=0.829, t7=0.837
  per_video_iou: A91_C1_2=0.807, A31_C1=0.883, A52_C2=0.861, A22_C2=0.847, A42_C1=0.861
```

**Reading `val_loss_components`:** these are **raw, unweighted** values. The
total applies the config weights, so a component's real contribution is
`raw × weight`. Comparing a raw component across runs with different weights —
or across the cbDice fixes — is not meaningful.

Plus, every validation epoch saves the **6 worst-IoU prediction PNGs** to:

```
checkpoints/val_samples/epoch_NNN/<tag>_rank00_iou0.42_<video>_t6_f47.png
```

Each PNG is `image | GT | prediction | overlay (red=pred, green=GT, yellow=both)`. Open them to see *what's actually failing*, not just an aggregate metric.

## Key metrics — what to look for

| metric | good range | Run D best | meaning |
|---|---|---|---|
| `val_iou` | 0.80–0.86 | 0.8520 | aggregate intersection-over-union across all val frames |
| `val_dice` | 0.88–0.92 | 0.9188 | aggregate Dice across all val frames |
| `val_betti_err` b0 | lower is better | 0.476 | mean \|Δβ0\| — **component-count errors** (breaks, spurious fragments). The metric IoU cannot see: a severed neurite costs ~1% of pixels but is a biological failure. |
| `val_betti_err` b1 | lower is better | 0.235 | mean \|Δβ1\| — hole-count errors |
| per-frame `t0` IoU | 0.90–0.95 | 0.950 | prompt frame — should be high |
| per-frame `t7` IoU | 0.78–0.85 | 0.837 | last frame after propagation chain |
| `t0 - t7` drop | < 0.15 | 0.113 | how much IoU you lose along the chain |
| train/val loss gap | small or closed | — | open gap = overfitting; closed gap can also mean under-fitting |

**Two things worth knowing before you compare runs:**

- **IoU has been saturated at ~0.851 ± 0.002 across five loss configurations**
  (cbDice 5 / 1 / 0.3 / off, plus supervoxel sweeps). Treat any IoU difference
  below ~0.005 as noise, and don't read a config change as an improvement on
  that basis alone.
- **Betti error is noisy per-validation** (±0.05–0.09 observed at constant IoU).
  Compare the **mean over the last ~10 validations**, not the best-epoch value,
  and treat differences under ~0.1 as noise. On that basis, cbDice at 0.3 gave
  mean b0 ≈ 0.51 vs ≈ 0.72 with cbDice disabled — a real separation on topology
  even though IoU was tied.

### What each loss term actually catches

The four terms are not redundant, and none of them sees everything. Measured on
one neuron with a single defect introduced, all terms at their shipped weights:

| defect | missed px | supervoxel | focal | dice | cbdice |
|---|---|---|---|---|---|
| none (perfect) | 0 | 0.0002 | 0.00000 | 0.0032 | 0.0000 |
| **hole** punched in the soma | 4096 | 0.0158 | **0.00781** | **0.0415** | 0.0000 |
| **split** through a branch | 280 | **1.8575** | 0.00053 | 0.0057 | **0.0338** |

The split has **15× fewer wrong pixels** than the hole yet produces **117× more
supervoxel loss** — that is the connectivity terms working as intended. Note
also that the hole scores **zero critical pixels**: the supervoxel loss asks
only "is this in the right number of pieces?", so a hole is invisible to it *by
construction*, and cbDice is near-blind to it too. Holes are caught by focal and
Dice instead.

- **focal + Dice** → area errors: holes, dents, fat/thin boundaries
- **supervoxel + cbDice** → connectivity errors: splits and false bridges

This is why dropping cbDice barely moved IoU but visibly worsened Betti-0, and
why an IoU-only comparison cannot tell you whether a config change helped.

## Project structure

```
NeuroVestigo/
├── cli.py                   # training + data-prep CLI (the entry point)
├── README.md                # ← you are here
├── METHODS_CHEAT_SHEET.md   # what each hyperparameter does, and why
├── CODE_EXPLAINED.md        # file-by-file walkthrough of src/
├── data_manifest.json       # recording name → source TIF + mask TIF
├── requirements.txt
│
├── src/                     # the training pipeline
│   ├── config.py            # ← single source of truth for hyperparameters
│   ├── trainer.py           # training loop, checkpointing, resume
│   ├── lora.py              # LoRA+ adapters, dual-rank targeting
│   ├── loss_fns.py          # focal, Dice, clDice, cbDice, supervoxel
│   ├── evaluate.py          # validation, per-frame/per-video/Betti metrics
│   ├── neuron_dataset.py    # Dataset: clips, prompts, train/val/test split
│   ├── augmentations.py     # clip-consistent video augmentation
│   ├── prepare_data.py      # TIF stacks → PNG frames
│   └── data_utils.py        # batched containers
│
├── benchmark/               # every number in the paper's comparison tables
│   ├── score.py             # ONE scorer for every method → results/
│   ├── eval_masks.py        # IoU / Dice / clDice / cbDice definitions
│   ├── data.py              # split resolution + ground-truth loading
│   ├── run_*.py             # one runner per method (ours, sam2, sam3, …)
│   ├── sweep_*.py           # per-competitor fairness sweeps, on val only
│   ├── SAM3_RUNBOOK.md      # SAM 3.1 setup, sweep, and findings
│   └── results/             # version-controlled scores
│       ├── test/summary.csv #   ← the reported table
│       └── val/summary.csv  #   ← what every configuration was chosen on
│
├── tests/                   # regression tests pinning the historical defects
│   ├── test_cbdice_loss.py          # cbDice is not inert; ordering holds
│   ├── test_cbdice_resize.py        # a perfect prediction costs exactly 0
│   ├── test_augmentation_alignment.py  # image and mask stay aligned
│   ├── test_supervoxel_speed.py     # optimised detector is bit-identical
│   └── test_betti.py                # Betti numbers on known shapes
│
├── checkpoints/             # current run (23 train recordings), gitignored
│   ├── lora_weights.pt      #   best-epoch adapters, 22 MB
│   ├── training_config.json #   read by benchmark/data.py for the splits
│   └── training_log.json    #   per-epoch metrics (LAST resume chunk only)
├── checkpoints-23rec/       # run 1 of the three splits, gitignored
│   └── lora_weights_topo.pt #   THE RELEASED MODEL (Betti-0 tie-break); ships
│                            #   in neurons-labeling as
│                            #   checkpoints-16-frames/lora_weights.pt
│
├── paper_assets/            # manuscript, figures, and their generators
│   ├── paper/main.tex
│   ├── figures/
│   └── make_figures.py
│
├── labeled_masks/           -> ../neurons-labeling/labeled_masks/
└── labeled_coordinates/     -> ../neurons-labeling/labeled_coordinates/
```

Run the tests with:

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python tests/test_cbdice_loss.py
```

(or `pytest tests/` if pytest is installed).

## Correctness fixes

Three defects were found after most of the run history was recorded. Each was
verified by measurement before and after, and each is now pinned by a test.
**Validation was never augmented and never used the training loss, so all
reported val IoU/Dice numbers remain valid** — but the *conclusions* drawn from
loss-weight sweeps do not.

| # | defect | measured impact | fixed in |
|---|---|---|---|
| 1 | **cbDice was inert.** Each skeleton was paired with the weight map of its *own* mask, whose support is zero off-mask, so the precision numerator and denominator were the same sum: `w_tprec ≡ 1.0` for every input. | Loss varied by 3e-6 while Dice fell 0.34. Replaced with a faithful port of the reference implementation; the term now orders correctly (perfect < severed < missing-one < missing-two). | `d4edf6c` |
| 2 | **cbDice resize artefact.** Prediction resized `bilinear`, target `nearest` — a bit-identical pair then disagreed on 100–400+ pixels, scored as topology error. | A **perfect** prediction cost 0.008–0.085 instead of 0. Now exactly 0.0000 at 256/384/512. `topology_resolution` also raised 256 → 512. | `c6b1b03` |
| 3 | **Elastic augmentation misaligned image and label.** `elastic_transform()` sampled a fresh random field per call while the mask used the clip's pre-computed one, so image and label were warped differently — and every frame in a clip got a different warp while all its masks shared one. | Image/label agreement only **IoU 0.668**; ~30% of training clips affected (`elastic_transform_p=0.3`) in every run to date. Now **0.983** (residual is the intended bilinear-vs-nearest boundary difference). | `76240c8` |

**Why this matters for interpretation.** Defects 1 and 2 mean the cbDice weight
sweep (5 / 1 / 0.3 / off) was comparing contaminated signal — cbDice was never
cleanly tested, which is the most likely reason those runs all landed within
noise of each other. Defect 3 is a candidate explanation for the ~0.851 IoU
ceiling itself: ~30% of clips carried labels ~33% wrong relative to their
images, and no loss function can compensate for that. Both are open hypotheses,
each testable with a single run (`elastic_transform_p = 0.0` is the cheaper
and more decisive of the two).

### Performance fix — no behavioural change (`ba13ff7`)

Listed separately on purpose: unlike the three defects above, this one changed
**no number the model ever sees**, so it invalidates nothing and was safe to
drop into a run already in progress.

`_detect_critical_components_2d` scanned the whole frame once per connected
component — a full-image `== comp_id`, dilation and `np.unique` for every
false-negative blob, and a full-image `ndimage_label` for every false-positive
blob. That is O(n_components × H × W), and *both* factors grow with frame area,
so raising `topology_resolution` 256 → 512 in `c6b1b03` cost ~16×, not 4×.

The fix restricts each per-component test to that component's bounding box, and
the false-positive relabel to the parent prediction component's box. Both are
exact — see `CODE_EXPLAINED.md` for why, including the one invariant that makes
it sound (**component IDs are assigned globally, once, and only *read* inside
the box**; re-labelling locally would misjudge a U-shaped neuron whose two tips
enter the same window).

| measured, per training step of 16 masks | before | after |
|---|---|---|
| `topology_resolution=256` | 2.59 s | 0.13 s |
| `topology_resolution=512` | 39.78 s | **0.32 s** |
| loss value | `0.1688096225` | `0.1688096225` |
| gradient | — | identical, `max\|Δ\|` = **0.000e+00** |

Verified bit-identical against the pre-fix implementation on 155 cases (neuron
shapes at five resolutions, 90 pure-random masks at five densities,
border-touching components, empty/full degenerates), and the critical-pixel
counts match too — so it did not go quietly inert. Pinned by
`tests/test_supervoxel_speed.py`, which keeps a whole-frame reference and
asserts equality on every run.

**The finding worth carrying forward:** the old detector was costing roughly
**1.3 of Run D's 1.9 s/step** — about two-thirds of every training step, at 256,
before the 512 change existed. The `topology_resolution` bump did not create
this bottleneck, it only made it impossible to ignore. **Every run in the
history above paid this tax.** Results are unaffected; wall-clock was not.

## Hardware

The current config targets a single B200 (or any modern NVIDIA GPU with ≥8 GB):

Measured on a single B200, 25 videos, 176 steps/epoch, no `torch.compile`:

| code state | s/step | epoch | 110-epoch wall clock |
|---|---|---|---|
| Run D — old detector, `topology_resolution=256` | 1.9 | ~335 s | ~10 h → 3 chained jobs |
| old detector, `topology_resolution=512` (`c6b1b03`) | 5.0 | ~950 s | ~29 h → ~8 chained jobs |
| **current — boxed detector, 512 (`ba13ff7`)** | **~0.9** | **~165 s** | **~5 h → 2 chained jobs** |

(The current row is measured over the first 100 steps of a live run; the other
two are full-epoch averages from `training_log.json`.)

Other hardware:

| GPU | relative |
|---|---|
| A100 | ~2× slower; not benchmarked end-to-end |
| Apple M-series (MPS) | seconds-to-tens-of-seconds per step — not realistic for the full Path B forward |

`topology_resolution=512` is now essentially free: after `ba13ff7` the
supervoxel detector costs ~0.02 s/step at 512, well under what it cost at 256
beforehand. The remaining topology cost is cbDice (~0.3–0.5 s/step at 512),
which is the term the 512 setting was actually raised for.

`--compile-encoder --compile-decoder` shaves ~10–25% off step time after the first-call compile cost (30–90 s).

## Documentation

Start with whichever question you actually have:

| you want to… | read |
|---|---|
| understand a hyperparameter or loss term | [`METHODS_CHEAT_SHEET.md`](METHODS_CHEAT_SHEET.md) |
| find where something is implemented | [`CODE_EXPLAINED.md`](CODE_EXPLAINED.md) |
| reproduce a number in the comparison table | [Reproducing the benchmark](#reproducing-the-benchmark) |
| know how SAM 3.1 was set up and swept | [`benchmark/SAM3_RUNBOOK.md`](benchmark/SAM3_RUNBOOK.md) |
| read the manuscript itself | `paper_assets/paper/main.tex` |

The planning notes under `paper_assets/*.md` are working documents kept for
provenance, not maintained references.

## References

- [SAM 2: Segment Anything in Images and Videos](https://github.com/facebookresearch/sam2)
- [LoRA+: Efficient Low Rank Adaptation](https://arxiv.org/abs/2402.12354) (Hayou et al., ICML 2024)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) (Hu et al., ICLR 2022)
- [cbDice: centerline boundary Dice loss](https://github.com/PengchengShi1220/cbDice) (Shi et al., MICCAI 2024)
- [Focal Loss](https://arxiv.org/abs/1708.02002) (Lin et al., ICCV 2017)
- [FlashAttention-3](https://tridao.me/blog/2024/flash3/) (Shah et al., 2024)
