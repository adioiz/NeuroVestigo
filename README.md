# SAM2 Neuron Fine-Tuning

LoRA+ fine-tuning pipeline for SAM2 on neuron microscopy data, tuned for
NVIDIA B200 (Blackwell) and small datasets (~20 videos).

## Overview

This repository fine-tunes SAM2 (Segment Anything Model 2) for neuron segmentation using **LoRA+** (Low-Rank Adaptation with asymmetric A/B learning rates) plus topology-preserving losses tailored to tree-like neurite structures.

The current shipping configuration is **Run D**: 25 labeled recordings (20 train / 5 val / 4 held-out test), dual-rank LoRA (rank 16 on decoder + memory_attention, rank 8 on the last 2 Hiera trunk blocks), the full memory_attention forward pass ("Path B"), and a 6-component loss including cbDice and supervoxel. Run D reaches **val IoU 0.8520 / Dice 0.9188** at epoch 84, up from Run 7's 0.8401.

> **Read [Correctness fixes](#correctness-fixes) before interpreting any run before Run D.** Three defects — an inert cbDice term, a resampling artefact inside cbDice, and an elastic augmentation that misaligned images from their labels — were found and fixed after most of the run history below was recorded. They do not invalidate reported validation numbers (validation was never augmented), but they do explain why sweeping the loss weights produced differences too small to call.

**For labeling/annotation**, use the companion [neurons-labeling](../neurons-labeling/) repository.

## Features

- **LoRA+ fine-tuning** — 13.9% of parameters trained (5.56M of 39.88M), asymmetric A/B learning rates for faster convergence.
- **Dual-rank LoRA** — rank 16 on 68 Linear layers in mask_decoder + memory_attention, rank 8 on 8 Linear layers in the last 2 Hiera trunk blocks. Inference loaders classify each tensor by regex to apply the right rank.
- **Path B forward pass** — uses SAM 2's real `track_step` API so the memory_attention adapters actually train (Runs 1–4 silently skipped them).
- **Topology-aware loss** — cbDice (centerline boundary Dice) emphasises thin neurite branches; supervoxel loss directly penalises split/merge connectivity errors.
- **B200-optimised** — bf16 + channels-last + cuDNN FlashAttention-3 + fused AdamW + (optional) `torch.compile`.
- **Topology metrics, not just overlap** — validation reports Betti-number error (β0 = component count, β1 = holes) alongside IoU/Dice. IoU is blind to a severed neurite; β0 is not.
- **Held-out test set** — 4 recordings are removed from the pool *before* the train/val split and can never leak in, even under the random split.
- **Diagnostics built into the loop** — per-component loss, per-frame IoU, per-video IoU, Betti error, worst-case prediction PNG dumps.
- **Resumable** — `train_state.pt` written every epoch lets a single logical run span multiple SLURM jobs under a 4-hour wall-clock limit.
- **Safe defaults from `config.py`** — CLI flags only override when explicitly passed; nothing silently changes between runs. Every augmentation and loss parameter is echoed in the startup banner.
- **Regression-tested loss and augmentation** — `tests/` pins the properties whose violation caused the three historical defects.

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
| `loss_mask_weight` (focal) | 3.0 | pixel-level loss. Saturates by ~epoch 3 (focal down-weights well-classified pixels by design) and contributes ~0.02 of the total thereafter — 3 vs 5 is not a meaningful lever. |
| `loss_dice_weight` | 2.0 | region overlap — carries most of the gradient |
| `loss_cbdice_weight` | 0.3 | skeleton topology. Was 5.0 while the term was inert; after the fixes it is ~20× larger in magnitude, so the weight was reduced. See [Correctness fixes](#correctness-fixes). |
| `use_supervoxel` | True | **re-enabled in Run 5** at small weight (0.3) once Path B made the connectivity signal actually learnable |
| `loss_supervoxel_weight` | 0.3 | small weight. Run 8 tried 2.0 → val IoU fell to 0.8229; the component sits at a structural floor and weight-pumping it does nothing. |
| `loss_iou_weight` | 1.0 | IoU prediction calibration |
| `loss_class_weight` | 1.0 | object-presence classifier |
| `topology_resolution` | 512 | cbDice skeleton + supervoxel connectivity grid. Raised 256 → 512 for finer thin-neurite detail (the loss receives 1024², so 256 is a 4× downsample). Costs 4× the topology compute — measured 225 → 896 ms/forward at B·T·N=16. |

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
├── benchmark/               # competitor comparison + test-set evaluation
│   ├── data.py              # TEST_VIDEOS definition
│   └── eval_masks.py        # reported IoU / Dice / clDice / cbDice metrics
├── tests/                   # regression tests for the historical defects
│   ├── test_cbdice_loss.py         # cbDice term is not inert; ordering holds
│   ├── test_cbdice_resize.py       # perfect prediction costs exactly 0
│   ├── test_augmentation_alignment.py  # image and mask stay aligned
│   └── test_betti.py               # Betti numbers on known shapes
└── src/
    ├── config.py            # ← single source of truth for hyperparameters
    ├── lora.py              # LoRA+ adapter implementation
    ├── trainer.py           # training loop
    ├── evaluate.py          # validation + per-frame/per-video/Betti metrics
    ├── neuron_dataset.py    # PyTorch Dataset
    ├── augmentations.py     # video augmentation
    ├── loss_fns.py          # focal, dice, cbDice, supervoxel
    ├── prepare_data.py      # TIF → PNG conversion
    └── data_utils.py        # batched data containers
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

## Hardware

The current config targets a single B200 (or any modern NVIDIA GPU with ≥8 GB):

| GPU | epoch time (Run D, 25 videos, no compile) | 110-epoch wall clock |
|---|---|---|
| B200 | ~335 s/epoch (~1.4 s/step, 176 steps) | ~10 h → 3 chained 4-hour SLURM jobs |
| A100 | ~2× slower | not benchmarked end-to-end |
| Apple M-series (MPS) | seconds-to-tens-of-seconds per step | not realistic for the full Path B forward |

`topology_resolution=512` costs 4× the topology compute versus 256 (225 → 896 ms
per cbDice forward at B·T·N=16), so budget more chained jobs than the table
above until you have measured your own step time.

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
