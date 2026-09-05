# NeuroVestigo — Training Methods Cheat Sheet

What each hyperparameter and loss term does, and why it is set where it is.
Training writes its adapters to `checkpoints/lora_weights.pt`. The model the
Results table was measured with is the earlier sixteen-frame run, whose weights
ship in [neurons-labeling](https://github.com/adioiz/neurons-labeling) at
`checkpoints-16-frames/lora_weights.pt`.

**Project:** Fine-tuning SAM 2 (Segment Anything Model 2) for neuron time-lapse microscopy segmentation, using LoRA+ on a small dataset (32 recordings: 23 train / 5 val / 4 held-out test, i.e. 28 seen by
the model), tuned for NVIDIA B200.

---

## 1. Architecture: What's Frozen vs. Trained

```
Frame t ──► [Image Encoder (Hiera-tiny)] ──► Features ──► [Memory Attention] ──► [Mask Decoder] ──► Mask
              FROZEN base + LoRA on                         LoRA-adapted          LoRA-adapted
              last 2 trunk blocks (Run 6)                + memory_encoder    + unfrozen embeddings
```

- **Frozen base:** Image encoder (Hiera backbone) base weights stay frozen — but as of Run 6 the **last 2 trunk blocks** are LoRA-adapted on top. Adapter trains, base weights don't. 8 layers wrapped (attn.qkv, attn.proj, mlp.layers.{0,1} × 2 blocks).
- **LoRA-adapted (main):** Mask decoder transformer (self-attn + cross-attn) and `memory_attention` layers (Q, K, V, Out projections + FFN linears) — 68 layers at rank 16.
- **LoRA-adapted (encoder, Run 6):** Last 2 Hiera trunk blocks (the deepest, 768-dim features) — 8 layers at rank 8.
- **Unfrozen embeddings:** `maskmem_tpos_enc`, `no_mem_embed`, `no_obj_ptr`, `obj_ptr_proj`, `obj_ptr_tpos_proj` (12 small parameter tensors). These control temporal reasoning and object tracking.

**Forward path (`use_memory_attention_path=True`, default since Run 5):** the wrapper now calls SAM 2's real `track_step(...)` API per frame. On frame `t`:
1. `memory_attention` conditions current-frame features on past memory bank entries — the 40 LoRA adapters wrapped around memory_attention now actually receive gradients.
2. The SAM mask decoder runs (GT mask as dense prompt on `t=0`; no prompt on `t≥1`).
3. `memory_encoder` encodes the prediction back into the memory bank for future frames.

The legacy simplified path (previous-mask-as-dense-prompt, used in runs 1–4) is retained as `_forward_simple()` and is selectable via `--no-memory-attention` for A/B comparison.

---

## 2. LoRA+ (Low-Rank Adaptation with Asymmetric LR)

**Main LoRA (decoder + memory_attention):**

| Parameter | Value | Why |
|---|---|---|
| Rank (r) | **16** | Bumped from 8 with the 16-video dataset for more adapter capacity |
| Alpha (α) | **32.0** | Scale = α/r = 2.0 (preserved as rank doubled) |
| Dropout | **0.10** | Light regularization |
| B/A LR ratio | **16.0** | LoRA+ asymmetric LR — see below |

**Hiera-encoder LoRA (last 2 trunk blocks, Run 6 addition):**

| Parameter | Value | Why |
|---|---|---|
| Rank (r) | **8** | Smaller than main LoRA — 768-dim Hiera layers give more absolute capacity per rank, and encoder edits are most impactful so a smaller adapter is safer with 16 videos |
| Alpha (α) | **16.0** | Scale = α/r = 2.0 (consistent with main LoRA) |
| Dropout | 0.10 | shared with main LoRA |
| Coverage | last 2 blocks (indices 10, 11) | The deepest features — most semantically loaded, most worth adapting for microscopy. Earlier blocks capture low-level edges/textures that transfer fine from natural images |

**Formula:** `y = W₀x + (α/r) · B(A(dropout(x)))` where A∈ℝ^(r×d_in), B∈ℝ^(d_out×r)

**Initialization:** A = Kaiming uniform, B = zeros → starts as identity (no change to pretrained behavior).

**LoRA+ improvement (Hayou et al., ICML 2024):** B starts at zeros, creating a gradient imbalance — B's gradient is much smaller than A's at init. LoRA+ fixes this by giving B a higher learning rate (lr_B = lr_A × 16). One extra hyperparameter, faster convergence + slight accuracy bump.

**Target modules (68 LoRA-wrapped Linear layers):**
- Mask decoder: `self_attn`, `cross_attn_token_to_image`, `cross_attn_image_to_token`, `final_attn` → Q, K, V, Out projections (across 2 transformer layers + final attn block)
- Memory attention: `self_attn`, `cross_attn_image` → Q, K, V, Out + `linear1`, `linear2` (FFN) (across 4 layers — wrapped but not active in our forward path)

**Trainable params:**
- Run 5 (main LoRA only, rank 16): ~5.4M (~14% of ~40M total)
- Run 6 (main LoRA + Hiera last-stage LoRA): **~7.5M (~19%)**

The Hiera encoder's base weights remain frozen — only the new LoRA A/B matrices on its last 2 blocks train.

---

## 3. Loss Functions

**Combined loss (current weights):** `L = 5·L_focal + 2·L_dice + 5·L_cbDice + 0.3·L_supervoxel + 1·L_iou + 1·L_class`

| Loss | Weight | Formula | Why |
|---|---|---|---|
| **Focal** | 5.0 | FL = −α(1−p_t)^γ log(p_t), α=0.25, γ=2.0 | Pixel-level accuracy; down-weights easy background (≈99% of image) |
| **Dice** | 2.0 | 1 − 2·\|P∩G\|/(\|P\|+\|G\|) | Region-level shape overlap |
| **cbDice** | **5.0** | 1 − 2·W_t·Tprec·W_p·Tsens / (W_t·Tprec + W_p·Tsens) on skeletons | Skeleton topology with boundary weighting; thin branches get higher weight via inverse-radius EDT. Bumped from 3.0 in Run 5 to push connectivity — predictions in Run 4 occasionally had broken thin branches |
| **Supervoxel** | **0.3** | (1−α)L₀ + α[β·∑L₀(neg_crit) + (1−β)·∑L₀(pos_crit)] | **Re-enabled at small weight** for Run 5 after the dataset grew to 16 videos and Path B (memory_attention) became active. Previously disabled because the component sat flat at 0.30; if it sits flat again here we'll disable. Diagnostics will tell us within a few epochs |
| **IoU** | 1.0 | \|predicted_iou − actual_iou\| | Confidence calibration |
| **Class** | 1.0 | BCE(pred_obj_score, obj_present) | Anti-hallucination on empty frames |

**cbDice details:** Extends clDice (Shit et al., CVPR 2021) with EDT-based boundary weighting (Shi et al., MICCAI 2024). Soft skeletonization via 5 iterations of morphological erosion/dilation. EDT computes local radius; inverse-radius weighting `1/(r+1)` makes thin branches weighted more than thick soma. Computed at downsampled `topology_resolution=256` (~16× faster than full 1024 with negligible quality loss). The scipy distance transforms are parallelised across 4 threads (scipy releases the GIL on `distance_transform_edt`).

---

## 3b. Prompt-Pathway Self-Distillation

**Why.** `_forward_memory` hands frame 0 the GT mask as a dense prompt and gives
every later frame `None` — memory is their only spatial context. Per-frame
validation metrics at the shipped model's best epoch show the cost:

| | frame 0 (prompted) | frames 1–15 (memory only) |
|---|---|---|
| IoU | 0.9466 | ~0.830 |
| Betti-0 error | 0.080 | ~0.70 |

The IoU gap is 12%; the topology gap is **~9×**. Losing the prompt does not blur
the mask, it fragments neurites. Frames 1–15 drift only 0.017 in total, so this
is a single conditioning step and not accumulating error — which means improving
the no-prompt pathway lifts all fifteen frames together, and no new cliff appears
between frames 1 and 2.

**What runs.** Each step, `distill_frames_per_clip` frames (default 4, sampled at
random per clip; 0 = every frame) are decoded a *second* time with that frame's
GT as a dense prompt. That teacher is the model's own frame-0 regime, detached,
so the loss only ever moves the student. The image encoder is shared, so a
teacher frame costs one prompt-encoder + mask-decoder pass, not a full forward.

Two terms, because the two gaps differ in size:

| term | weight | target | why |
|---|---|---|---|
| `loss_distill_dice` | 1.0 | soft teacher | region agreement; the soft target carries which pixels the model finds ambiguous |
| `loss_distill_cbdice` | 0.5 | teacher (thresholded internally) | skeleton agreement — the 9× gap is topological |

**Frame-0 prompt dropout.** With probability `prompt_dropout` (default 0.4,
ramped over `prompt_dropout_warmup_epochs` = 40) frame 0's prompt is withheld and
the conditioning frame must be segmented from features alone. Without this, 1/T
of the training signal goes to a regime that does not exist at inference —
exposure bias. The ramp stops an early bad frame-0 prediction from poisoning a
whole clip's memory bank.

**Both are on by default.** They are part of how the model trains, not opt-in
experiments, so there are no CLI flags. To reproduce a checkpoint trained before
this — including the run the Results table reports — set
`loss.use_prompt_distill = False` and `prompt_dropout = 0.0` in `src/config.py`.

**Watch `val_betti0`, not only `val_iou`.** The mechanism predicts topology
improves first, since that is where the 9× gap is. A run that pulls b0 from ~0.70
toward ~0.30 while IoU barely moves is the intended result.

References: Hinton et al. arXiv:1503.02531 (soft targets); Zhang et al. ICCV 2019
arXiv:1905.08094 and Furlanello et al. ICML 2018 arXiv:1805.04770
(self-distillation); Bengio et al. NeurIPS 2015 arXiv:1506.03099 (scheduled
sampling).

---

## 4. Optimizer & Schedule (LoRA+)

**AdamW with three parameter groups:**

| Setting | LoRA-A params | LoRA-B params | Embedding params |
|---|---|---|---|
| Optimizer | AdamW (fused) | AdamW (fused) | AdamW (fused) |
| Learning rate | **1×10⁻⁴** | **1.6×10⁻³** | **1×10⁻⁵** |
| Weight decay | **0.05** | **0.05** | 0.0 |

**Schedule (single global cosine, step-level):**

- **Warmup:** 2 epochs, linear from `0.05 × peak` to `peak`.
- **Cosine annealing:** from peak down to `min_lr_ratio × peak`.
- **LR floor:** `min_lr_ratio = 0.05` (half of GPT-3's 10% — late epochs run at finer LR steps).
- **Gradient clipping:** `max_grad_norm = 1.0`.
- **Mixed precision:** bfloat16 on CUDA. **No GradScaler** — bf16 has fp32 exponent range, no underflow risk.
- **Early stopping:** patience = 10 validation checks without IoU improvement.

**Why three LR groups (LoRA+):**
- A matrices have Kaiming init and produce reasonable gradients out of the box.
- B matrices start at zeros and have a chain-rule-induced gradient imbalance — they need 16× higher LR to catch up.
- Embeddings are pretrained — large LR steps would destroy their learned values, so 10× lower than LoRA-A.

**Why this schedule shape:** GPT-3 used the same shape (linear warmup → cosine to 10% → constant). Our 5% floor is more aggressive because in earlier runs the cosine bottomed out ~10 epochs before training ended, and that flat tail produced no improvements.

---

## 5. Data Augmentation

All spatial transforms use **identical parameters across all frames** in a clip (temporal consistency).

| Transform | Parameters | Why |
|---|---|---|
| Horizontal flip | p=0.5 | Neurons have no left-right preference |
| Vertical flip | p=0.5 | 4× effective dataset with H+V flip |
| Rotation | ±25° | Neurons grow in arbitrary orientations |
| Shear | ±20° | Simulates tissue deformation |
| Brightness | ±15% | Simulates illumination drift |
| Contrast | ±15% | Simulates microscope settings |

**No color jitter** — images are grayscale (duplicated to RGB for SAM2).

**Interpolation:** Bilinear for images, nearest-neighbor for masks (preserves integer labels).

**Note:** `precompute_features=False` is the current default. With precompute on, only feature-level h-flip / v-flip would survive, which dramatically reduces augmentation diversity on a small dataset.

---

## 6. Data Pipeline

| Setting | Value | Why |
|---|---|---|
| Image size | 1024×1024 | SAM 2 native resolution |
| Frames per clip | 8 | B×T = 16 — power of 2, GPU-friendly |
| Max objects | 3 | Typical neuron count per field of view |
| Train/val split | **75/25** | 15 train / 5 val videos with 20 labelled neurons — mean of 5 keeps the headline IoU stable run-to-run |
| Batch size | 2 | Small batch → implicit regularization on a small dataset |
| Normalization | Dataset-specific mean/std | Microscopy ≠ ImageNet distribution |

**Empty mask frames included** — teaches the model that "no neuron" is a valid output.

**Input format:** TIF stacks → per-frame PNG (lossless). Per-frame min-max normalization to [0, 255] preserves faint structures that JPEG would destroy.

**Validation split override:** `--val-videos A,B,C` lets you pin a specific 3-video val set for split-rotation experiments.

---

## 7. B200 / GPU Performance Optimizations

All applied automatically when `device == "cuda"` (no flag needed):

| Optimization | What it does |
|---|---|
| bfloat16 AMP | 2× memory savings, 2× throughput on Tensor Cores |
| **No GradScaler** | bf16 doesn't need it; removed per-step unscale/step/update syncs |
| Channels-last memory format | Conv-heavy paths (Hiera + decoder upsamplers) hit cuDNN fast path |
| Pinned memory + `non_blocking=True` | H2D copies overlap with GPU compute |
| Deferred `.item()` syncs | Loss tensors stay on-device; sync only at logging boundaries (every 10 steps) |
| Parallel scipy distance transforms | cbDice's per-batch CPU stall split across 4 threads |
| **Fused AdamW** | Single CUDA kernel for the optimizer step (vs ~136 kernel launches with default `foreach`) |
| **cuDNN SDPA backend** | FlashAttention-3 on B200 for `F.scaled_dot_product_attention` |

**Optional opt-in flags:**

| Flag | What it does | When to use |
|---|---|---|
| `--compile-encoder` | `torch.compile` on `model.image_encoder` | After warmup compile cost (~30s-2min), 15-30% faster encoder forward |
| `--compile-decoder` | `torch.compile` on `model.sam_mask_decoder` | SAM2 source compiles this same module — known safe |

---

## 8. Diagnostics in the Training Loop

Each validation epoch prints (in the SLURM log):

```
val_loss=0.4350, val_iou=0.7426, val_dice=0.8489
val_loss_components: mask=0.014, dice=0.159, cbdice=0.015, supervoxel=0.000, ...
per_frame_iou: t0=0.834, t1=0.767, t2=0.738, t3=0.731, t4=0.726, t5=0.729, t6=0.711, t7=0.705
per_video_iou: A52_C2=0.789, A51_C1=0.649, A31_C2=0.790
```

And saves the **6 worst-IoU prediction PNGs** to `checkpoints/val_samples/epoch_NNN/` for visual inspection. Filenames embed video name and IoU rank so worst cases are obvious.

**What each diagnostic tells you:**
- `train_loss_components` vs `val_loss_components`: shows whether overfitting is per-component (e.g. dice memorized but cbdice still high) or global.
- `per_frame_iou` t0→t7 drop: how much IoU the propagation chain costs. t0 has the GT prompt so it's the ceiling.
- `per_video_iou`: which validation video is dragging the mean. Lets you spot OOD videos vs general capacity issues.
- Worst-IoU PNGs: the actual failure mode. Catches things like "model over-extends to neighbouring neurons" that a single number misses.

---

## 9. Evaluation

| Metric | Formula | Range |
|---|---|---|
| **IoU** (primary) | TP / (TP + FP + FN) | 0–1, higher = better |
| **Dice** (secondary) | 2·TP / (2·TP + FP + FN) | 0–1, always ≥ IoU |

- Validation every 2 epochs.
- Best checkpoint saved when val_iou improves.
- Both-empty case → score = 1.0 (correct "nothing").

---

## 9b. Benchmark Protocol (how the comparison table is produced)

| rule | why |
|---|---|
| one scorer for every method (`benchmark/score.py`) | a method cannot be advantaged by its own metric code |
| configurations chosen on **val**, applied unchanged to **test** | no method is tuned on the reported split |
| frame 0 excluded (`--skip-first-frame`) | prompted methods are given frame 0; scoring it would flatter them |
| predictions stored as TIF stacks, scores in `benchmark/results/` | numbers are re-derivable from artefacts, not from memory |

**Prompt regimes differ between methods, and that is the point.** This work and
zero-shot SAM 2 receive a mask on frame 0. micro-SAM receives a box every frame.
SAM 3.1's video path refuses masks entirely, so its best available analogue is
points — 10 per frame, chosen by sweep. Automatic methods (Cellpose, Omnipose,
DeepSea, NeuroQuantify) receive nothing and are union-matched to the target.

The prompt column belongs beside every number: a method given the exact target
silhouette is solving a different problem from one given a word.

**Metrics.** IoU and Dice measure area overlap and are blind to a severed
neurite. clDice and cbDice are computed on skeletons and are not — cbDice
additionally weights thin structures by inverse local radius. Betti errors
(β0 components, β1 holes) are reported during training but not in the
comparison table. For evaluation, `SoftCBDiceLoss` is constructed with an
**exact** skeleton rather than the soft one used during training: the soft
skeleton is five morphological openings and cannot fully erode a thick soma,
and that bias is method-dependent (Omnipose +0.169, micro-SAM −0.004), so it
would distort the gaps between methods.

---

## 10. Chained Training Across SLURM Jobs

Single-job training is capped by the cluster's wall-clock limit (4 hours on `B200-4h` — fits ~60 epochs at current ~225 s/epoch, not 80–90). The `--resume` flag lets one logical training run span multiple jobs.

**What gets saved.** At the end of every epoch the trainer writes `checkpoints/train_state.pt` (in addition to the inference-only `lora_weights.pt`). Contents:

- model trainable params (LoRA A/B + unfrozen embeddings) — same payload as `lora_weights.pt`
- Adam optimiser moments
- LR scheduler state (preserves cosine-schedule step position)
- counters: epoch, best_val_iou, epochs_without_improvement
- `num_epochs` and `steps_per_epoch` the original schedule was built with (sanity check)

**Resuming.** Submit again with `--resume checkpoints/train_state.pt`. The training loop restores everything in-place and runs `for epoch in range(saved_epoch + 1, num_epochs)`.

**Smart sbatch pattern.** Auto-detect whether to resume or start fresh:

```bash
STATE_FILE=checkpoints/train_state.pt
if [ -f "$STATE_FILE" ]; then
    python -u cli.py train --epochs 90 --resume "$STATE_FILE"
else
    python -u cli.py train --epochs 90
fi
```

To **start a new** training run after a previous one finished, delete `train_state.pt` manually first — otherwise the smart sbatch silently resumes the old one.

**Loss of progress on a mid-epoch SIGTERM.** Save happens at end-of-epoch only. If SLURM kills the job halfway through an epoch, the saved state is one epoch behind the killed epoch — about 3–4 minutes of lost compute, no more.

---

## 11. References

| Method | Paper |
|---|---|
| SAM 2 | Ravi et al., "Segment Anything in Images and Videos", Meta 2024 |
| LoRA | Hu et al., "Low-Rank Adaptation of Large Language Models", ICLR 2022 |
| LoRA+ | Hayou et al., "LoRA+: Efficient Low Rank Adaptation of Large Models", ICML 2024 |
| Focal Loss | Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017 |
| Dice Loss | Milletari et al., "V-Net", 3DV 2016 |
| cbDice | Shi et al., "centerline boundary Dice loss for vascular segmentation", MICCAI 2024 |
| clDice | Shit et al., "clDice — Topology-Preserving Loss for Tubular Structures", CVPR 2021 |
| AdamW | Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019 |
| FlashAttention-3 | Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", 2024 |
