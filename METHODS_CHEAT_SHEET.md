# Training Methods Cheat Sheet

**Project:** Fine-tuning SAM 2 (Segment Anything Model 2) for neuron time-lapse microscopy segmentation, using LoRA+ on a small dataset (~10 videos), tuned for NVIDIA B200.

---

## 1. Architecture: What's Frozen vs. Trained

```
Frame t ──► [Image Encoder (Hiera-tiny)] ──► Features ──► [Mask Decoder] ──► Mask
                   FROZEN                                  LoRA-adapted
                  (~25M params)                          + unfrozen embeddings
```

- **Frozen:** Image encoder (Hiera backbone). Pretrained features transfer well to microscopy without modification; retraining on 10 videos would erase them.
- **LoRA-adapted:** Mask decoder transformer (self-attn + cross-attn) and `memory_attention` layers (Q, K, V, Out projections + FFN linears) — 68 layers total.
- **Unfrozen embeddings:** `maskmem_tpos_enc`, `no_mem_embed`, `no_obj_ptr`, `obj_ptr_proj`, `obj_ptr_tpos_proj` (12 small parameter tensors). These control temporal reasoning and object tracking.

**Note on `memory_attention`:** the `SAM2TrainingWrapper` in `src/trainer.py` uses a *simplified* mask-prompt propagation between frames (interpolating the previous prediction as a dense prompt) rather than calling SAM2's full `memory_attention.forward`. So although LoRA wraps memory_attention layers, those adapters do not receive gradients in our training. This is a known simplification.

---

## 2. LoRA+ (Low-Rank Adaptation with Asymmetric LR)

| Parameter | Value | Why |
|---|---|---|
| Rank (r) | **8** | Sufficient expressiveness; power of 2 for GPU alignment |
| Alpha (α) | **16.0** | Scale = α/r = 2.0 |
| Dropout | **0.10** | Light regularization (over-regularizing previously closed train/val gap from below) |
| B/A LR ratio | **16.0** | LoRA+ asymmetric LR — see below |

**Formula:** `y = W₀x + (α/r) · B(A(dropout(x)))` where A∈ℝ^(r×d_in), B∈ℝ^(d_out×r)

**Initialization:** A = Kaiming uniform, B = zeros → starts as identity (no change to pretrained behavior).

**LoRA+ improvement (Hayou et al., ICML 2024):** B starts at zeros, creating a gradient imbalance — B's gradient is much smaller than A's at init. LoRA+ fixes this by giving B a higher learning rate (lr_B = lr_A × 16). One extra hyperparameter, faster convergence + slight accuracy bump.

**Target modules (68 LoRA-wrapped Linear layers):**
- Mask decoder: `self_attn`, `cross_attn_token_to_image`, `cross_attn_image_to_token`, `final_attn` → Q, K, V, Out projections (across 2 transformer layers + final attn block)
- Memory attention: `self_attn`, `cross_attn_image` → Q, K, V, Out + `linear1`, `linear2` (FFN) (across 4 layers — wrapped but not active in our forward path)

**Trainable params:** ~12% of total model (5M of 39M). Encoder is the other 88%.

---

## 3. Loss Functions

**Combined loss (current weights):** `L = 5·L_focal + 2·L_dice + 3·L_cbDice + 1·L_iou + 1·L_class`

(Supervoxel loss is **disabled** — see "Supervoxel" row below.)

| Loss | Weight | Formula | Why |
|---|---|---|---|
| **Focal** | 5.0 | FL = −α(1−p_t)^γ log(p_t), α=0.25, γ=2.0 | Pixel-level accuracy; down-weights easy background (≈99% of image) |
| **Dice** | 2.0 | 1 − 2·\|P∩G\|/(\|P\|+\|G\|) | Region-level shape overlap |
| **cbDice** | **3.0** | 1 − 2·W_t·Tprec·W_p·Tsens / (W_t·Tprec + W_p·Tsens) on skeletons | Skeleton topology with boundary weighting; thin branches get higher weight via inverse-radius EDT. Bumped from 1.0 — neurons are tree-like, topology matters more than bulk pixels |
| **Supervoxel** | (off) | — | **Disabled by default**: in earlier runs sat flat at ~0.30 from epoch 4 onward without contributing to learning. Re-enable via `--supervoxel-weight F` if topology regressions appear |
| **IoU** | 1.0 | \|predicted_iou − actual_iou\| | Confidence calibration |
| **Class** | 1.0 | BCE(pred_obj_score, obj_present) | Anti-hallucination on empty frames |

**cbDice details:** Extends clDice (Shit et al., CVPR 2021) with EDT-based boundary weighting (Shi et al., MICCAI 2024). Soft skeletonization via 5 iterations of morphological erosion/dilation. EDT computes local radius; inverse-radius weighting `1/(r+1)` makes thin branches weighted more than thick soma. Computed at downsampled `topology_resolution=256` (~16× faster than full 1024 with negligible quality loss). The scipy distance transforms are parallelised across 4 threads (scipy releases the GIL on `distance_transform_edt`).

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
| Train/val split | **75/25** | 9 train / 3 val videos — mean of 3 is more stable than mean of 2 |
| Batch size | 2 | Small batch → implicit regularization on a tiny dataset |
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

## 10. References

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
