# Training Methods Cheat Sheet

**Project:** Fine-tuning SAM 2 (Segment Anything Model 2) for neuron time-lapse microscopy segmentation using LoRA.

---

## 1. Architecture: What's Frozen vs. Trained

```
Frame t ──► [Image Encoder (Hiera)] ──► Features ──► [Memory Attention] ──► [Mask Decoder] ──► Mask
                   FROZEN                              LoRA-adapted          LoRA-adapted
                  (~25M params)                       + FFN adapted       + unfrozen embeddings
```

- **Frozen:** Image encoder (Hiera backbone) — general spatial features, too many params to fine-tune with 10 videos
- **LoRA-adapted:** Mask decoder attention + Memory attention layers — these decide *what* to segment and *how* to propagate across frames
- **Unfrozen embeddings:** `maskmem_tpos_enc`, `no_mem_embed`, `no_obj_ptr`, `obj_ptr_proj` — small vectors controlling temporal reasoning and object tracking

---

## 2. LoRA+ (Low-Rank Adaptation with Asymmetric LR)

| Parameter | Value | Why |
|-----------|-------|-----|
| Rank (r) | 8 | Sufficient expressiveness for ~10 videos |
| Alpha (α) | 16.0 | Scale = α/r = 2.0, moderate adaptation strength |
| Dropout | 0.15 | Regularization for small dataset |
| B/A LR ratio | 16.0 | LoRA+ asymmetric learning rates (Hayou et al., ICML 2024) |

**Formula:** `y = W₀x + (α/r) · B(A(dropout(x)))` where A∈ℝ^(r×d_in), B∈ℝ^(d_out×r)

**Initialization:** A = Kaiming uniform, B = zeros → starts as identity (no change to pretrained behavior)

**LoRA+ improvement:** B starts at zeros, creating a gradient imbalance — gradients for B are much smaller than for A. LoRA+ fixes this by giving B a higher learning rate (lr_B = lr_A × 16). Zero extra cost, faster convergence.

**Target modules:**
- Mask decoder: `self_attn`, `cross_attn_token_to_image`, `cross_attn_image_to_token`, `final_attn` → Q, K, V, Out projections
- Memory attention: `self_attn`, `cross_attn_image` → Q, K, V, Out + `linear1`, `linear2` (FFN)

**Trainable params:** ~1-3% of total model — rest stays frozen

---

## 3. Loss Functions

**Combined loss:** `L = 5·L_focal + 2·L_dice + 1·L_cbDice + 1·L_supervoxel + 1·L_iou + 1·L_class`

| Loss | Weight | Formula | Why |
|------|--------|---------|-----|
| **Focal** | 5.0 | FL = −α(1−p_t)^γ log(p_t), α=0.25, γ=2.0 | Pixel-level accuracy; down-weights easy background pixels (99% of image) |
| **Dice** | 2.0 | 1 − 2\|P∩G\|/(|P|+\|G\|) | Region-level shape overlap; cares about whole-mask morphology |
| **cbDice** | 1.0 | 1 − 2·W_t·Tprec·W_p·Tsens/(W_t·Tprec+W_p·Tsens) on skeletons | Skeleton topology with boundary weighting; thin branches get higher weight via inverse-radius EDT |
| **Supervoxel** | 1.0 | (1−α)L₀ + α[β·∑L₀(neg_crit) + (1−β)·∑L₀(pos_crit)] | Connectivity; penalizes split/merge errors via BFS critical component detection |
| **IoU** | 1.0 | \|predicted_iou − actual_iou\| | Confidence calibration; trains model to know when its predictions are bad |
| **Class** | 1.0 | BCE(pred_obj_score, obj_present) | Anti-hallucination; teaches "no neuron here" on empty frames |

**cbDice details:** Extends clDice (Shit et al., CVPR 2021) with distance-transform boundary weighting (Shi et al., MICCAI 2024). Soft skeletonization via iterative morphological erosion/dilation (5 iterations, 3×3 kernel). EDT computes local radius; inverse-radius weighting gives higher importance to thinner structures (thin dendrites weighted more than thick soma).

**Supervoxel details:** α=0.5, β=0.5. BFS on 4-connected components of error regions. O(n) complexity. Two critical types: (1) false negatives causing disconnections, (2) false positives causing merges.

**Disable topology losses for speed:** `python cli.py train --no-topology` (disables both cbDice + Supervoxel)

---

## 4. Optimizer & Schedule (LoRA+)

| Setting | LoRA A params | LoRA B params | Embedding params |
|---------|---------------|---------------|------------------|
| Optimizer | AdamW | AdamW | AdamW |
| Learning rate | 1×10⁻⁴ | 1.6×10⁻³ | 1×10⁻⁵ |
| Weight decay | 0.1 | 0.1 | 0.0 |

- **LoRA+ ratio:** B lr = A lr × 16 (B starts at zeros, needs higher LR)
- **Warmup:** 1 epoch linear ramp from 0 → lr
- **Schedule:** Cosine annealing after warmup, min LR ratio = 0.1
- **Gradient clipping:** max_norm = 1.0
- **Mixed precision:** bfloat16 on CUDA (2× speed, 2× memory savings)
- **Early stopping:** patience = 10 validation checks without IoU improvement

**Why three LR groups (LoRA+)?** A matrices (down-projection) have Kaiming initialization and produce reasonable gradients. B matrices start at zeros, creating a gradient imbalance — they need 16× higher LR to catch up. Embeddings are pretrained — large steps would destroy their learned values.

---

## 5. Data Augmentation

All spatial transforms use **identical parameters across all frames** in a clip (temporal consistency).

| Transform | Parameters | Why |
|-----------|------------|-----|
| Horizontal flip | p=0.5 | Neurons have no left-right preference |
| Vertical flip | p=0.5 | 4× effective dataset with H+V flip |
| Rotation | ±25° | Neurons grow in arbitrary orientations |
| Shear | ±20° | Simulates tissue deformation |
| Scale | 0.85–1.15 | Simulates zoom variation |
| Elastic deformation | p=0.3, α=50, σ=5 | Mimics biological morphology variation |
| Brightness | ±15% | Simulates illumination drift |
| Contrast | ±15% | Simulates microscope settings |
| Gaussian blur | p=0.3, σ=0.1–2.0 | Focus variation |
| Gaussian noise | p=0.2, std=0.01–0.05 | Sensor noise |

**No color jitter** — images are grayscale (duplicated to RGB for SAM2).

**Interpolation:** Bilinear for images, nearest-neighbor for masks (preserves integer labels).

---

## 6. Data Pipeline

| Setting | Value | Why |
|---------|-------|-----|
| Image size | 1024×1024 | SAM 2 native resolution |
| Frames per clip | 8 | Temporal context vs. GPU memory tradeoff |
| Max objects | 3 | Typical neuron count per field of view |
| Train/val split | 80/20 | ~8 train, 2 val videos |
| Batch size | 2 | GPU memory constraint |
| Normalization | Dataset-specific mean/std | Microscopy ≠ ImageNet distribution |

**Empty mask frames included** — teaches model that "no neuron" is a valid output, preventing hallucination on background frames.

**Input format:** TIF stacks → per-frame PNG (lossless). Per-frame min-max normalization to [0, 255] preserves faint structures that JPEG would destroy.

---

## 7. Evaluation

| Metric | Formula | Range |
|--------|---------|-------|
| **IoU** (primary) | TP / (TP + FP + FN) | 0–1, higher = better |
| **Dice** (secondary) | 2·TP / (2·TP + FP + FN) | 0–1, always ≥ IoU |

- Validation every 2 epochs
- Best checkpoint saved when val_iou improves
- Both-empty case → score = 1.0 (correct "nothing")

---

## References

| Method | Paper |
|--------|-------|
| SAM 2 | Ravi et al., "Segment Anything in Images and Videos", Meta 2024 |
| LoRA | Hu et al., "Low-Rank Adaptation of Large Language Models", ICLR 2022 |
| LoRA+ | Hayou et al., "LoRA+: Efficient Low Rank Adaptation of Large Models", ICML 2024 |
| Focal Loss | Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017 |
| Dice Loss | Milletari et al., "V-Net", 3DV 2016 |
| cbDice | Shi et al., "centerline boundary Dice loss for vascular segmentation", MICCAI 2024 |
| clDice | Shit et al., "clDice — Topology-Preserving Loss for Tubular Structures", CVPR 2021 |
| Supervoxel Loss | Grim et al., "Connectivity-Preserving Segmentation with Supervoxel Loss", AAAI 2025 |
| AdamW | Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019 |
