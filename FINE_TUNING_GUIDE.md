# Fine-Tuning SAM 2 for Neuron Time-Lapse Segmentation: A Technical Guide

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Background: SAM 2 Architecture](#2-background-sam-2-architecture)
3. [Why LoRA? The Fine-Tuning Strategy](#3-why-lora-the-fine-tuning-strategy)
4. [LoRA Mathematics](#4-lora-mathematics)
5. [Which Layers We Adapt and Why](#5-which-layers-we-adapt-and-why)
6. [Data Pipeline](#6-data-pipeline)
7. [Augmentation Strategy](#7-augmentation-strategy)
8. [Loss Function: Multi-Objective Training](#8-loss-function-multi-objective-training)
9. [Optimizer and Learning Rate Schedule](#9-optimizer-and-learning-rate-schedule)
10. [Training Loop Walkthrough](#10-training-loop-walkthrough)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Inference Integration](#12-inference-integration)
13. [Hyperparameter Summary](#13-hyperparameter-summary)
14. [Running the Pipeline](#14-running-the-pipeline)
15. [File Reference](#15-file-reference)

---

## 1. Problem Statement

We have a neuron time-lapse microscopy dataset: grayscale TIF stacks where each stack contains ~72 frames taken 1 hour apart. Neurons appear as faint structures (cell bodies, dendrites, axon tips) against a dark background.

**Current workflow:**
1. A human labels the first frame (frame 0) in napari using a brush tool (e.g., label 4 = purple)
2. SAM 2 (Segment Anything Model 2) propagates that label to all remaining ~71 frames
3. The output is a full-video segmentation mask

**The problem:** The base SAM 2 model was trained on natural images and videos (YouTube-VOS, SA-V dataset). It has never seen microscopy data. As a result:
- It may lose track of neurons that move subtly between frames
- It may hallucinate neurons where there are none (background regions)
- It may miss faint structures (thin dendrites, axon tips) that look like noise to a model trained on natural images

**Our goal:** Fine-tune SAM 2 on our 20 previously labeled neuron datasets so that propagation quality improves, while preserving SAM 2's general video understanding capabilities. The shipping configuration ("Run 7") trains for 90 epochs on a 15/5 train/val split and reaches **val IoU 0.8401 / Dice 0.9117** on the held-out 5 videos.

---

## 2. Background: SAM 2 Architecture

SAM 2 is a video segmentation model with four major components:

```
                    Frame t image
                         |
                   [Image Encoder]          <-- Hiera backbone (frozen)
                         |
                    Image Features
                         |
              +----------+----------+
              |                     |
       [Memory Attention]    [Mask Decoder]
              |                     |
       Temporal Context      Mask Prediction
              |                     |
       [Memory Encoder]      Output Mask
              |
        Memory Bank
```

### 2.1 Image Encoder (Hiera Backbone)

A hierarchical vision transformer that converts each video frame into a spatial feature map. For SAM 2.1 Hiera Tiny:
- Input: RGB image 1024 x 1024 x 3
- Output: Feature map 64 x 64 x 256

This component was pretrained on millions of images. Its features are highly general-purpose and transfer well to microscopy without modification.

**Decision: Freeze the image encoder base weights, but adapt its last stage with LoRA (Run 6+).** Retraining the full encoder on 20 videos would destroy its general features (catastrophic forgetting) and waste compute. The encoder already produces good spatial features; our primary problem is in *how those features are interpreted across time*, not in how individual frames are encoded. But the deepest features had never seen phase-contrast microscopy, so Run 6 added a smaller LoRA (rank 8) to the last 2 trunk blocks for gentle domain adaptation. See Section 5 for the full target list.

### 2.2 Mask Decoder

A two-way transformer that takes image features + a prompt (mask from frame 0) and produces a segmentation mask. It contains:
- **Self-attention layers**: tokens attend to each other
- **Cross-attention (token-to-image)**: prompt tokens query image features
- **Cross-attention (image-to-token)**: image features attend to prompt tokens
- **Final attention**: produces the output mask

Each attention layer has four linear projections: Q (query), K (key), V (value), and output projection.

**Why adapt this:** The mask decoder decides *what* to segment. A decoder trained on natural objects (people, cars, animals) has learned to look for high-contrast edges and salient objects. Neurons are low-contrast with diffuse boundaries. Adapting the decoder teaches it to recognize neuron-specific visual patterns.

### 2.3 Memory Attention

The core temporal reasoning module. It allows each frame to attend to features stored in a *memory bank* from previously processed frames. It contains:
- **Self-attention**: current frame features attend to themselves
- **Cross-attention (image)**: current frame queries the memory bank
- **Feed-forward layers** (linear1, linear2): process the attended features

**Why adapt this:** This is the component that decides *how* information propagates across frames. The base model was trained on fast-moving natural video (30fps). Our data is slow time-lapse (1 frame/hour) with gradual morphological changes. Adapting the memory attention teaches the model our data's specific temporal dynamics.

### 2.4 Memory Encoder

Encodes the predicted mask from each frame into a compact representation that gets stored in the memory bank. Future frames then read from this bank via memory attention.

---

## 3. Why LoRA? The Fine-Tuning Strategy

We have 20 labeled videos (~1440 frames total). This is still a tiny dataset by deep learning standards. We need a fine-tuning strategy that:

1. **Avoids overfitting** -- with only 20 videos, a fully fine-tuned model would memorize training data
2. **Preserves pretrained knowledge** -- SAM 2's video understanding should remain intact
3. **Is parameter-efficient** -- small checkpoint size, fast training
4. **Works on a single GPU** -- single NVIDIA GPU, not a server cluster

### Alternatives Considered

| Strategy | Trainable Params | Risk of Overfitting | Preserves Pretrained? |
|----------|-----------------|---------------------|-----------------------|
| Full fine-tuning | ~40M (100%) | Very high | No |
| Fine-tune decoder only | ~10M (26%) | High | Partially |
| **LoRA, dual-rank (our choice)** | **~5.6M (13.9%)** | **Low** | **Yes** |
| Train from scratch | ~40M (100%) | Impossible (too little data) | N/A |
| Feature extraction (freeze all) | 0 | None | Yes, but can't adapt |

**LoRA** (Low-Rank Adaptation) is the optimal choice: it adds a small number of trainable parameters *alongside* the frozen original weights, allowing adaptation without destroying pretrained features.

---

## 4. LoRA Mathematics

### 4.1 The Core Idea

Every linear layer in a neural network performs:

```
y = W * x + b
```

where W is a weight matrix of shape (d_out, d_in). In a pretrained model, W encodes learned knowledge we want to preserve.

LoRA adds a *low-rank perturbation* to W:

```
y = W_0 * x + (alpha/r) * B * A * x
    --------   ----------------------
    frozen     trainable (low-rank)
    original   adaptation
```

where:
- **W_0** is the original frozen weight matrix (d_out x d_in)
- **A** is a "down-projection" matrix (r x d_in) -- compresses input to rank r
- **B** is an "up-projection" matrix (d_out x r) -- projects back to output dimension
- **r** is the rank (a small integer). Run 7 uses **r=16** on the main LoRA (mask_decoder + memory_attention) and **r=8** on the Hiera last-stage LoRA — a dual-rank checkpoint.
- **alpha** is a scaling factor. We use alpha=32 (main) and alpha=16 (Hiera) so that scale = alpha/r = 2.0 in both groups.

### 4.2 Why This Works

The key insight from the LoRA paper (Hu et al., 2021) is that the *change* needed to adapt a pretrained model to a new task often has **low intrinsic rank**. That is, the adaptation delta-W = W_new - W_0 can be well-approximated by a low-rank matrix B * A.

**Parameter count comparison** for a single layer:

| Method | Parameters |
|--------|-----------|
| Full weight matrix (d_out x d_in) | d_out * d_in |
| LoRA (rank r) | r * (d_out + d_in) |

For a typical attention projection (256 x 256):
- Full: 256 * 256 = 65,536 parameters
- LoRA rank 16: 16 * (256 + 256) = 8,192 parameters (**8x reduction**)
- LoRA rank 8: 8 * (256 + 256) = 4,096 parameters (**16x reduction**)

### 4.3 The Scaling Factor

The LoRA output is scaled by alpha/r:

```
scale = alpha / r = 32 / 16 = 2.0   # main LoRA
scale = alpha / r = 16 / 8  = 2.0   # Hiera last-stage LoRA
```

This controls *how much* the adaptation influences the output. The ratio alpha/r is the effective scaling:
- **Higher ratio** (e.g., alpha=32, r=8 -> scale=4): Stronger adaptation, faster learning, but risks instability
- **Lower ratio** (e.g., alpha=8, r=8 -> scale=1): Gentler adaptation, more stable, but may need more epochs
- **Our choice** (scale=2 in both groups): A moderate balance. The adaptation has meaningful influence without dominating the pretrained features. Keeping `alpha/r` constant when we doubled the rank from 8 to 16 in Run 5 preserved the relative magnitude of the adaptation per layer.

### 4.4 Initialization

This is critical for stable training:
- **A is initialized** with Kaiming uniform (random): provides diverse input projections from the start
- **B is initialized** to zeros: at the start of training, B * A = 0, so the model behaves *exactly* like the pretrained model

This means training begins from the pretrained model's behavior and gradually learns the adaptation. The model never "forgets" its pretrained state at initialization.

### 4.5 Dropout

We apply dropout (p=0.10) to the LoRA path:

```
y = W_0 * x + scale * B(A(dropout(x)))
```

This acts as regularization: during training, random input dimensions are zeroed out with 10% probability, forcing the adaptation to be robust and preventing overfitting to specific input patterns. This is especially important with only 20 training videos. We tuned this: Run 1 used 0.05 (underfit), Run 2 tried 0.25 (over-regularised — train and val losses collapsed together at a capacity wall), and Run 3+ settled on 0.10 as the sweet spot.

### 4.6 Code Reference (lora.py)

```python
class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, rank=16, alpha=32.0, dropout=0.10):
        self.original = original                           # Frozen W_0
        self.lora_A = nn.Linear(in_features, rank, bias=False)   # A: (d_in -> r)
        self.lora_B = nn.Linear(rank, out_features, bias=False)  # B: (r -> d_out)
        self.scale = alpha / rank                          # = 2.0
        self.lora_dropout = nn.Dropout(p=dropout)          # p = 0.10

        # Initialize A with Kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Freeze original weights
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        original_out = self.original(x)                    # W_0 * x
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))  # B(A(dropout(x)))
        return original_out + self.scale * lora_out        # W_0*x + (alpha/r)*B*A*x
```

---

## 5. Which Layers We Adapt and Why

Not all layers benefit equally from LoRA. We target specific components based on their role in the pipeline.

### 5.1 Targeted Modules

**Main LoRA (rank 16, alpha 32) — 68 Linear layers:**

```
sam_mask_decoder.transformer.layers.{0,1}.self_attn.{q,k,v,out}_proj
sam_mask_decoder.transformer.layers.{0,1}.cross_attn_token_to_image.{q,k,v,out}_proj
sam_mask_decoder.transformer.layers.{0,1}.cross_attn_image_to_token.{q,k,v,out}_proj
sam_mask_decoder.transformer.final_attn_token_to_image.{q,k,v,out}_proj
memory_attention.layers.{0,1,2,3}.self_attn.{q,k,v,out}_proj
memory_attention.layers.{0,1,2,3}.cross_attn_image.{q,k,v,out}_proj
memory_attention.layers.{0,1,2,3}.linear1
memory_attention.layers.{0,1,2,3}.linear2
```

**Hiera last-stage LoRA (rank 8, alpha 16) — 8 Linear layers, added in Run 6:**

```
image_encoder.trunk.blocks.{10,11}.attn.qkv
image_encoder.trunk.blocks.{10,11}.attn.proj
image_encoder.trunk.blocks.{10,11}.mlp.layers.{0,1}
```

These are the last 2 blocks of the Hiera trunk (stages `[1,2,7,2]` → indices 10 and 11 are the last stage). Each `MultiScaleBlock` exposes `attn.qkv` (fused QKV projection), `attn.proj` (output projection), and two MLP linears.

**This is a dual-rank checkpoint.** Each saved tensor lives in one of two regex groups; inference loaders classify by which target list matches and instantiate the LoRA with the matching rank. `load_lora_weights` does this automatically.

Total trainable parameter count (LoRA A/B matrices + unfrozen embeddings): **5,557,250 of 39,876,930 (13.9%)**.

### 5.2 Why These Specific Layers

**Attention projections (Q, K, V, Out):**

In a transformer attention mechanism:
```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

- **Q (query)**: "What am I looking for?" -- Adapting Q changes what features the model searches for (neuron-specific patterns vs natural object patterns)
- **K (key)**: "What do I represent?" -- Adapting K changes how spatial features are advertised to queries
- **V (value)**: "What information do I provide?" -- Adapting V changes what content is passed through when attention activates
- **Out projection**: "How do I combine?" -- Adapting the output projection changes how multi-head results are mixed

By adapting all four projections, we give the model full flexibility to relearn attention patterns for neuron morphology.

**Feed-forward layers (linear1, linear2) in memory attention:**

The feed-forward network in each memory attention layer processes features after attention:
```
FFN(x) = linear2(GELU(linear1(x)))
```

This is where the model performs *nonlinear feature transformation* on temporally-attended features. Adapting these layers lets the model learn neuron-specific temporal feature processing (e.g., "this pattern of gradual brightness change is a neuron growing, not noise").

### 5.3 What We Don't Adapt and Why

**Image encoder (Hiera) base weights:** Stay frozen. Adapting the full encoder with 20 videos would cause overfitting. The encoder has ~25M parameters -- keeping them frozen is also a major compute saving. Run 6 adds *small* LoRA adapters on the last 2 trunk blocks (the deepest features, most domain-specific), but the base Hiera weights are untouched.

**Memory encoder:** Not adapted via LoRA. It's a relatively simple module that encodes masks into memory tokens. The representation is close to the mask itself, and the base encoding works fine for binary neuron masks.

### 5.4 Unfrozen Embeddings

In addition to LoRA, we also unfreeze several small learnable embeddings:

```python
"maskmem_tpos_enc"    # Temporal position encoding for mask memory
"no_mem_embed"        # Embedding for "no memory available" token
"no_mem_pos_enc"      # Position encoding when no memory exists
"no_obj_ptr"          # Pointer for "no object detected" state
"obj_ptr_proj"        # Object pointer projection
"obj_ptr_tpos_proj"   # Object pointer temporal projection
```

**Why:** These are small embedding vectors (a few thousand parameters total) that control meta-level behavior:
- How the model represents time (temporal position encodings)
- What happens when the model loses track of the object (no_obj_ptr)
- How object identity is maintained across frames (obj_ptr_proj)

These are cheap to fine-tune and directly influence tracking quality, which is our primary concern.

---

## 6. Data Pipeline

### 6.1 Input Format

Raw data:
- **Image TIF**: Grayscale 16-bit microscopy stack, shape (T, H, W), T~72 frames
- **Mask TIF**: Label stack, shape (T, H, W), integer values (0=background, 4=neuron, etc.)

### 6.2 Conversion Process (prepare_data.py)

Each video is converted to per-frame PNG files:

**Image processing:**
```python
# For each frame:
frame = frame.astype(np.float32)
fmin, fmax = frame.min(), frame.max()
frame = (frame - fmin) / (fmax - fmin) * 255    # Per-frame min-max to [0, 255]
frame = frame.astype(np.uint8)
rgb_frame = np.stack([frame, frame, frame], axis=-1)  # Grayscale -> RGB
imageio.imwrite("00042.png", rgb_frame)          # Lossless PNG
```

**Why per-frame min-max normalization:** Microscopy images have varying intensity levels across frames (due to photobleaching, focus drift, etc.). Per-frame normalization ensures each frame uses the full dynamic range, making neuron structures maximally visible.

**Why PNG and not JPEG:** JPEG is lossy. It achieves compression by discarding high-frequency details. In microscopy, faint dendrites and axon tips occupy exactly those high frequencies -- they look like noise to JPEG but are biologically meaningful. A single round of JPEG compression at quality 95 can destroy structures that are critical for accurate segmentation.

**Mask processing:**
```python
mask = mask.astype(np.int32)
if label_id > 0:
    mask = (mask == label_id).astype(np.uint8) * label_id  # Keep specific label
else:
    mask = np.clip(mask, 0, 255).astype(np.uint8)          # Keep all labels
imageio.imwrite("00042.png", mask)  # 8-bit grayscale PNG
```

### 6.3 Dataset Normalization Statistics

During preparation, we compute per-channel mean and standard deviation across the entire dataset:

```python
# Per-video: accumulate pixel statistics on [0,1] scale
pixel_sum += frame_float.sum()          # Sum of all pixel values
pixel_sq_sum += (frame_float ** 2).sum() # Sum of squared values
pixel_count += frame_float.size          # Total pixel count

# Per-video stats:
mean_i = pixel_sum / pixel_count
std_i = sqrt(pixel_sq_sum / pixel_count - mean_i^2)

# Aggregate across all videos (combined variance formula):
dataset_mean = sum(mean_i * count_i) / total_pixels

dataset_E_X2 = sum((std_i^2 + mean_i^2) * count_i) / total_pixels
dataset_std = sqrt(dataset_E_X2 - dataset_mean^2)
```

The mathematical identity used here is:

```
Var(X) = E[X^2] - (E[X])^2
```

For combining variances from subsets:

```
E_total[X^2] = sum_i( (Var_i + Mean_i^2) * N_i ) / N_total
```

This is exact (not an approximation) when combining statistics from disjoint subsets.

**Why not use ImageNet stats:** ImageNet images are natural photos (well-lit, colorful). Our microscopy images are dark, grayscale, and low-contrast. Using ImageNet mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225] would incorrectly center and scale our data, placing neuron signal in the wrong distribution range for the network.

### 6.4 Empty Mask Frames

Many frames in the 72-frame stack may have no labeled neuron (the neuron hasn't appeared yet, or has died). These frames have all-zero masks.

**We explicitly include these frames in training.** Without them, the model only sees positive examples ("here is a neuron") and never learns "there is no neuron here." This leads to hallucination: the model predicts neurons on background-only frames because it was never taught that "no neuron" is a valid output.

In the dataset code, when a clip contains no objects at all:
```python
if not obj_ids:
    obj_ids = [1]  # Dummy object with all-zero mask
```

Combined with `pred_obj_scores=True` in the loss function, this teaches the model to predict low object-presence scores on empty frames, directly addressing hallucination.

### 6.5 Output Structure

```
training_data/
  images/
    A31_C2/
      00000.png    # Frame 0, RGB, 0-255
      00001.png    # Frame 1
      ...
      00071.png    # Frame 71
    A42_C1/
      ...
  masks/
    A31_C2/
      00000.png    # Frame 0, 0=bg, 4=neuron
      ...
    A42_C1/
      ...
  stats.json       # Video metadata + normalization stats
```

---

## 7. Augmentation Strategy

### 7.1 Design Principles

1. **Temporal consistency**: All spatial transforms use the *same random parameters* for every frame in a clip. If frame 3 is flipped horizontally, frame 4 must also be flipped horizontally. Otherwise the model would see physically impossible motion.

2. **Microscopy-appropriate**: No color jitter (our images are grayscale duplicated to RGB). No random crop (neurons may be at image edges). Geometric transforms only.

3. **Conservative intensity**: Microscopy has specific intensity characteristics. We apply mild brightness/contrast changes (+-15%) to simulate photobleaching and illumination variation.

### 7.2 Spatial Augmentations

**Horizontal flip** (p=0.5):
```
flipped[x, y] = original[W-1-x, y]
```
Neurons have no left-right preference. This doubles the effective dataset.

**Vertical flip** (p=0.5):
```
flipped[x, y] = original[x, H-1-y]
```
Same reasoning. Combined with horizontal flip, this gives 4x effective dataset.

**Rotation** (uniform in [-25deg, +25deg]):
```
[x']   [cos(theta)  -sin(theta)] [x]
[y'] = [sin(theta)   cos(theta)] [y]
```
Neurons grow in arbitrary orientations. Training with rotations makes the model orientation-invariant.

**Shear** (uniform in [-20deg, +20deg]):
```
[x']   [1        shear_x] [x]
[y'] = [shear_y  1      ] [y]
```
Simulates perspective distortion and tissue deformation.

**Interpolation choices:**
- Frames: Bilinear (smooth, avoids aliasing)
- Masks: Nearest-neighbor (preserves integer label values -- bilinear would create invalid fractional labels)

### 7.3 Intensity Augmentations

**Brightness** (factor in [0.85, 1.15]):
```
pixel_new = pixel_old * factor
```
Simulates global illumination variation.

**Contrast** (factor in [0.85, 1.15]):
```
pixel_new = mean + (pixel_old - mean) * factor
```
Simulates contrast changes from microscope settings.

Both are applied with the *same factor* across all frames in a clip (temporal consistency).

### 7.4 Resize

All frames are resized to 1024x1024 (SAM 2's expected input size):
- Frames: Bilinear with antialiasing
- Masks: Nearest-neighbor

### 7.5 Normalization

After augmentation, frames are normalized using dataset-specific statistics:
```python
frame = (frame - mean) / std
```

This centers the data around zero with unit variance, matching what the network expects from training.

### 7.6 Feature-Level Augmentation (with Precomputed Features)

When feature precomputation is enabled (default), pixel-level augmentation cannot be applied (features are already computed). Instead, flip augmentation is applied at the feature level:

| Augmentation | With precomputed features | Without (`--no-precompute`) |
|---|---|---|
| H-flip, V-flip | **Keep** — flip feature maps + masks | Applied at pixel level |
| Rotation, shear | **Dropped** — can't apply at feature level | Applied at pixel level |
| Elastic deformation | **Dropped** | Applied at pixel level |
| Brightness, contrast | **Dropped** | Applied at pixel level |
| Gaussian blur/noise | **Dropped** | Applied at pixel level |

For flips: `torch.flip(feat, dims=[-1])` for hflip, `dims=[-2]` for vflip. Applied to backbone_fpn tensors. Masks are flipped to match.

**Trade-off:** Feature precomputation gives ~10x speedup but reduces augmentation diversity. For small datasets (<5 videos), consider `--no-precompute` for more augmentation diversity at the cost of slower training.

---

## 8. Loss Function: Multi-Objective Training

We implement our own `MultiStepMultiMasksAndIous` loss in `src/loss_fns.py`, which combines six loss terms. This mirrors the loss used to train SAM 2 on the SA-V dataset, but is implemented locally since Meta's training code is not included in the pip-installable SAM2 package.

### 8.1 Focal Loss (weight = 5.0)

**Purpose:** Pixel-level mask prediction quality.

Standard binary cross-entropy:
```
BCE(p, y) = -y * log(p) - (1-y) * log(1-p)
```

The problem: most pixels are background (class imbalance). The easy background pixels dominate the loss.

**Focal loss** (Lin et al., 2017) down-weights easy examples:

```
FL(p, y) = -alpha * (1 - p_t)^gamma * log(p_t)

where p_t = p    if y = 1
      p_t = 1-p  if y = 0
```

Parameters:
- **alpha = 0.25**: Balances foreground/background. Since neurons occupy <5% of pixels, alpha gives slightly more weight to non-neuron pixels to prevent false positives.
- **gamma = 2.0**: The focusing parameter. When a pixel is correctly classified with high confidence (p_t close to 1), (1-p_t)^2 becomes very small, effectively ignoring that pixel. Training focuses on hard, uncertain pixels (the neuron boundary).

**Why weight 5.0:** Mask quality is the primary objective. The high weight ensures pixel-level accuracy is the dominant training signal.

### 8.2 Dice Loss (weight = 2.0)

**Purpose:** Region-level overlap quality.

```
Dice = 2 * |pred intersection target| / (|pred| + |target|)
Loss_dice = 1 - Dice
```

**Why we need this in addition to focal loss:** Focal loss operates per-pixel independently. It doesn't consider the *shape* of the prediction. Dice loss operates on the entire mask as a region, encouraging predictions that match the overall shape and area of the ground truth.

This is especially important for neuron structures where preserving morphology (branching patterns, soma shape) matters as much as pixel accuracy.

### 8.3 IoU Prediction Loss (weight = 1.0)

**Purpose:** Calibrated confidence estimation.

SAM 2 predicts not just the mask but also an estimated IoU score (how good it thinks its prediction is). This loss trains that score to be accurate:

```
Loss_iou = |predicted_iou - actual_iou|
```

**Why this matters:** During inference, the IoU score helps the model decide which mask prediction to trust. An accurate IoU predictor means the model can flag low-quality frames during propagation, enabling potential recovery strategies.

### 8.4 Object Score / Classification Loss (weight = 1.0)

**Purpose:** Object presence detection (anti-hallucination).

```
Loss_class = BCE(predicted_obj_score, target_obj_present)
```

This is activated by `pred_obj_scores=True`. The model learns to predict whether the target object is actually present in each frame.

**Why this is critical for our use case:** In our 72-frame videos, the neuron may not be visible in all frames. Without this loss, the model always predicts "object present" and hallucinates neurons on empty frames. With this loss, it learns to output low confidence when the neuron isn't there.

### 8.5 cbDice Loss (weight = 5.0)

**Purpose:** Preserve skeleton topology of tubular structures with boundary weighting.

cbDice (centerline boundary Dice, Shi et al., MICCAI 2024) extends clDice (Shit et al., CVPR 2021) by adding distance-transform-based boundary weighting. It computes Dice on *skeletonized* masks with inverse-radius weights from the Euclidean distance transform (EDT):

```
cbTprec = |W_t · skel(pred) · target| / |W_t · skel(pred)|   (weighted skeleton precision)
cbTsens = |W_p · skel(target) · pred| / |W_p · skel(target)| (weighted skeleton sensitivity)
cbDice = 2 × cbTprec × cbTsens / (cbTprec + cbTsens)
```

Where W_t and W_p are inverse-radius distance transform weights: for a structure with local radius r, weight = 1/(r+1). Thin dendrites (r≈1) get weight ≈0.5, thick soma (r≈10) gets weight ≈0.09.

**Why for neurons:** Breaking a thin neuron branch removes its centerline skeleton. Standard Dice barely notices (few pixels lost), but cbDice heavily penalizes it because the skeleton is destroyed AND the thin structure receives high weight from the EDT. This is the key improvement over clDice: thinner structures are automatically upweighted.

**Performance optimization:** Both the soft skeletonization and EDT are computed at reduced resolution (`topology_resolution=256` by default, configurable in `src/config.py`). This is a 16× pixel reduction that gives ~100× speedup for EDT and ~16× for skeletonization, with minimal topology quality loss since skeleton connectivity is a coarse property. Gradients flow through the bilinear interpolation used for downsampling.

**Memory note:** cbDice requires soft skeletonization on GPU plus EDT computation on CPU. Use `--no-cbdice` if you encounter OOM errors on MPS.

**Why weight 5.0:** Iteratively bumped. Run 1 used weight 1.0 (cbDice value sat at ~0.056). Run 3 bumped to 3.0 (cbDice dropped to ~0.016, then ~0.008). Run 5 bumped to 5.0 because user inspection of worst-IoU PNG dumps showed thin neuron branches still being broken. The high weight directly pressures connectivity in tree-like neurons.

### 8.6 Supervoxel Loss (weight = 0.3)

**Purpose:** Directly penalize split/merge errors in connectivity.

**The insight:** cbDice focuses on skeleton overlap with boundary weighting, but doesn't explicitly detect when a single connected structure becomes two disconnected components (split error). Supervoxel loss identifies "critical supervoxels" — connected components whose addition/removal changes the topology.

**Mathematical formulation:**

```
L = (1−α)L₀(y,ŷ) + αβ∑L₀(neg_critical) + α(1−β)∑L₀(pos_critical)
```

Where:
- L₀ is the base loss (BCE)
- α controls topology-aware vs standard loss balance (default: 0.5)
- β balances negatively vs positively critical components (default: 0.5)

**Two types of critical components:**
1. **Negatively critical (false negatives):** Missing pixels that cause splits — e.g., a 1-pixel gap breaking a neuron branch
2. **Positively critical (false positives):** Extra pixels that cause merges — e.g., a false bridge connecting two neurons

**Algorithm (O(n) complexity, scipy-accelerated):**
1. Binarize predictions at threshold 0.5
2. Downsample to `topology_resolution` (default 256×256) for fast connectivity analysis
3. Label connected components using `scipy.ndimage.label` (C-compiled, ~100× faster than Python BFS)
4. For each FN component: use `scipy.ndimage.binary_dilation` to find adjacent prediction labels — if 2+ distinct labels, the component is a critical bridge
5. For each FP component: remove from prediction, relabel with `scipy.ndimage.label`, check if neighbors split into different components
6. Upscale critical masks back to full resolution (1024×1024) for accurate loss gradients

**Why this helps neurons:** A 1-pixel gap may seem minor (standard losses barely notice), but if it splits a neuron into two components, supervoxel loss heavily penalizes it.

**Performance:** The original implementation used pure Python BFS with nested pixel loops over 1024×1024 images (~230s/step). Replacing with scipy.ndimage.label + topology downsampling reduces this to <0.1s/step (~2000× speedup). Loss gradients remain at full resolution for accuracy — only the critical region detection runs at lower resolution.

**Command line usage:**
```bash
# Default (enabled with weight 1.0)
python cli.py train --epochs 30

# Increase weight for more connectivity emphasis
python cli.py train --supervoxel-weight 2.0

# Disable only supervoxel
python cli.py train --no-supervoxel

# Disable BOTH cbDice and supervoxel (fast CPU/MPS training)
python cli.py train --no-topology
```

**Why weight 0.3 (small):** Was disabled in Runs 3–4 because the per-component value sat flat at ~0.30 — supervoxel was effectively a constant offset, not a signal. Run 5 re-enabled it at the smaller weight of 0.3 once two things changed: (a) the dataset grew to 16 → 20 videos (the flat-value behaviour may have been an undersampling artefact), and (b) Path B made forward-pass predictions temporally coherent, so the connectivity signal was actually learnable. Run 7 diagnostics confirm the val component does drop (val supervoxel ≈ 0.064 at the best epoch).

**Reference:** Grim et al., "Efficient Connectivity-Preserving Instance Segmentation with Supervoxel-Based Loss Function", AAAI 2025

### 8.7 Combined Loss

```
L_total = 5.0 * L_focal + 2.0 * L_dice + 5.0 * L_cbDice + 0.3 * L_supervoxel + 1.0 * L_iou + 1.0 * L_class
```

This is referred to as `core_loss` in the codebase.

**Topology-preserving losses (cbDice + Supervoxel):** Together, these two losses provide comprehensive topology protection:
- **cbDice:** "Does the predicted skeleton match the ground truth skeleton?" (with higher weight on thin structures via inverse-radius EDT)
- **Supervoxel:** "Does the predicted connectivity match the ground truth connectivity?"

A prediction that passes cbDice (good skeleton overlap) but fails supervoxel (split errors) will still be penalized. This combination is especially effective for thin, branching neuron structures.

**Fast training mode:** If training on CPU/MPS is too slow, disable both topology losses:
```bash
python cli.py train --no-topology
```
This is equivalent to `--no-cbdice --no-supervoxel` but more convenient.

---

## 9. Optimizer and Learning Rate Schedule

### 9.1 Optimizer: AdamW

We use AdamW (Adam with decoupled weight decay):

```
m_t = beta1 * m_{t-1} + (1-beta1) * g_t          # Momentum
v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2         # Squared gradient
m_hat = m_t / (1-beta1^t)                          # Bias correction
v_hat = v_t / (1-beta2^t)

theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + wd * theta_{t-1})
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^
                                                    decoupled weight decay
```

**Why AdamW over SGD:** Transformers are notoriously sensitive to learning rate and benefit from adaptive learning rates. AdamW's per-parameter adaptive rates handle the heterogeneous gradient scales across attention projections, feed-forward layers, and embeddings.

### 9.2 Three Learning Rate Groups (LoRA+)

We use LoRA+ (Hayou et al., ICML 2024), which uses asymmetric learning rates for the A and B matrices:

```python
lora_b_lr = config.optimizer.lora_lr * config.optimizer.lora_b_lr_ratio  # 1e-4 × 16 = 1.6e-3

optimizer = AdamW([
    {"params": lora_a_params,    "lr": 1e-4,   "weight_decay": 0.05},  # A matrices
    {"params": lora_b_params,    "lr": 1.6e-3, "weight_decay": 0.05},  # B matrices (16× higher)
    {"params": embedding_params, "lr": 1e-5,   "weight_decay": 0.0},   # Embeddings
], fused=True)
```

**LoRA A parameters (lr=1e-4):** Down-projection matrices, initialized with Kaiming uniform. They produce reasonable gradients from the start.

**LoRA B parameters (lr=1.6e-3, 16× higher):** Up-projection matrices, initialized to zeros. Because B starts at zero, gradients for B are much smaller than for A — they need a higher learning rate to overcome this gradient imbalance. This is the key insight of LoRA+: zero extra cost, faster convergence.

**Embedding parameters (lr=1e-5):** These are pretrained and already contain useful information. We want gentle, conservative updates — large steps could destroy their pretrained values.

**No weight decay for embeddings:** These are small vectors that encode specific information (e.g., temporal position). Weight decay would push them toward zero, which would erase that information. LoRA weights get weight_decay=0.05 to prevent them from growing too large.

**Fused AdamW:** The `fused=True` flag uses a single CUDA kernel for the optimizer step, vs ~136 small kernel launches with the default `foreach` impl. Numerically identical, just faster on GPU when there are many small parameter tensors (LoRA has 68 layers × 2 = 136 tiny adapter tensors).

### 9.3 Learning Rate Schedule

**Warmup phase** (2 epochs, ~3.3% of training):
```
lr(t) = lr_base * (min_lr_ratio + (1 - min_lr_ratio) * (t / warmup_steps))
```

Linear ramp from `0.05 × lr_base` to `lr_base`. This prevents large, destabilizing gradient updates at the start. We use 2 epochs (rather than 1) because LoRA-B is zero-initialized — the first gradient steps push it out of a sharp init region and benefit from a gentler ramp.

**Cosine annealing phase** (epochs 3 to 90):
```
progress = (t - warmup_steps) / (total_steps - warmup_steps)
lr(t) = lr_base * max(min_lr_ratio, 0.5 * (1 + cos(pi * progress)))
```

The learning rate smoothly decreases following a cosine curve, **floored at 5% of peak** (`min_lr_ratio = 0.05`):
- At the start (progress=0): lr = lr_base (full speed)
- At the midpoint (progress=0.5): lr = 0.5 * lr_base
- After the cosine drops below `min_lr_ratio × lr_base`: lr stays at the floor

**Why this exact shape (linear warmup → cosine → 5% floor):** This is GPT-3's recipe (Brown et al., 2020). They used 10% as the floor; we use 5% because in earlier runs the cosine bottomed out ~10 epochs before training ended and the flat 10% tail produced no improvements. Halving the floor lets the late phase run at finer LR steps.

The schedule is **a single global curve evaluated step-by-step** — there are no per-epoch resets or restarts. Same as GPT-3.

### 9.4 Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

If the L2 norm of all gradients exceeds 1.0, all gradients are scaled down proportionally:
```
if ||g|| > max_norm:
    g = g * (max_norm / ||g||)
```

**Why 1.0:** Gradient clipping prevents rare large gradient spikes from destabilizing training while allowing normal gradient flow.

### 9.5 Mixed Precision (AMP)

On CUDA GPUs:
```python
with torch.amp.autocast(device, dtype=torch.bfloat16):
    outputs = model(batch)
    loss = loss_fn(outputs, targets)
loss.backward()  # backward runs OUTSIDE autocast, in default precision
optimizer.step()
```

**bfloat16:** Half-precision floating point that preserves the full exponent range of float32 while halving memory. This allows:
- ~2× faster matrix multiplications on modern GPUs
- ~2× less memory for activations (crucial for 1024×1024 images)
- Minimal impact on training quality (unlike float16, bfloat16 doesn't suffer from underflow in gradients)

**No GradScaler.** Older training code uses `torch.cuda.amp.GradScaler` to handle fp16 underflow. Bfloat16 has the same exponent range as fp32, so underflow is impossible — GradScaler does nothing useful and adds per-step sync overhead. Removed from this codebase. This matches PyTorch's recommended bf16 pattern (`pytorch.org/docs/stable/notes/amp_examples.html`).

---

## 10. Training Loop Walkthrough

Here is the complete training flow, step by step:

### 10.1 Feature Precomputation (optional, off by default in Run 7)

SAM2's image encoder is mostly frozen, but Run 6 added LoRA adapters on the last 2 Hiera trunk blocks, so the encoder output is **no longer deterministic per frame** — adapter weights change every step. Feature precomputation would silently miss those adapter contributions.

For this reason, `precompute_features = False` in Run 7's config. The encoder runs every step. We also wanted the full pixel-level augmentation pipeline (rotation, shear, brightness, contrast), which precomputation drops in favour of feature-level flips only.

You can still opt back into precomputation with `precompute_features = True` for an A/B test against an encoder-frozen baseline (`--no-hiera-last-stage`), but it is not the shipping configuration.

### 10.2 Data Loading

```
1. Get list of video names from training_data/images/
2. Split into train (75%) and val (25%) sets — Run 7 default is 15 train / 5 val
3. Create NeuronVideoDataset for each (with clip_stride and cache_in_memory)
4. Each __getitem__ call:
   a. Pick a random clip (video_idx, start_frame) using clip stride
   b. Load 8 consecutive PNG frames and masks (from cache or disk)
   c. Apply augmentations (full pixel-level pipeline)
   d. Find object IDs present in clip
   e. Build VideoDatapoint with Frame/Object structures
   f. Return (VideoDatapoint, frame_ids) for feature cache lookup
5. Collate into (BatchedVideoDatapoint, frame_id_lists) via neuron_collate_fn
```

### 10.3 Forward Pass — Path B (`_forward_memory`, default since Run 5)

```
(batch, frame_id_lists) -> model (SAM2TrainingWrapper)

  1. Image encoder processes all frames -> features [B*T, C, H, W]
     (full pixel-level augmentation is applied at the dataset level;
      precomputation is off because the Hiera last-stage LoRA adapters
      mutate the encoder output every step.)
  2. Features are reshaped to video format [B, T, C, H, W]
  3. For each frame t = 0..T-1, call SAM2Base.track_step(...) with
     is_init_cond_frame = (t == 0):
       - t = 0: GT mask is the dense prompt; result stored under
         cond_frame_outputs.
       - t > 0: mask_inputs = None; memory_attention pulls from the
         memory bank's cond + non-cond entries to produce the
         conditioning, the SAM heads emit the mask, and the result is
         stored under non_cond_frame_outputs and re-encoded into the
         bank.
     Gradients flow through the bank (no detach), so the 40
     memory_attention LoRA adapters actually receive gradient.
  4. Return dict with 'pred_masks_high_res': [B, T, N_obj, H, W]
```

Path A (legacy, `--no-memory-attention`) is the simplified previous-mask-as-prompt loop used in Runs 1–4. It bypasses `memory_attention.forward` entirely; the memory_attention LoRA adapters get zero gradient on this path. Turning Path B on alone — without any other change — jumped Run 5's val IoU from Run 4's 0.7755 to ~0.8295.

**Critical wrapper fix.** When `use_memory_path=True`, the wrapper sets `model.pred_obj_scores = False` and `model.use_mask_input_as_output_without_sam = False`. Without these flags, `SAM2Base._forward_sam_heads` replaces masks with `NO_OBJ_SCORE` (~`-inf`) whenever its untrained obj-score head outputs <0. At init the score is near zero with sign noise, so half the predictions silently went to all-zero, dice became 1.0 trivially on those, val IoU collapsed to 0.0, and no gradient flowed. This bug is the reason Path B looked broken in early Run 5 attempts. Anyone touching the wrapper must preserve those two flag overrides.

We use `SAM2TrainingWrapper` (defined in `src/trainer.py`) which wraps the standard SAM2 model from the pip package. This is necessary because Meta's internal `SAM2Train` class is not included in the public pip package.

### 10.3 Loss Computation

```
outputs -> loss_fn (MultiStepMultiMasksAndIous)
  pred_masks = outputs['pred_masks']  # [B, T, N_obj, H, W]
  gt_masks = batch.masks

  # Pixel-level losses (computed at full 1024×1024 resolution)
  L_focal = focal_loss(pred_masks, gt_masks) * 5.0
  L_dice = dice_loss(pred_masks, gt_masks) * 2.0

  # Topology losses (computed at 256×256 for speed, ~2000× faster)
  L_cbdice = cbdice_loss(pred_masks, gt_masks) * 5.0   # Downsample → skeleton + EDT → cbDice
  L_supervoxel = supervoxel_loss(pred_masks, gt_masks) * 0.3  # Downsample → scipy label → upscale critical masks → loss at full res

  # Auxiliary losses
  L_iou = iou_loss(pred_iou, actual_iou) * 1.0
  L_class = bce_loss(pred_obj_score, obj_present) * 1.0

  core_loss = L_focal + L_dice + L_cbdice + L_supervoxel + L_iou + L_class
```

### 10.4 Backward Pass

```
1. optimizer.zero_grad()           -- Clear old gradients
2. loss.backward()                 -- Compute gradients (no GradScaler — bf16)
3. clip_grad_norm_(params, 1.0)    -- Clip large gradients
4. optimizer.step()                -- Update weights (fused AdamW)
5. scheduler.step()                -- Update learning rate (per-step cosine)
```

### 10.5 Validation

Every 2 epochs:
```
1. model.eval(), torch.no_grad()
2. For each validation batch:
   a. Forward pass
   b. Compute loss
   c. Threshold predictions at 0.0 -> binary masks
   d. Compute IoU and Dice vs ground truth
3. Average metrics across batches
4. If val_iou > best_val_iou:
   - Save LoRA weights to checkpoints/lora_weights.pt
   - Update best_val_iou
5. model.train()
```

### 10.6 Checkpointing

There are **two** checkpoint files:

**`lora_weights.pt`** — inference-only, saved when val IoU improves:
```python
{
    'lora_state_dict': {      # LoRA A and B matrices (both groups, dual-rank)
        "sam_mask_decoder.transformer.layers.0.self_attn.q_proj.lora_A.weight": tensor,
        "sam_mask_decoder.transformer.layers.0.self_attn.q_proj.lora_B.weight": tensor,
        ...
        "image_encoder.trunk.blocks.10.attn.qkv.lora_A.weight": tensor,  # Hiera group
        ...
    },
    'other_state_dict': {     # Unfrozen embeddings
        "maskmem_tpos_enc": tensor,
        "no_mem_embed": tensor,
        ...
    },
    'config': {'rank': 16, 'alpha': 32.0, 'hiera_rank': 8, 'hiera_alpha': 16.0}
}
```

This is ~25 MB at Run 7 sizes (vs ~150 MB for the full model). The loader regex-classifies each tensor name to decide which rank to instantiate.

**`train_state.pt`** — written every epoch for resume:

Contains everything in `lora_weights.pt` plus optimizer state, scheduler state, current epoch, best val IoU, and the early-stopping counter. Heavier (~40 MB extra from Adam moments) but lets a SLURM job pick up exactly where the previous one stopped. See Section 14's "Chained training" for the sbatch pattern.

---

## 11. Evaluation Metrics

### 11.1 Intersection over Union (IoU)

Also known as the Jaccard Index:

```
IoU = |A intersection B| / |A union B|
    = TP / (TP + FP + FN)
```

where:
- TP (true positives): pixels correctly labeled as neuron
- FP (false positives): background pixels incorrectly labeled as neuron
- FN (false negatives): neuron pixels missed by the prediction

IoU ranges from 0 (no overlap) to 1 (perfect match). It penalizes both missed regions and extra regions equally.

**Edge case:** If both prediction and ground truth are empty (no neuron in this frame), IoU = 1.0 (correct "nothing").

### 11.2 Dice Coefficient

Also known as F1-score:

```
Dice = 2 * |A intersection B| / (|A| + |B|)
     = 2 * TP / (2*TP + FP + FN)
```

Dice is related to IoU:
```
Dice = 2 * IoU / (1 + IoU)
```

Dice is always >= IoU for the same prediction. It's commonly used in medical image segmentation and provides a complementary view of quality.

### 11.3 Why Both?

IoU is the primary metric for checkpoint saving (more strict, better for comparing models). Dice is logged as a secondary metric because it's the standard in biomedical segmentation literature, making results comparable to published work.

---

## 12. Inference Integration

### 12.1 GUI Checkbox

The napari GUI adds a `use_finetuned` checkbox to the SAM2 propagation widget. When checked:

```python
def _load_sam2_predictor(self, use_finetuned, device):
    lora_path = Path("checkpoints/lora_weights.pt")

    if use_finetuned and lora_path.exists():
        # 1. Load base SAM 2.1 Hiera Tiny
        config_name, ckpt_path = _hf_download("facebook/sam2.1-hiera-tiny")
        predictor = build_sam2_video_predictor(
            config_file=config_name,
            ckpt_path=ckpt_path,
            device=device,
            mode="eval"
        )
        # 2. Apply LoRA structure (creates empty A/B matrices)
        apply_lora(predictor, target_patterns=SAM2_LORA_TARGETS)

        # 3. Load trained LoRA weights
        load_lora_weights(predictor, str(lora_path), device=device)

        # 4. Set to eval mode
        predictor.eval()
    else:
        # Use base model (no LoRA)
        predictor = build_sam2_video_predictor_hf(
            "facebook/sam2.1-hiera-tiny", device=device
        )

    return predictor
```

### 12.2 Model Caching

The predictor is cached in `self._sam2_predictor` with a `self._sam2_is_finetuned` flag. This avoids reloading the model every time you click "Propagate." The cache is invalidated when you switch between base and fine-tuned.

### 12.3 Graceful Fallback

If `use_finetuned` is checked but no checkpoint exists, the system falls back to the base model with a warning message. This means you can always propagate labels, even before training.

---

## 13. Hyperparameter Summary

`src/config.py` is the single source of truth. CLI flags only override when explicitly passed.

| Category | Parameter | Value | Rationale |
|----------|-----------|-------|-----------|
| **LoRA (main)** | rank | **16** | Bumped from 8 in Run 5 with the larger dataset for more adapter capacity (68 Linear layers wrapped) |
| | alpha | **32.0** | Scale = α/r = 2.0 (preserved as rank doubled) |
| | dropout | 0.10 | Light regularization (0.25 over-regularized; 0.05 underfit) |
| **LoRA (Hiera)** | hiera_last_stage | **True** | Run 6: LoRA on the last 2 Hiera trunk blocks (8 Linear layers in encoder visual features) |
| | hiera_rank | **8** | Smaller than main LoRA — 768-dim layers, more capacity per rank |
| | hiera_alpha | **16.0** | Scale = α/r = 2.0 (consistent) |
| **Data** | image_size | 1024 | SAM 2 native resolution |
| | num_frames | 8 | B×T = 16, GPU-friendly |
| | max_objects | 3 | Typical max neurons per field of view |
| | val_fraction | 0.25 | 5 of 20 videos held out — mean of 5 keeps the headline IoU stable run-to-run |
| | val_videos | None | When set, overrides val_fraction. Run 7 default: `A91_C1_2, A31_C1, A52_C2, A22_C2, A42_C1` |
| | clip_stride | 4 | Skip overlapping clips |
| | precompute_features | **False** | Full pixel-level augmentation enabled (rotation, shear, brightness, contrast); also required because Hiera last-stage LoRA mutates encoder output every step |
| | cache_in_memory | True | Load all frames into RAM at init |
| **Augmentation** | h_flip, v_flip | p=0.5 | Microscopy has no orientation preference |
| | rotation | ±25° | Neurons grow in arbitrary directions |
| | shear | ±20° | Simulates tissue deformation |
| | brightness | ±15% | Simulates illumination variation |
| | contrast | ±15% | Simulates microscope settings |
| **Optimizer** | lora_lr (A) | 1e-4 | Standard for LoRA A matrices |
| | lora_b_lr_ratio | 16.0 | LoRA+ ratio: B lr = A lr × 16 (= 1.6e-3) |
| | base_lr | 1e-5 | Conservative for pretrained embeddings |
| | weight_decay | **0.05** | Prevents LoRA weight explosion (was 0.1, lowered with rank tuning) |
| | max_grad_norm | 1.0 | Gradient clipping for stability |
| | warmup_epochs | **2** | Smooth start (was 1; bumped for LoRA-B zero init stability) |
| | min_lr_ratio | **0.05** | LR floor at 5% of peak (was 10%, lowered for finer late-phase steps) |
| | fused | True | CUDA fused AdamW kernel |
| **Loss** | focal weight | 5.0 | Pixel accuracy is primary objective |
| | dice weight | 2.0 | Region shape quality |
| | cbdice weight | **5.0** | Skeleton topology — bumped from 3.0 in Run 5 to push connectivity |
| | use_supervoxel | **True** | Re-enabled in Run 5 (small weight 0.3) with the larger dataset and Path B forward |
| | supervoxel weight | **0.3** | Small but non-zero; Run 7 diagnostics show val component ≈ 0.064 (it is learning, not flat) |
| | iou weight | 1.0 | Confidence calibration |
| | class weight | 1.0 | Anti-hallucination |
| | focal alpha | 0.25 | Balances class imbalance |
| | focal gamma | 2.0 | Focuses on hard examples |
| | topology_resolution | **256** | Resolution for topology losses (16× fewer pixels than 1024). Note: post-Run-7 experiments suggest this may be a ceiling for finer connectivity errors — see "Post-Run-7 experiments" below |
| **Training** | epochs | **90** | Run 7: 90 epochs × ~260s ≈ 6.5 h on B200 → 2 chained 4-hour SLURM jobs |
| | use_memory_attention_path | **True** | Path B: SAM 2's real track_step API; activates the 40 memory_attention LoRA adapters that were inactive in Runs 1–4 |
| | val_every_n_epochs | 2 | Validate every 2 epochs |
| | early_stopping_patience | 15 | Stop if no IoU improvement for 15 validation checks |
| | batch_size | 2 | Small batch → implicit regularization on small dataset |
| | seed | 42 | Reproducibility |
| | AMP | bfloat16 | 2× speed, 2× memory; no GradScaler needed |
| **B200 perf** | channels_last | True (auto) | Conv-heavy paths use cuDNN fast path |
| | pin_memory + non_blocking | True (auto) | H2D copies overlap with compute |
| | cuDNN SDPA backend | enabled (auto) | FlashAttention-3 on Blackwell |
| | compile_encoder / compile_decoder | False (opt-in flags) | Optional `torch.compile` |

**Post-Run-7 experiments (not the shipping config).** Run 8 tried rebalancing the loss (dice 2 → 1.5, cbDice 5 → 10, supervoxel 0.3 → 2.0) and lost 0.017 IoU (0.8401 → 0.8229). Diagnosis: the supervoxel component hit a structural ceiling at val ≈ 0.05 because connectivity is computed at 256 — finer breaks were invisible to the loss. Run 9 (in progress) bumps `topology_resolution: 256 → 384` to lift the ceiling and reverts the other weights to Run 7 values (only supervoxel kept slightly above Run 7 at 0.5). Run 7 remains the documented best.

---

## 14. Running the Pipeline

### Step 1: Create a manifest file

Create `data_manifest.json` listing your TIF pairs:

```json
[
    {
        "image_tif": "/Volumes/Adi-SSD/EfratData/.../A31 C2.tif",
        "mask_tif": "labeled_masks/A31_C2_masks.tif",
        "name": "A31_C2"
    },
    {
        "image_tif": "/Volumes/Adi-SSD/EfratData/.../A42 C1.tif",
        "mask_tif": "labeled_masks/A42_C1_masks.tif",
        "name": "A42_C1"
    }
]
```

### Step 2: Prepare training data

```bash
python cli.py prepare --manifest data_manifest.json --output-dir training_data
```

This converts TIF stacks to PNG frames, computes normalization statistics, and saves everything to `training_data/`.

### Step 3: Train

```bash
# Recommended — uses everything in src/config.py:
python cli.py train

# Maximum throughput on B200 (adds first-epoch compile cost ~30-90s):
python cli.py train --compile-encoder --compile-decoder

# Pin a specific val split (e.g. for split-rotation experiments):
python cli.py train --val-videos A11_C1,A12_C1,A52_C1

# Resume from a saved training state (used to chain SLURM jobs under
# a wall-clock limit — see "Chained training" below):
python cli.py train --resume checkpoints/train_state.pt
```

The startup banner echoes the resolved config so you can verify what's running.

#### Chained training across SLURM jobs

For training runs that exceed the cluster's wall-clock limit (e.g. Run 7's 90 epochs × ~260 s ≈ 6.5 h on a 4-hour partition), `--resume` lets a single logical run span multiple submissions. The training loop writes `checkpoints/train_state.pt` at the end of every epoch — this contains the model weights, optimiser moments, LR scheduler position, and the early-stopping counter.

A self-contained sbatch script that auto-detects whether to resume or start fresh:

```bash
STATE_FILE=checkpoints/train_state.pt
if [ -f "$STATE_FILE" ]; then
    python -u cli.py train --epochs 90 --resume "$STATE_FILE"
else
    python -u cli.py train --epochs 90
fi
```

Submit the script repeatedly until training reaches `num_epochs` or early stopping triggers. When you want to begin a **new** training run, delete `checkpoints/train_state.pt` manually first — otherwise the script silently picks up the old one.

### Step 4: Use in napari

After training, `checkpoints/lora_weights.pt` will exist. Open the labeling tool, check **"Use fine-tuned model"**, click **"SAM2: Propagate from frame 0"**.

---

## 15. File Reference

| File | Purpose | Key Functions |
|------|---------|---------------|
| `cli.py` | CLI entry point | `cmd_prepare()`, `cmd_train()` |
| `src/config.py` | All hyperparameters as dataclasses | `TrainingConfig`, `LoRAConfig` |
| `src/prepare_data.py` | TIF to PNG conversion + stats | `prepare_dataset()`, `prepare_single_video()` |
| `src/lora.py` | LoRA adapter implementation (dual-rank-aware load) | `LoRALinear`, `apply_lora()`, `save/load_lora_weights()`, `save/load_train_state()` |
| `src/data_utils.py` | Data containers for training | `Frame`, `Object`, `VideoDatapoint`, `BatchedVideoDatapoint`, `collate_fn()` |
| `src/loss_fns.py` | Loss function implementations | `FocalLoss`, `DiceLoss`, `SoftCBDiceLoss`, `SupervoxelLoss2D`, `MultiStepMultiMasksAndIous` |
| `src/neuron_dataset.py` | PyTorch Dataset for video clips | `NeuronVideoDataset`, `neuron_collate_fn()` |
| `src/augmentations.py` | Video augmentation pipeline | `VideoAugmentation`, `VideoValidationTransform` |
| `src/trainer.py` | Training loop + model wrapper | `SAM2TrainingWrapper`, `train()`, `build_training_model()` |
| `src/evaluate.py` | Validation metrics | `compute_iou()`, `compute_dice()`, `validate_epoch()` |
| `src/gui.py` | Napari GUI with SAM2 integration | `_load_sam2_predictor()`, `_run_sam2()` |

---

## References

- Hu, E.J. et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
- Hayou, S. et al. "LoRA+: Efficient Low Rank Adaptation of Large Models." ICML 2024.
- Ravi, N. et al. "SAM 2: Segment Anything in Images and Videos." Meta AI, 2024.
- Lin, T.Y. et al. "Focal Loss for Dense Object Detection." ICCV 2017.
- Milletari, F. et al. "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation." 3DV 2016. (Dice loss)
- Shi, P. et al. "centerline boundary Dice loss for vascular segmentation." MICCAI 2024. (cbDice)
- Shit, S. et al. "clDice - A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation." CVPR 2021. (predecessor to cbDice)
- Grim, M. et al. "Efficient Connectivity-Preserving Instance Segmentation with Supervoxel-Based Loss Function." AAAI 2025.
- Loshchilov, I. & Hutter, F. "Decoupled Weight Decay Regularization." ICLR 2019. (AdamW)
