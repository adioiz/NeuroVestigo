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

**Our goal:** Fine-tune SAM 2 on our 10 previously labeled neuron datasets so that propagation quality improves, while preserving SAM 2's general video understanding capabilities.

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

**Decision: Freeze the image encoder.** Retraining it on 10 videos would destroy its general features (catastrophic forgetting) and waste compute. The encoder already produces good spatial features; our problem is in *how those features are interpreted across time*, not in how individual frames are encoded.

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

We have only 10 labeled videos (~720 frames total). This is a tiny dataset by deep learning standards. We need a fine-tuning strategy that:

1. **Avoids overfitting** -- with only 10 videos, a fully fine-tuned model would memorize training data
2. **Preserves pretrained knowledge** -- SAM 2's video understanding should remain intact
3. **Is parameter-efficient** -- small checkpoint size, fast training
4. **Works on consumer GPUs** -- single NVIDIA GPU, not a server cluster

### Alternatives Considered

| Strategy | Trainable Params | Risk of Overfitting | Preserves Pretrained? |
|----------|-----------------|---------------------|-----------------------|
| Full fine-tuning | ~38M (100%) | Very high | No |
| Fine-tune decoder only | ~10M (26%) | High | Partially |
| **LoRA (our choice)** | **~1-3M (3-8%)** | **Low** | **Yes** |
| Train from scratch | ~38M (100%) | Impossible (too little data) | N/A |
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
- **r** is the rank (a small integer, we use r=8)
- **alpha** is a scaling factor (we use alpha=16)

### 4.2 Why This Works

The key insight from the LoRA paper (Hu et al., 2021) is that the *change* needed to adapt a pretrained model to a new task often has **low intrinsic rank**. That is, the adaptation delta-W = W_new - W_0 can be well-approximated by a low-rank matrix B * A.

**Parameter count comparison** for a single layer:

| Method | Parameters |
|--------|-----------|
| Full weight matrix (d_out x d_in) | d_out * d_in |
| LoRA (rank r) | r * (d_out + d_in) |

For a typical attention projection (256 x 256):
- Full: 256 * 256 = 65,536 parameters
- LoRA rank 8: 8 * (256 + 256) = 4,096 parameters (**16x reduction**)

### 4.3 The Scaling Factor

The LoRA output is scaled by alpha/r:

```
scale = alpha / r = 16 / 8 = 2.0
```

This controls *how much* the adaptation influences the output. The ratio alpha/r is the effective scaling:
- **Higher ratio** (e.g., alpha=32, r=8 -> scale=4): Stronger adaptation, faster learning, but risks instability
- **Lower ratio** (e.g., alpha=8, r=8 -> scale=1): Gentler adaptation, more stable, but may need more epochs
- **Our choice** (alpha=16, r=8 -> scale=2): A moderate balance. The adaptation has meaningful influence without dominating the pretrained features.

### 4.4 Initialization

This is critical for stable training:
- **A is initialized** with Kaiming uniform (random): provides diverse input projections from the start
- **B is initialized** to zeros: at the start of training, B * A = 0, so the model behaves *exactly* like the pretrained model

This means training begins from the pretrained model's behavior and gradually learns the adaptation. The model never "forgets" its pretrained state at initialization.

### 4.5 Dropout

We apply dropout (p=0.05) to the LoRA path:

```
y = W_0 * x + scale * B(A(dropout(x)))
```

This acts as regularization: during training, random input dimensions are zeroed out with 5% probability, forcing the adaptation to be robust and preventing overfitting to specific input patterns. This is especially important with only 10 training videos.

### 4.6 Code Reference (lora.py)

```python
class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, rank=8, alpha=16.0, dropout=0.0):
        self.original = original                           # Frozen W_0
        self.lora_A = nn.Linear(in_features, rank, bias=False)   # A: (d_in -> r)
        self.lora_B = nn.Linear(rank, out_features, bias=False)  # B: (r -> d_out)
        self.scale = alpha / rank                          # = 2.0
        self.lora_dropout = nn.Dropout(p=dropout)          # p = 0.05

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

**Image encoder (Hiera):** Frozen entirely. Its spatial features are general-purpose and excellent. Adapting it with 10 videos would cause overfitting. The encoder has ~25M parameters -- keeping them frozen is also a major compute saving.

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

### 8.5 cbDice Loss (weight = 1.0)

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

### 8.6 Supervoxel Loss (weight = 1.0) — NEW!

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

**Reference:** Grim et al., "Efficient Connectivity-Preserving Instance Segmentation with Supervoxel-Based Loss Function", AAAI 2025

### 8.7 Combined Loss

```
L_total = 5.0 * L_focal + 2.0 * L_dice + 1.0 * L_cbDice + 1.0 * L_supervoxel + 1.0 * L_iou + 1.0 * L_class
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
    {"params": lora_a_params,    "lr": 1e-4,  "weight_decay": 0.1},  # A matrices
    {"params": lora_b_params,    "lr": 1.6e-3, "weight_decay": 0.1},  # B matrices (16× higher)
    {"params": embedding_params, "lr": 1e-5,  "weight_decay": 0.0},  # Embeddings
])
```

**LoRA A parameters (lr=1e-4):** Down-projection matrices, initialized with Kaiming uniform. They produce reasonable gradients from the start.

**LoRA B parameters (lr=1.6e-3, 16× higher):** Up-projection matrices, initialized to zeros. Because B starts at zero, gradients for B are much smaller than for A — they need a higher learning rate to overcome this gradient imbalance. This is the key insight of LoRA+: zero extra cost, faster convergence.

**Embedding parameters (lr=1e-5):** These are pretrained and already contain useful information. We want gentle, conservative updates -- large steps could destroy their pretrained values.

**No weight decay for embeddings:** These are small vectors that encode specific information (e.g., temporal position). Weight decay would push them toward zero, which would erase that information. LoRA weights get weight_decay=0.1 to prevent them from growing too large.

### 9.3 Learning Rate Schedule

**Warmup phase** (1 epoch):
```
lr(t) = lr_base * (t / warmup_steps)
```

Linear ramp from 0 to lr_base. This prevents large, destabilizing gradient updates at the start when the model hasn't settled into the new loss landscape.

**Cosine annealing phase** (epochs 2 to 60):
```
progress = (t - warmup_steps) / (total_steps - warmup_steps)
lr(t) = lr_base * 0.5 * (1 + cos(pi * progress))
```

The learning rate smoothly decreases following a cosine curve:
- At the start (progress=0): lr = lr_base (full speed)
- At the midpoint (progress=0.5): lr = 0.5 * lr_base
- At the end (progress=1): lr approaches 0

**Why cosine over step decay:** Cosine is smooth (no sudden drops that can destabilize training) and empirically produces better convergence in transformer fine-tuning. The gradual decay in the final epochs allows the model to settle into a local minimum without overshooting.

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
```

**bfloat16:** Half-precision floating point that preserves the full exponent range of float32 while halving memory. This allows:
- ~2x faster matrix multiplications on modern GPUs
- ~2x less memory for activations (crucial for 1024x1024 images)
- Minimal impact on training quality (unlike float16, bfloat16 doesn't suffer from underflow in gradients)

The GradScaler handles any remaining numerical issues by dynamically scaling the loss before backward pass.

---

## 10. Training Loop Walkthrough

Here is the complete training flow, step by step:

### 10.1 Feature Precomputation (Before Training Loop)

SAM2's image encoder (Hiera backbone + FPN + conv_s0/conv_s1) is frozen during training (no LoRA on Conv2d layers), so its output is deterministic for each input frame. Computing it once and caching eliminates ~60-70% of per-step compute.

```
1. Before training, run the encoder on every frame once (~3-5 min for 876 frames)
2. Cache the 3-level FPN features in float16 on CPU (~4-8 MB/frame)
3. Store positional encodings once (they're purely spatial, independent of content)
4. During training, assemble cached features into backbone_out and skip the encoder
```

**Impact:** ~80 sec/step -> ~8 sec/step (10x speedup). 5 epochs in ~55 min instead of ~36 hours.

### 10.2 Data Loading

```
1. Get list of video names from training_data/images/
2. Split into train (80%) and val (20%) sets
3. Create NeuronVideoDataset for each (with clip_stride and cache_in_memory)
4. Each __getitem__ call:
   a. Pick a random clip (video_idx, start_frame) using clip stride
   b. Load 8 consecutive PNG frames and masks (from cache or disk)
   c. Apply augmentations (validation transform when precomputing)
   d. Find object IDs present in clip
   e. Build VideoDatapoint with Frame/Object structures
   f. Return (VideoDatapoint, frame_ids) for feature cache lookup
5. Collate into (BatchedVideoDatapoint, frame_id_lists) via neuron_collate_fn
```

### 10.3 Forward Pass

```
(batch, frame_id_lists) -> model (SAM2TrainingWrapper)

  With precomputed features (default):
    1. Assemble backbone_out from cached per-frame features
    2. Optionally flip feature maps (feature-level augmentation)
    3. Flip masks to match
    4. Skip image encoder entirely

  Without precomputation:
    1. Image encoder processes all frames -> features [B*T, C, H, W]

  Then (both paths):
    2. Features are reshaped to video format [B, T, C, H, W]
    3. For each object, propagate through time:
       - Frame 0: Use GT mask as prompt
       - Frames 1-7: Use previous prediction as memory
       - Mask decoder produces prediction for each frame
    4. Return dict with 'pred_masks': [B, T, N_obj, H, W]
```

Note: We use `SAM2TrainingWrapper` (defined in `src/trainer.py`) which wraps the standard SAM2 model from the pip package. This is necessary because Meta's internal `SAM2Train` class is not included in the public pip package.

### 10.3 Loss Computation

```
outputs -> loss_fn (MultiStepMultiMasksAndIous)
  pred_masks = outputs['pred_masks']  # [B, T, N_obj, H, W]
  gt_masks = batch.masks

  # Pixel-level losses (computed at full 1024×1024 resolution)
  L_focal = focal_loss(pred_masks, gt_masks) * 5.0
  L_dice = dice_loss(pred_masks, gt_masks) * 2.0

  # Topology losses (computed at 256×256 for speed, ~2000× faster)
  L_cbdice = cbdice_loss(pred_masks, gt_masks) * 1.0   # Downsample → skeleton + EDT → cbDice
  L_supervoxel = supervoxel_loss(pred_masks, gt_masks) * 1.0  # Downsample → scipy label → upscale critical masks → loss at full res

  # Auxiliary losses
  L_iou = iou_loss(pred_iou, actual_iou) * 1.0
  L_class = bce_loss(pred_obj_score, obj_present) * 1.0

  core_loss = L_focal + L_dice + L_cbdice + L_supervoxel + L_iou + L_class
```

### 10.4 Backward Pass

```
1. optimizer.zero_grad()           -- Clear old gradients
2. scaler.scale(loss).backward()   -- Compute gradients (with AMP scaling)
3. scaler.unscale_(optimizer)      -- Unscale for gradient clipping
4. clip_grad_norm_(params, 1.0)    -- Clip large gradients
5. scaler.step(optimizer)          -- Update weights
6. scaler.update()                 -- Adjust AMP scale factor
7. scheduler.step()                -- Update learning rate
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

The checkpoint file (`lora_weights.pt`) contains only:
```python
{
    'lora_state_dict': {      # LoRA A and B matrices
        "sam_mask_decoder.transformer.layers.0.self_attn.q_proj.lora_A.weight": tensor,
        "sam_mask_decoder.transformer.layers.0.self_attn.q_proj.lora_B.weight": tensor,
        ...
    },
    'other_state_dict': {     # Unfrozen embeddings
        "maskmem_tpos_enc": tensor,
        "no_mem_embed": tensor,
        ...
    },
    'config': {'rank': 8, 'alpha': 16.0}
}
```

This is typically 1-2 MB (vs ~150 MB for the full model), because we only save the adaptation, not the base model.

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

| Category | Parameter | Value | Rationale |
|----------|-----------|-------|-----------|
| **LoRA** | rank | 8 | Sufficient expressiveness for 10-video adaptation |
| | alpha | 16.0 | Scale=2.0, moderate adaptation strength |
| | dropout | 0.05 | Light regularization for small dataset |
| **Data** | image_size | 1024 | SAM 2 native resolution |
| | num_frames | 8 | Balance: long enough for temporal learning, short enough for GPU memory |
| | max_objects | 3 | Typical max neurons per field of view |
| | val_fraction | 0.2 | 2 videos for validation, 8 for training |
| | clip_stride | 4 | Skip overlapping clips: 329 -> ~82 steps/epoch |
| | precompute_features | True | 10x speedup by caching frozen encoder output |
| | cache_in_memory | True | Load all frames into RAM at init, eliminating disk I/O |
| **Augmentation** | h_flip, v_flip | p=0.5 | Microscopy has no orientation preference |
| | rotation | +/-25 deg | Neurons grow in arbitrary directions |
| | shear | +/-20 deg | Simulates tissue deformation |
| | brightness | +/-15% | Simulates illumination variation |
| | contrast | +/-15% | Simulates microscope settings |
| **Optimizer** | lora_lr (A) | 1e-4 | Standard for LoRA A matrices |
| | lora_b_lr_ratio | 16.0 | LoRA+ ratio: B lr = A lr × 16 (= 1.6e-3) |
| | base_lr | 1e-5 | Conservative for pretrained embeddings |
| | weight_decay | 0.1 | Prevents LoRA weight explosion |
| | max_grad_norm | 1.0 | Gradient clipping for stability |
| | warmup | 1 epoch | Smooth start for frozen-backbone fine-tuning |
| **Loss** | focal weight | 5.0 | Pixel accuracy is primary objective |
| | dice weight | 2.0 | Region shape quality |
| | cbdice weight | 1.0 | Skeleton topology preservation |
| | supervoxel weight | 1.0 | Connectivity preservation |
| | iou weight | 1.0 | Confidence calibration |
| | class weight | 1.0 | Anti-hallucination |
| | focal alpha | 0.25 | Balances class imbalance |
| | focal gamma | 2.0 | Focuses on hard examples |
| | topology_resolution | 256 | Resolution for topology losses (16× fewer pixels than 1024) |
| **Training** | epochs | 60 | Sufficient for convergence with cosine schedule |
| | batch_size | 1 | GPU memory constraint (1024x1024, 8 frames) |
| | seed | 42 | Reproducibility |
| | AMP | bfloat16 | 2x speed, 2x memory savings on CUDA |

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
python3 cli.py prepare --manifest data_manifest.json --output-dir training_data
```

This converts TIF stacks to PNG frames, computes normalization statistics, and saves everything to `training_data/`.

### Step 3: Train

```bash
# Default settings (recommended first run):
python3 cli.py train --data-dir training_data

# Custom settings:
python3 cli.py train --epochs 80 --lora-rank 16 --data-dir training_data

# Force ImageNet normalization (not recommended):
python3 cli.py train --use-imagenet-stats
```

### Step 4: Use in napari

After training, `checkpoints/lora_weights.pt` will exist. Open your labeling tool:

```bash
python3 main.py /path/to/your/tif
```

In the SAM2 propagation panel, check the "Use fine-tuned model" checkbox. Click propagate. The model will automatically load and apply the LoRA weights.

---

## 15. File Reference

| File | Purpose | Key Functions |
|------|---------|---------------|
| `cli.py` | CLI entry point | `cmd_prepare()`, `cmd_train()` |
| `src/config.py` | All hyperparameters as dataclasses | `TrainingConfig`, `LoRAConfig` |
| `src/prepare_data.py` | TIF to PNG conversion + stats | `prepare_dataset()`, `prepare_single_video()` |
| `src/lora.py` | LoRA adapter implementation | `LoRALinear`, `apply_lora()`, `save/load_lora_weights()` |
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
