# Complete Code Walkthrough: SAM 2 LoRA Fine-Tuning

This document explains **every file, every function, and every important line** with the mathematical and logical background. You should be able to read this and explain every aspect to your professor.

---

# Table of Contents

1. [train.py (Entry Point)](#file-1-trainpy---entry-point)
2. [config.py (Configuration)](#file-2-configpy---configuration)
3. [lora.py (LoRA Implementation)](#file-3-lorapy---lora-implementation)
4. [data_utils.py (Data Containers)](#file-4-data_utilspy---data-containers)
5. [loss_fns.py (Loss Functions)](#file-45-loss_fnspy---loss-functions)
6. [prepare_data.py (Data Conversion)](#file-5-prepare_datapy---data-conversion)
7. [neuron_dataset.py (PyTorch Dataset)](#file-6-neuron_datasetpy---pytorch-dataset)
8. [augmentations.py (Data Augmentation)](#file-7-augmentationspy---data-augmentation)
9. [train.py (Training Loop)](#file-8-srctrainingtrainpy---training-loop)
10. [evaluate.py (Metrics)](#file-9-evaluatepy---metrics)

---

# File 1: `train.py` - Entry Point

This is the command-line interface (CLI) that you run.

## Full Code with Explanations

```python
#!/usr/bin/env python3
```
**Line 1:** Shebang line. Tells Unix systems to run this with Python 3.

```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```
**Line 34:** Fixes a macOS bug where Intel MKL and OpenMP conflict. Without this, you'd get a crash on Mac.

---

## Function: `cmd_prepare(args)`

```python
def cmd_prepare(args):
    """Prepare training data from TIF stacks."""
    from src.training.prepare_data import prepare_dataset

    with open(args.manifest) as f:
        manifest = json.load(f)

    prepare_dataset(
        manifest=manifest,
        output_dir=args.output_dir,
        label_id=args.label_id,
    )
```

**Purpose:** Converts your TIF files to PNG format for training.

**Line by line:**
- `from src.training.prepare_data import prepare_dataset` — Imports the conversion function (lazy import, only when needed)
- `with open(args.manifest) as f:` — Opens your manifest.json file
- `manifest = json.load(f)` — Parses JSON into a Python list of dicts
- `prepare_dataset(...)` — Calls the conversion function

---

## Function: `cmd_train(args)`

```python
def cmd_train(args):
    from src.training.config import (
        TrainingConfig, LoRAConfig, DataConfig, OptimizerConfig, AugmentationConfig,
    )
    from src.training.train import train

    aug_config = AugmentationConfig()
    if args.use_imagenet_stats:
        aug_config.normalize_mean = [0.485, 0.456, 0.406]
        aug_config.normalize_std = [0.229, 0.224, 0.225]

    config = TrainingConfig(
        num_epochs=args.epochs,
        lora=LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha),
        ...
    )

    train(config)
```

**Purpose:** Builds the configuration object and starts training.

**Key decision (lines 61-65):**
```python
if args.use_imagenet_stats:
    aug_config.normalize_mean = [0.485, 0.456, 0.406]
    aug_config.normalize_std = [0.229, 0.224, 0.225]
# else: normalize_mean/std remain None → auto-loaded from manifest
```

**Logic:**
- If you pass `--use-imagenet-stats`, we use ImageNet normalization
- Otherwise, `normalize_mean/std` stay as `None`
- Later, `train.py` sees `None` and auto-loads your dataset's actual mean/std from manifest.json
- **Why this matters:** Your microscopy images are dark and grayscale. ImageNet stats are for bright, colorful photos. Using wrong stats hurts training.

---

## Function: `main()` — Argument Parsing

```python
parser = argparse.ArgumentParser(...)
subparsers = parser.add_subparsers(dest="command", required=True)
```

**Concept:** We have two subcommands: `prepare` and `train`. Like `git commit` vs `git push` — same program, different actions.

**Key arguments explained:**

```python
p_train.add_argument("--epochs", type=int, default=60)
```
60 epochs — how many times we go through all training data. More = better learning but slower.

```python
p_train.add_argument("--batch-size", type=int, default=1)
```
Batch size 1 — we process 1 video clip at a time. Why? Memory constraint (1024×1024 images × 8 frames = lots of RAM).

```python
p_train.add_argument("--lora-rank", type=int, default=8)
```
LoRA rank 8 — the "r" in LoRA. Lower = fewer parameters, less capacity. Higher = more parameters, risk of overfitting.

```python
p_train.add_argument("--lora-alpha", type=float, default=16.0)
```
LoRA alpha 16 — scaling factor. Combined with rank: scale = alpha/rank = 16/8 = 2.0

```python
p_train.add_argument("--lora-lr", type=float, default=1e-4)
p_train.add_argument("--base-lr", type=float, default=5e-6)
```
**Two learning rates:**
- `lora-lr = 1e-4 = 0.0001` — for LoRA parameters (higher, they start at zero)
- `base-lr = 5e-6 = 0.000005` — for embeddings (lower, they're already pretrained)

---

# File 2: `config.py` - Configuration

All hyperparameters organized as Python dataclasses.

## LoRAConfig

```python
@dataclass
class LoRAConfig:
    rank: int = 8           # r in the LoRA paper
    alpha: float = 16.0     # α scaling factor
    dropout: float = 0.05   # Regularization
```

**Mathematical background:**

The LoRA forward pass is:
```
y = W₀·x + (α/r)·B·A·x
```

- `rank = 8`: Matrix A has shape (8 × input_dim), B has shape (output_dim × 8)
- `alpha = 16.0`: Scaling factor
- `scale = alpha/rank = 16/8 = 2.0`: How much LoRA influences output
- `dropout = 0.05`: 5% of inputs are randomly zeroed during training (regularization)

---

## DataConfig

```python
@dataclass
class DataConfig:
    training_data_dir: str = "training_data"
    num_frames: int = 8        # Frames per clip
    image_size: int = 1024     # SAM 2 input size (fixed!)
    max_objects: int = 3       # Max neurons tracked simultaneously
    val_fraction: float = 0.2  # 20% for validation
```

**Why `num_frames = 8`?**

Your 72-frame video is split into overlapping 8-frame "clips":
```
Frames:  [0][1][2][3][4][5][6][7][8]...[71]
Clip 1:  [0][1][2][3][4][5][6][7]
Clip 2:     [1][2][3][4][5][6][7][8]
...
```

Number of clips = 72 - 8 + 1 = 65 clips per video.

**Why 8 and not more?**
- 8 frames × 1024×1024 × 3 channels = ~100 MB per clip
- Model activations add ~4-8 GB
- Batch size 1 with 8 frames fits in ~10 GB VRAM

**Why `image_size = 1024`?**
SAM 2 was pretrained at 1024×1024. Using a different size would break the positional encodings.

---

## AugmentationConfig

```python
@dataclass
class AugmentationConfig:
    horizontal_flip_p: float = 0.5    # 50% chance of flipping
    vertical_flip_p: float = 0.5
    rotation_degrees: float = 25.0    # ±25 degrees
    shear: float = 20.0               # ±20 degrees
    brightness: float = 0.15          # ±15%
    contrast: float = 0.15            # ±15%
    normalize_mean: Optional[List[float]] = None  # Auto-load
    normalize_std: Optional[List[float]] = None   # Auto-load
```

**Why these values?**

- **Flips (p=0.5):** Neurons have no orientation preference. Flipping doubles effective dataset.
- **Rotation (±25°):** Neurons grow in any direction. Makes model rotation-invariant.
- **Shear (±20°):** Simulates tissue deformation.
- **Brightness/Contrast (±15%):** Simulates microscope illumination variation. Conservative because microscopy has specific intensity characteristics.

---

## OptimizerConfig

```python
@dataclass
class OptimizerConfig:
    base_lr: float = 5e-6     # For embeddings (pretrained)
    lora_lr: float = 1e-4     # For LoRA (new)
    weight_decay: float = 0.1
    max_grad_norm: float = 0.1
    warmup_epochs: int = 3
    use_cosine_schedule: bool = True
```

**Why two learning rates?**

LoRA parameters start at ~0 (B matrix is initialized to zeros). They need larger steps to have any effect.

Embedding parameters are already pretrained with useful values. Large steps would destroy them.

**Ratio:** `lora_lr / base_lr = 1e-4 / 5e-6 = 20×`

**Why `max_grad_norm = 0.1`?**

Gradient clipping. If gradients get too large (exploding gradients), we scale them down:
```
if ||gradients|| > 0.1:
    gradients = gradients × (0.1 / ||gradients||)
```

This prevents training instability.

---

## LossConfig

```python
@dataclass
class LossConfig:
    loss_mask_weight: float = 20.0   # Focal loss weight
    loss_dice_weight: float = 1.0    # Dice loss weight
    loss_iou_weight: float = 1.0     # IoU prediction loss
    loss_class_weight: float = 1.0   # Object presence loss
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
```

**Combined loss:**
```
L_total = 20·L_focal + 1·L_dice + 1·L_iou + 1·L_class
```

**Why weight 20 for focal loss?**
Pixel accuracy is the primary objective. Meta uses weight 20 in their SAM 2 training configs.

**Focal loss parameters:**
- `alpha = 0.25`: Balances foreground/background
- `gamma = 2.0`: How much to down-weight easy examples

---

# File 3: `lora.py` - LoRA Implementation

This is the core innovation — efficient fine-tuning.

## Class: `LoRALinear`

```python
class LoRALinear(nn.Module):
    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original           # The frozen pretrained weights
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank          # = 16/8 = 2.0

        in_features = original.in_features
        out_features = original.out_features

        # Low-rank matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)   # Down-project
        self.lora_B = nn.Linear(rank, out_features, bias=False)  # Up-project

        # Dropout for regularization
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # CRITICAL: Initialize A randomly, B as zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        # Freeze original weights
        for p in self.original.parameters():
            p.requires_grad = False
```

**Mathematical explanation:**

Original linear layer: `y = W·x` where W is (out_features × in_features)

LoRA adds: `y = W₀·x + (α/r)·B·A·x`

**Shapes:**
- `W₀`: (256 × 256) = 65,536 parameters (frozen)
- `A`: (8 × 256) = 2,048 parameters (trainable)
- `B`: (256 × 8) = 2,048 parameters (trainable)
- Total trainable: 4,096 (16× reduction!)

**Why `B = zeros`?**

At initialization:
```
y = W₀·x + scale·B·A·x
  = W₀·x + scale·0·A·x    (B is zeros)
  = W₀·x                   (exactly the pretrained model!)
```

Training starts from the exact pretrained behavior. Gradients then slowly move B away from zero.

**Why Kaiming init for A?**

Kaiming initialization preserves variance through layers. Formula:
```
std = sqrt(2 / fan_in)
```
where `fan_in` is input dimension. This prevents vanishing/exploding gradients.

---

## Forward Pass

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    original_out = self.original(x)
    lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
    return original_out + self.scale * lora_out
```

**Step by step:**
1. `original_out = self.original(x)` — Compute W₀·x (frozen path)
2. `self.lora_dropout(x)` — Apply dropout to input (5% of values → 0)
3. `self.lora_A(...)` — Down-project: (batch, in_features) → (batch, rank)
4. `self.lora_B(...)` — Up-project: (batch, rank) → (batch, out_features)
5. `return original_out + self.scale * lora_out` — Combine with scaling

**Dropout explanation:**

During training, 5% of input dimensions are randomly set to 0. This:
- Forces the network to not rely on any single feature
- Acts as regularization (prevents overfitting)
- Is disabled during inference (model.eval())

---

## SAM2_LORA_TARGETS — Which Layers to Adapt

```python
SAM2_LORA_TARGETS = [
    # Mask decoder transformer attention layers
    r"sam_mask_decoder\.transformer\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)",
    r"sam_mask_decoder\.transformer\.layers\.\d+\.cross_attn_token_to_image\.(q_proj|k_proj|v_proj|out_proj)",
    r"sam_mask_decoder\.transformer\.layers\.\d+\.cross_attn_image_to_token\.(q_proj|k_proj|v_proj|out_proj)",
    r"sam_mask_decoder\.transformer\.final_attn_token_to_image\.(q_proj|k_proj|v_proj|out_proj)",
    # Memory attention layers
    r"memory_attention\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)",
    r"memory_attention\.layers\.\d+\.cross_attn_image\.(q_proj|k_proj|v_proj|out_proj)",
    r"memory_attention\.layers\.\d+\.(linear1|linear2)",
]
```

**These are regex patterns.** Let me decode one:

```
r"sam_mask_decoder\.transformer\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)"
```

- `sam_mask_decoder` — The mask decoder module
- `\.transformer\.layers\.` — Its transformer layers
- `\d+` — Any digit (layer 0, 1, 2, ...)
- `\.self_attn\.` — Self-attention submodule
- `(q_proj|k_proj|v_proj|out_proj)` — One of the four projection layers

**What are Q, K, V, Out?**

In attention:
```
Q = W_q · x    (Query: "What am I looking for?")
K = W_k · x    (Key: "What do I represent?")
V = W_v · x    (Value: "What information do I carry?")

Attention = softmax(Q·Kᵀ / √d) · V

Output = W_out · Attention
```

By adapting all four projections, we give the model full flexibility to relearn attention patterns for neurons.

---

## Function: `apply_lora()`

```python
def apply_lora(
    model: nn.Module,
    target_patterns: Optional[List[str]] = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> Set[str]:
    if target_patterns is None:
        target_patterns = SAM2_LORA_TARGETS

    wrapped_names = set()
    named_modules = dict(model.named_modules())

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not _matches_any_pattern(name, target_patterns):
            continue

        # Split name into parent path and attribute name
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = named_modules[parent_name]
        else:
            parent = model
            attr_name = parts[0]

        # Replace the Linear with LoRALinear
        lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attr_name, lora_layer)

        wrapped_names.add(name)

    return wrapped_names
```

**What this does:**
1. Iterates through ALL modules in the model
2. For each `nn.Linear` that matches our target patterns:
   - Find its parent module
   - Replace it with a `LoRALinear` wrapper
3. Returns names of wrapped layers (for logging)

**The tricky part (lines 120-126):**

```python
parts = name.rsplit(".", 1)  # Split from right, max 1 split
# "sam_mask_decoder.transformer.layers.0.self_attn.q_proj"
# → ["sam_mask_decoder.transformer.layers.0.self_attn", "q_proj"]
```

We need the parent (`self_attn`) to replace its child (`q_proj`) with our LoRA wrapper.

---

## Functions: `save_lora_weights()` and `load_lora_weights()`

```python
def save_lora_weights(model: nn.Module, path: str):
    lora_state = {}
    other_state = {}

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A.weight"] = module.lora_A.weight.data.cpu()
            lora_state[f"{name}.lora_B.weight"] = module.lora_B.weight.data.cpu()

    # Also save unfrozen embeddings
    for name, param in model.named_parameters():
        if param.requires_grad:
            is_lora = ...  # Check if it's a LoRA param
            if not is_lora:
                other_state[name] = param.data.cpu()

    torch.save({
        "lora_state_dict": lora_state,
        "other_state_dict": other_state,
        "config": {"rank": ..., "alpha": ...},
    }, path)
```

**Why save only LoRA weights?**

- Full SAM 2 checkpoint: ~150 MB
- LoRA weights only: ~1-2 MB

We don't save the frozen W₀ weights — they're identical to the pretrained model. At inference, we load the pretrained model, apply LoRA structure, then load our small checkpoint.

---

# File 4: `data_utils.py` - Data Containers

This file defines the data structures used for training. We implement these locally because Meta's internal `training.utils.data_utils` is not included in the pip package.

## Classes

```python
@dataclass
class Object:
    """A single object (neuron) in a frame."""
    object_id: int           # Unique identifier
    frame_index: int         # Which frame
    segment: torch.Tensor    # [H, W] boolean mask

@dataclass
class Frame:
    """A single video frame with its objects."""
    data: torch.Tensor       # [C, H, W] normalized image
    objects: List[Object]    # Objects in this frame

@dataclass
class VideoDatapoint:
    """A video clip for training."""
    frames: List[Frame]      # Sequence of frames
    video_id: int           # Which video this came from
    size: Tuple[int, int]   # (H, W)

@dataclass
class BatchedVideoDatapoint:
    """Batched data ready for model input."""
    images: torch.Tensor     # [B, T, C, H, W]
    masks: torch.Tensor      # [B, T, N_obj, H, W]
    obj_ids: List[List[int]] # Object IDs per batch item
    video_ids: List[int]     # Video identifiers

    def to(self, device):
        """Move tensors to device."""
        ...
```

## Function: `collate_fn()`

```python
def collate_fn(batch: List[VideoDatapoint], dict_key: str = None) -> BatchedVideoDatapoint:
    """Collate video datapoints into a batch."""
    # 1. Find dimensions
    B = len(batch)
    T = len(batch[0].frames)
    H, W = batch[0].size

    # 2. Find max objects (for padding)
    max_objs = max(len(vd.frames[0].objects) for vd in batch)

    # 3. Allocate output tensors
    images = torch.zeros(B, T, C, H, W)
    masks = torch.zeros(B, T, max_objs, H, W)

    # 4. Fill tensors from batch items
    for b, vd in enumerate(batch):
        for t, frame in enumerate(vd.frames):
            images[b, t] = frame.data
            for o, obj in enumerate(frame.objects):
                masks[b, t, o] = obj.segment.float()

    return BatchedVideoDatapoint(images=images, masks=masks, ...)
```

**What is a collate function?**

PyTorch's DataLoader calls `collate_fn` to combine multiple dataset items into a batch. We stack individual `VideoDatapoint` objects into a single `BatchedVideoDatapoint` with properly shaped tensors.

---

# File 4.5: `loss_fns.py` - Loss Functions

We implement the loss functions locally because Meta's `training.loss_fns` is not in the pip package.

## Class: `FocalLoss`

```python
class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        self.alpha = alpha  # Class balance factor
        self.gamma = gamma  # Focusing parameter

    def forward(self, pred, target):
        # 1. Sigmoid to get probabilities
        pred_sigmoid = torch.sigmoid(pred)

        # 2. Standard cross-entropy
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # 3. Probability of being correct
        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)

        # 4. Focal weight: down-weight easy examples
        focal_weight = (1 - p_t) ** self.gamma

        # 5. Class balance
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        return (alpha_t * focal_weight * ce_loss).mean()
```

**The focal loss formula:**
```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```

- When `p_t` is high (easy example), `(1-p_t)^γ` is small → low loss
- When `p_t` is low (hard example), `(1-p_t)^γ` is large → high loss

## Class: `DiceLoss`

```python
class DiceLoss(nn.Module):
    """Dice loss for shape optimization."""

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred).view(-1)
        target_flat = target.view(-1)

        intersection = (pred_sigmoid * target_flat).sum()

        dice = (2 * intersection + smooth) / (pred_sigmoid.sum() + target_flat.sum() + smooth)

        return 1 - dice
```

**The Dice formula:**
```
Dice = 2 × |A ∩ B| / (|A| + |B|)
```

## Class: `MultiStepMultiMasksAndIous`

```python
class MultiStepMultiMasksAndIous(nn.Module):
    """Combined loss for SAM 2 training."""

    def forward(self, outputs, targets):
        pred_masks = outputs["pred_masks"]

        # Compute individual losses
        loss_mask = self.focal(pred_masks, targets)    # Pixel accuracy
        loss_dice = self.dice(pred_masks, targets)     # Shape quality
        loss_iou = ...  # IoU prediction accuracy
        loss_class = ...  # Object presence

        # Weighted sum
        core_loss = (
            20.0 * loss_mask +
            1.0 * loss_dice +
            1.0 * loss_iou +
            1.0 * loss_class
        )

        return {"core_loss": core_loss, ...}
```

---

# File 5: `prepare_data.py` - Data Conversion

Converts your TIF files to the format needed for training.

## Function: `prepare_single_video()`

```python
def prepare_single_video(image_tif_path, mask_tif_path, output_dir, video_name, label_id=0):
    # Load TIF stacks
    imgs = imread(str(image_tif_path))   # Shape: (T, H, W)
    masks = imread(str(mask_tif_path))   # Shape: (T, H, W)

    # Handle 2D images (single frame)
    if imgs.ndim == 2:
        imgs = imgs[np.newaxis, ...]     # Add batch dimension
```

**Line 61-67:** Load TIF files. If they're 2D (single image), add a dimension to make them (1, H, W).

---

### Image Processing (Lines 83-101)

```python
for i in range(T):
    # --- Image ---
    frame = imgs[i].copy().astype(np.float32)

    # Per-frame min-max normalization to 0-255
    fmin, fmax = frame.min(), frame.max()
    if fmax > fmin:
        frame = (frame - fmin) / (fmax - fmin) * 255.0
    frame = frame.astype(np.uint8)
```

**Per-frame min-max normalization formula:**
```
pixel_normalized = (pixel - min) / (max - min) × 255
```

**Why per-frame?** Microscopy images have varying brightness due to:
- Photobleaching (fluorescence fades)
- Focus drift
- Illumination fluctuation

Each frame is normalized independently to use full [0, 255] range.

---

### Statistics Accumulation (Lines 92-96)

```python
    # Accumulate stats on normalized [0,1] values
    frame_01 = frame.astype(np.float64) / 255.0
    pixel_sum += frame_01.sum()
    pixel_sq_sum += (frame_01 ** 2).sum()
    pixel_count += frame_01.size
```

**Why accumulate squared sum?**

To compute variance (and thus std) with a single pass:
```
Var(X) = E[X²] - (E[X])²
       = (pixel_sq_sum / count) - (pixel_sum / count)²
```

This is more memory-efficient than storing all pixels.

---

### RGB Conversion and PNG Save (Lines 98-101)

```python
    # Grayscale to RGB (SAM expects RGB)
    frame_rgb = np.stack([frame, frame, frame], axis=-1)

    # Save as PNG (lossless!)
    Image.fromarray(frame_rgb).save(img_dir / f"{i:05d}.png")
```

**Why duplicate to RGB?** SAM 2 expects 3-channel input. We copy grayscale to all 3 channels.

**Why PNG not JPEG?** JPEG is lossy. It removes high-frequency details (what looks like "noise"). But faint dendrites ARE those high-frequency details. PNG preserves everything.

---

### Mask Processing (Lines 103-115)

```python
    # --- Mask ---
    mask = masks[i].copy().astype(np.int32)
    if label_id > 0:
        # Binarize: keep only the specified label
        mask = (mask == label_id).astype(np.uint8) * label_id
    else:
        # Keep all labels
        mask = np.clip(mask, 0, 255).astype(np.uint8)
```

**Two modes:**
- `label_id > 0`: Keep only that label (e.g., only label 4)
- `label_id = 0`: Keep all non-zero labels

---

## Function: `prepare_dataset()` — Aggregate Statistics

```python
# Compute aggregate dataset normalization statistics
total_pixels = sum(r["pixel_count"] for r in results)
dataset_mean = sum(r["pixel_mean"] * r["pixel_count"] for r in results) / total_pixels

# Combined variance formula
dataset_sq_mean = sum(
    (r["pixel_std"] ** 2 + r["pixel_mean"] ** 2) * r["pixel_count"]
    for r in results
) / total_pixels
dataset_std = float(np.sqrt(dataset_sq_mean - dataset_mean ** 2))
```

**The math here is important.** We're combining statistics from multiple videos.

**Weighted mean:**
```
mean_total = Σ(mean_i × count_i) / Σcount_i
```

**Combined variance (this is the key formula):**

For disjoint sets, we can combine variances using:
```
E[X²]_total = Σ((var_i + mean_i²) × count_i) / total_count

var_total = E[X²]_total - mean_total²
```

**Derivation:**
- We know: `Var = E[X²] - E[X]²`, so `E[X²] = Var + E[X]²`
- For each video: `E[X²]_i = var_i + mean_i²`
- Total: weighted average of E[X²] values, then subtract mean²

---

# File 6: `neuron_dataset.py` - PyTorch Dataset

This file defines how training data is loaded and served to the model.

## Imports

```python
from src.training.data_utils import (
    BatchedVideoDatapoint, Frame, Object, VideoDatapoint, collate_fn,
)
```

Note: We import from our local `data_utils.py`, not from Meta's internal modules.

## Class: `NeuronVideoDataset`

### Constructor

```python
def __init__(self, data_dir, video_names, num_frames=8, max_objects=3, transform=None):
    self.data_dir = Path(data_dir)
    self.num_frames = num_frames
    ...

    # Build index of all videos
    self.videos = []
    for name in video_names:
        img_dir = self.data_dir / "images" / name
        frame_files = sorted(img_dir.glob("*.png"))
        n_frames = len(frame_files)

        if n_frames < num_frames:
            continue  # Skip videos that are too short

        self.videos.append({
            "name": name,
            "num_frames": n_frames,
            "img_dir": img_dir,
            "mask_dir": mask_dir,
        })
```

**Building the clip index (Lines 91-95):**

```python
    self.clips = []
    for vid_idx, vid in enumerate(self.videos):
        n_clips = vid["num_frames"] - num_frames + 1
        for start in range(n_clips):
            self.clips.append((vid_idx, start))
```

**Example:**
- Video has 72 frames, we want 8-frame clips
- Number of clips = 72 - 8 + 1 = 65
- Clips: (0,0), (0,1), ..., (0,64) — all starting positions

**What `self.clips` looks like:**
```python
[
    (0, 0),   # Video 0, starting at frame 0
    (0, 1),   # Video 0, starting at frame 1
    ...
    (0, 64),  # Video 0, starting at frame 64
    (1, 0),   # Video 1, starting at frame 0
    ...
]
```

---

### `__getitem__()` — Load One Clip (Lines 117-182)

```python
def __getitem__(self, index: int) -> VideoDatapoint:
    vid_idx, start_frame = self.clips[index]
    vid = self.videos[vid_idx]

    # Load frames and masks for this clip
    frames_list = []
    masks_list = []
    for t in range(self.num_frames):
        frame_idx = start_frame + t
        frame = self._load_frame(vid["img_dir"], frame_idx)
        mask = self._load_mask(vid["mask_dir"], frame_idx)
        frames_list.append(frame)
        masks_list.append(mask)
```

**What happens when you call `dataset[42]`:**
1. Look up `self.clips[42]` → e.g., (video_idx=0, start_frame=42)
2. Load frames 42, 43, 44, 45, 46, 47, 48, 49
3. Load corresponding masks
4. Apply augmentations
5. Return as SAM 2's expected format

---

### Object Detection

```python
    # Determine object IDs present in this clip
    all_labels = set()
    for mask in masks_list:
        labels = torch.unique(mask)
        for l in labels:
            l_val = l.item()
            if l_val > 0:  # Skip background (0)
                all_labels.add(l_val)

    # Limit to max_objects
    obj_ids = sorted(all_labels)[: self.max_objects]

    # If no objects found, create dummy object (empty mask)
    if not obj_ids:
        obj_ids = [1]
```

**Why the dummy object?**

If a clip has no neurons (all-zero masks), we still need to tell the model "there's nothing here." The dummy object with empty masks teaches the model that "no object" is valid output.

This prevents hallucination — without it, the model always predicts "there's a neuron somewhere."

---

### Building SAM 2 Format (Lines 154-182)

```python
    video_frames = []
    for t in range(self.num_frames):
        frame_tensor = frames_list[t]  # [C, H, W]
        mask_tensor = masks_list[t]    # [H, W]

        objects = []
        for obj_id in obj_ids:
            binary_mask = (mask_tensor == obj_id)  # True where neuron is

            objects.append(Object(
                object_id=obj_id,
                frame_index=t,
                segment=binary_mask,
            ))

        video_frames.append(Frame(data=frame_tensor, objects=objects))

    return VideoDatapoint(
        frames=video_frames,
        video_id=vid_idx,
        size=(H, W),
    )
```

**SAM 2's expected format:**
- `VideoDatapoint`: Contains list of `Frame` objects
- Each `Frame`: Contains image tensor + list of `Object`
- Each `Object`: Contains object ID + binary mask for that frame

This is Meta's data structure from their training code.

---

## Function: `neuron_collate_fn()`

```python
def neuron_collate_fn(batch: List[VideoDatapoint]) -> BatchedVideoDatapoint:
    return collate_fn(batch, dict_key="neuron_train")
```

**What is a collate function?**

PyTorch DataLoader loads multiple items and needs to combine them into a batch. The collate function does this.

`collate_fn` is from Meta's SAM 2 code — it stacks VideoDatapoints into a BatchedVideoDatapoint with proper tensor shapes.

---

## Function: `train_val_split()`

```python
def train_val_split(video_names, val_fraction=0.2, seed=42):
    rng = random.Random(seed)
    names = list(video_names)
    rng.shuffle(names)

    n_val = max(1, int(len(names) * val_fraction))
    val_names = names[:n_val]
    train_names = names[n_val:]

    return train_names, val_names
```

**Why fixed seed?**

`seed=42` ensures the same split every time you run. This is critical for reproducibility — you want to compare different training runs on the same data split.

With 10 videos and 20% validation:
- 2 videos for validation
- 8 videos for training

---

# File 7: `augmentations.py` - Data Augmentation

Augmentations artificially increase dataset size and make the model more robust.

## Class: `VideoAugmentation`

### Key Principle: Temporal Consistency

```python
def __call__(self, frames, masks):
    # Sample random parameters ONCE for the whole clip
    do_hflip = random.random() < self.horizontal_flip_p
    do_vflip = random.random() < self.vertical_flip_p
    angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
    ...
```

**CRITICAL:** All frames in a clip get the SAME random parameters.

If frame 3 is flipped, frame 4 must also be flipped. Otherwise the model sees physically impossible motion (neuron jumps to mirror position between frames).

---

### Spatial Transforms (Lines 79-106)

```python
    for frame, mask in zip(frames, masks):
        # Horizontal flip
        if do_hflip:
            frame = TF.hflip(frame)
            mask_3d = TF.hflip(mask_3d)

        # Vertical flip
        if do_vflip:
            frame = TF.vflip(frame)
            mask_3d = TF.vflip(mask_3d)

        # Affine (rotation + shear)
        frame = TF.affine(
            frame,
            angle=angle,
            translate=[0, 0],
            scale=1.0,
            shear=[shear_x, shear_y],
            interpolation=TF.InterpolationMode.BILINEAR,  # Smooth for images
            fill=0,
        )
        mask_3d = TF.affine(
            mask_3d,
            ...
            interpolation=TF.InterpolationMode.NEAREST,  # Exact for masks!
            fill=0,
        )
```

**Interpolation modes:**
- **BILINEAR for images:** Smooth interpolation, avoids jagged edges
- **NEAREST for masks:** Preserves exact integer labels. Bilinear would create fractional values like 2.5, which isn't a valid label.

---

### Resize (Lines 108-119)

```python
        frame = TF.resize(
            frame,
            [self.image_size, self.image_size],  # 1024 × 1024
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )
```

**Why 1024×1024?**

SAM 2 was pretrained with this exact size. The positional encodings are designed for 1024×1024. Using a different size would break the model.

**What is `antialias=True`?**

When downscaling, antialiasing prevents jagged edges by smoothly blending pixels. Important for preserving fine neuron structures.

---

### Intensity Transforms (Lines 121-124)

```python
        frame = TF.adjust_brightness(frame, brightness_factor)
        frame = TF.adjust_contrast(frame, contrast_factor)
        frame = torch.clamp(frame, 0.0, 1.0)
```

**Brightness formula:**
```
pixel_new = pixel_old × factor
```
With factor in [0.85, 1.15], pixels are scaled by ±15%.

**Contrast formula:**
```
pixel_new = mean + (pixel_old - mean) × factor
```
Contrast stretches or compresses values around the mean.

**Why `torch.clamp(frame, 0.0, 1.0)`?**

After brightness/contrast, some pixels might exceed [0, 1]. Clamp forces them back into valid range.

---

### Normalization (Lines 126-127)

```python
        frame = TF.normalize(frame, self.normalize_mean, self.normalize_std)
```

**Normalization formula:**
```
pixel_normalized = (pixel - mean) / std
```

This transforms pixel values to have:
- Mean ≈ 0
- Standard deviation ≈ 1

**Why normalize?**

Neural networks work best when inputs have zero mean and unit variance. Without normalization:
- Gradients can be too large or too small
- Training becomes unstable
- Convergence is slower

---

## Class: `VideoValidationTransform`

```python
class VideoValidationTransform:
    """Validation-only: resize + normalize, no augmentation."""
```

During validation, we DON'T augment:
- We want consistent measurements
- Augmentation adds randomness that would make validation metrics noisy
- We only resize (to 1024×1024) and normalize

---

# File 8: `src/training/train.py` - Training Loop

The main training script. This file includes a custom `SAM2TrainingWrapper` class because Meta's internal `SAM2Train` is not available in the pip package.

## Function: `detect_device()`

```python
def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

**Priority:** CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU

---

## Class: `SAM2TrainingWrapper`

Since Meta's `SAM2Train` class is not in the pip package, we wrap the standard SAM2 model for training:

```python
class SAM2TrainingWrapper(nn.Module):
    """Wrapper that adapts SAM2Base for training with video clips."""

    def __init__(self, sam2_model: nn.Module, num_frames: int = 8):
        super().__init__()
        self.model = sam2_model
        self.num_frames = num_frames

    def forward(self, batch: BatchedVideoDatapoint) -> Dict[str, torch.Tensor]:
        images = batch.images  # [B, T, C, H, W]
        gt_masks = batch.masks  # [B, T, N_obj, H, W]

        # 1. Encode all frames through image encoder
        images_flat = images.view(B * T, C, H, W)
        backbone_out = self.model.forward_image(images_flat)

        # 2. Get features and reshape to video format
        feat = self._extract_features(backbone_out)
        feat = feat.view(B, T, C_feat, H_feat, W_feat)

        # 3. For each object, propagate masks through time
        for obj_idx in range(N_obj):
            for t in range(T):
                if t == 0:
                    # Use GT mask as prompt
                    mask_input = F.interpolate(gt_masks[:, 0, obj_idx], ...)
                else:
                    # Use previous prediction as memory
                    mask_input = F.interpolate(prev_pred, ...)

                pred_mask = self._decode_mask(frame_feat, mask_input)
                ...

        return {"pred_masks": pred_masks}
```

**Why we need this wrapper:**
- Standard `build_sam2()` returns a model for inference
- Training requires a forward pass that takes video batches and returns mask predictions
- The wrapper handles frame-by-frame encoding and mask propagation

---

## Function: `build_training_model()`

```python
def build_training_model(config, device):
    # 1. Build standard SAM2 model (from pip package)
    model = build_sam2(
        config_file=config_name,
        ckpt_path=ckpt_path,
        device="cpu",
        apply_postprocessing=False,
    )

    # 2. Freeze image encoder if configured
    if config.freeze_image_encoder:
        for p in model.image_encoder.parameters():
            p.requires_grad = False

    # 3. Apply LoRA adapters
    apply_lora(model, target_patterns=SAM2_LORA_TARGETS, ...)

    # 4. Wrap for training
    training_model = SAM2TrainingWrapper(model, num_frames=config.data.num_frames)

    # 5. Build loss function (from our local implementation)
    loss_fn = MultiStepMultiMasksAndIous(...)

    return training_model, loss_fn
```

---

### Applying LoRA (Lines 80-91)

```python
    print("\nApplying LoRA adapters...")
    wrapped = apply_lora(
        model,
        target_patterns=SAM2_LORA_TARGETS,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
    )
    print(f"  Wrapped {len(wrapped)} Linear layers with LoRA")
```

This wraps ~50 Linear layers with LoRALinear.

---

### Unfreezing Embeddings (Lines 93-108)

```python
    unfrozen_embeddings = []
    for name, param in model.named_parameters():
        if any(key in name for key in [
            "maskmem_tpos_enc",    # Mask memory temporal position encoding
            "no_mem_embed",        # "No memory" embedding
            "no_mem_pos_enc",      # "No memory" position encoding
            "no_obj_ptr",          # "No object" pointer
            "obj_ptr_proj",        # Object pointer projection
            "obj_ptr_tpos_proj",   # Object pointer temporal projection
        ]):
            param.requires_grad = True
            unfrozen_embeddings.append(name)
```

**Why unfreeze these specific embeddings?**

These are small vectors that control meta-level behavior:
- **Temporal position encodings:** How the model represents "this is frame 5"
- **No-object pointers:** What to output when there's no object
- **Object pointers:** How to maintain object identity across frames

They directly influence tracking quality and are cheap to fine-tune (~thousands of parameters).

---

### Building Loss Function (Lines 121-127)

```python
    loss_fn = MultiStepMultiMasksAndIous(
        weight_dict=config.loss_weight_dict,
        focal_alpha=config.loss.focal_alpha,
        focal_gamma=config.loss.focal_gamma,
        pred_obj_scores=True,
    )
```

**`pred_obj_scores=True`:**

Enables the object presence prediction. The model predicts not just "where is the object" but also "is there an object at all."

This is crucial for preventing hallucination on empty frames.

---

## Function: `build_optimizer()`

```python
def build_optimizer(model: nn.Module, config: TrainingConfig):
    param_groups = get_trainable_parameters(model)  # {"lora": [...], "other": [...]}

    optimizer = torch.optim.AdamW([
        {
            "params": param_groups["lora"],
            "lr": config.optimizer.lora_lr,           # 1e-4
            "weight_decay": config.optimizer.weight_decay,  # 0.1
        },
        {
            "params": param_groups["other"],
            "lr": config.optimizer.base_lr,           # 5e-6
            "weight_decay": 0.0,                      # No decay for embeddings!
        },
    ])
    return optimizer
```

**Two parameter groups with different settings:**

| Group | LR | Weight Decay | Reason |
|-------|-----|--------------|--------|
| LoRA | 1e-4 | 0.1 | Start at ~0, need large steps; decay prevents explosion |
| Embeddings | 5e-6 | 0.0 | Already pretrained, need small steps; no decay to preserve info |

---

## Function: `build_scheduler()`

```python
def build_scheduler(optimizer, config, steps_per_epoch):
    total_steps = config.num_epochs * steps_per_epoch
    warmup_steps = config.optimizer.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)  # Linear warmup

        if config.optimizer.use_cosine_schedule:
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))  # Cosine

        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

**Visual representation:**

```
LR
│
│       Warmup (3 epochs)
│      ╱
│     ╱   Cosine annealing
│    ╱ ╲
│   ╱    ╲
│  ╱       ╲
│ ╱          ╲____
│╱                 ╲
└─────────────────────── epochs
  0   3            57  60
```

**Warmup formula:**
```
lr(step) = lr_base × (step / warmup_steps)
```
Linear ramp from 0 to lr_base.

**Cosine annealing formula:**
```
progress = (step - warmup) / (total - warmup)
lr(step) = lr_base × 0.5 × (1 + cos(π × progress))
```

At progress=0: cos(0) = 1, so lr = lr_base
At progress=0.5: cos(π/2) = 0, so lr = 0.5 × lr_base
At progress=1: cos(π) = -1, so lr ≈ 0

---

## Function: `train()` — The Main Loop

### Mixed Precision Setup (Lines 297-300)

```python
    use_amp = config.use_amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
```

**What is mixed precision (AMP)?**

Use float16/bfloat16 for forward pass (faster, less memory) but float32 for sensitive operations (gradients, loss).

**bfloat16 vs float16:**
- float16: 5-bit exponent, 10-bit mantissa — can underflow
- bfloat16: 8-bit exponent, 7-bit mantissa — same range as float32, safer

---

### Training Loop (Lines 310-356)

```python
    for epoch in range(config.num_epochs):
        model.train()  # Enable dropout, batch norm training mode

        for step, batch in enumerate(train_loader):
            batch = batch.to(device)

            # Forward with mixed precision
            with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_amp):
                outputs = model(batch)
                targets = batch.masks.float()
                losses = loss_fn(outputs, targets)
                loss = losses["core_loss"]

            # Backward
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()    # Scale loss for numerical stability
                scaler.unscale_(optimizer)       # Unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                optimizer.step()

            scheduler.step()
```

**Step by step:**

1. `model.train()` — Enable training mode (dropout active, etc.)
2. `batch = batch.to(device)` — Move data to GPU
3. `with torch.amp.autocast(...)` — Enable mixed precision for forward pass
4. `outputs = model(batch)` — Forward pass through SAM2Train
5. `losses = loss_fn(outputs, targets)` — Compute all loss terms
6. `loss = losses["core_loss"]` — Get combined loss
7. `optimizer.zero_grad()` — Clear old gradients
8. `loss.backward()` — Compute gradients (backpropagation)
9. `clip_grad_norm_(..., 0.1)` — Prevent exploding gradients
10. `optimizer.step()` — Update weights
11. `scheduler.step()` — Update learning rate

---

### Validation and Checkpointing (Lines 372-387)

```python
        if (epoch + 1) % config.val_every_n_epochs == 0:
            val_metrics = validate_epoch(model, val_loader, loss_fn, device)

            if val_metrics["val_iou"] > best_val_iou:
                best_val_iou = val_metrics["val_iou"]
                save_lora_weights(model, checkpoint_dir / "lora_weights.pt")
```

**We save the checkpoint with BEST validation IoU, not the final epoch.**

Why? The final epoch might be overfit. The best validation IoU represents the model that generalizes best to unseen data.

---

# File 9: `evaluate.py` - Metrics

## Function: `compute_iou()`

```python
def compute_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.bool().flatten()
    target = target.bool().flatten()

    intersection = (pred & target).sum().float()
    union = (pred | target).sum().float()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (intersection / union).item()
```

**IoU (Intersection over Union) formula:**

```
IoU = |A ∩ B| / |A ∪ B|
    = TP / (TP + FP + FN)
```

Where:
- TP = True Positives (correctly predicted neuron pixels)
- FP = False Positives (background predicted as neuron)
- FN = False Negatives (neuron predicted as background)

**Range:** 0 (no overlap) to 1 (perfect match)

**Edge case (line 26-27):**
If both masks are empty (no neuron in either), IoU = 1.0 (correct "nothing").

---

## Function: `compute_dice()`

```python
def compute_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.bool().flatten().float()
    target = target.bool().flatten().float()

    intersection = (pred * target).sum()
    denominator = pred.sum() + target.sum()

    if denominator == 0:
        return 1.0 if intersection == 0 else 0.0
    return (2.0 * intersection / denominator).item()
```

**Dice coefficient formula:**

```
Dice = 2 × |A ∩ B| / (|A| + |B|)
     = 2 × TP / (2×TP + FP + FN)
```

**Relationship to IoU:**
```
Dice = 2 × IoU / (1 + IoU)
```

Dice is always ≥ IoU for the same prediction.

**Why both metrics?**
- IoU is stricter, better for model comparison
- Dice is standard in medical imaging literature, makes results comparable to published work

---

## Function: `evaluate_batch()`

```python
def evaluate_batch(outputs, targets):
    all_ious = []
    all_dices = []

    for t in range(num_frames):
        pred_masks = outputs[t]["pred_masks_high_res"]  # [B, 1, H, W]
        pred_binary = (pred_masks > 0).squeeze(1)       # Threshold at 0

        gt = targets[t]

        # Resize GT if shapes don't match
        if pred_binary.shape[-2:] != gt.shape[-2:]:
            gt = interpolate(gt, size=pred_binary.shape[-2:], mode="nearest")

        for b in range(batch_size):
            iou = compute_iou(pred_binary[b], gt[b])
            dice = compute_dice(pred_binary[b], gt[b])
            all_ious.append(iou)
            all_dices.append(dice)

    return {"mean_iou": np.mean(all_ious), "mean_dice": np.mean(all_dices)}
```

**Key line:**
```python
pred_binary = (pred_masks > 0).squeeze(1)
```

SAM 2 outputs logits (can be any value). We threshold at 0:
- logit > 0 → predict neuron (probability > 50%)
- logit ≤ 0 → predict background (probability ≤ 50%)

---

## Function: `validate_epoch()`

```python
@torch.no_grad()  # Disable gradient computation (faster, less memory)
def validate_epoch(model, val_loader, loss_fn, device):
    model.eval()  # Disable dropout, use running stats for batch norm

    for batch in val_loader:
        outputs = model(batch)
        losses = loss_fn(outputs, batch.masks)
        metrics = evaluate_batch(outputs, batch.masks)
        ...

    model.train()  # Re-enable training mode
    return {"val_loss": ..., "val_iou": ..., "val_dice": ...}
```

**`@torch.no_grad()`:**

During validation, we don't need gradients. This:
- Reduces memory usage (no gradient storage)
- Speeds up computation (no gradient tracking)

**`model.eval()` vs `model.train()`:**

| Aspect | train() | eval() |
|--------|---------|--------|
| Dropout | Active (random zeros) | Disabled (all neurons used) |
| Batch Norm | Update running stats | Use frozen running stats |

---

# Summary: What You Should Know

After reading this document, you should be able to explain:

1. **Why LoRA?** — Parameter-efficient fine-tuning that avoids overfitting with small datasets
2. **LoRA math:** `y = W₀·x + (α/r)·B·A·x`, B initialized to zeros
3. **Why those specific layers?** — Mask decoder (what to segment) + memory attention (temporal reasoning)
4. **Why freeze image encoder?** — General features work, would overfit with 10 videos
5. **Data pipeline:** TIF → PNG (lossless), per-frame normalization, dataset stats
6. **Augmentations:** Temporal consistency (same transform for all frames in clip)
7. **Loss function:** Focal (handles class imbalance) + Dice (shape quality) + IoU pred + Object score
8. **Two learning rates:** LoRA (1e-4) vs embeddings (5e-6)
9. **Cosine schedule:** Warmup then smooth decay
10. **Evaluation:** IoU and Dice metrics, save best validation checkpoint

Good luck with your paper and presentation!
