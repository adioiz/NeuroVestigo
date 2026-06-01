# Complete Code Walkthrough: SAM 2 LoRA Fine-Tuning

This document explains **every file, every function, and every important line** with the mathematical and logical background. You should be able to read this and explain every aspect to your professor.

---

# Table of Contents

1. [cli.py (Entry Point)](#file-1-trainpy---entry-point)
2. [config.py (Configuration)](#file-2-configpy---configuration)
3. [lora.py (LoRA Implementation)](#file-3-lorapy---lora-implementation)
4. [data_utils.py (Data Containers)](#file-4-data_utilspy---data-containers)
5. [loss_fns.py (Loss Functions)](#file-45-loss_fnspy---loss-functions)
6. [prepare_data.py (Data Conversion)](#file-5-prepare_datapy---data-conversion)
7. [neuron_dataset.py (PyTorch Dataset)](#file-6-neuron_datasetpy---pytorch-dataset)
8. [augmentations.py (Data Augmentation)](#file-7-augmentationspy---data-augmentation)
9. [trainer.py (Training Loop)](#file-8-srctrainerpy---training-loop)
10. [evaluate.py (Metrics)](#file-9-evaluatepy---metrics)

---

# File 1: `cli.py` - Entry Point

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
    from src.prepare_data import prepare_dataset

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
- `from src.prepare_data import prepare_dataset` — Imports the conversion function (lazy import, only when needed)
- `with open(args.manifest) as f:` — Opens your data_manifest.json file
- `manifest = json.load(f)` — Parses JSON into a Python list of dicts
- `prepare_dataset(...)` — Calls the conversion function

---

## Function: `cmd_train(args)`

```python
def cmd_train(args):
    """Run the training loop.

    CLI flags override config.py defaults *only when explicitly provided*.
    Unspecified flags fall through to whatever is in src/config.py, so
    config.py is the single source of truth.
    """
    from src.config import TrainingConfig
    from src.trainer import train

    config = TrainingConfig()  # Start from config.py defaults

    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    # ... and so on for every override ...

    train(config)
```

**Purpose:** Builds the configuration object from `config.py` defaults and applies CLI overrides only where the user explicitly passed them. Then prints a "resolved config" banner so the SLURM log makes the actual values obvious.

**The previous version had a bug:** it built `TrainingConfig(...)` from CLI defaults that hadn't been kept in sync with `config.py`, which meant edits to `config.py` silently never took effect on the cluster. Now the CLI defaults are all `None` and `config.py` is the single source of truth.

**Special-cased flags:**
- `--use-imagenet-stats` → forces ImageNet normalization (otherwise `stats.json` is auto-loaded).
- `--no-precompute`, `--no-cache`, `--no-cbdice`, `--no-supervoxel`, `--no-topology` — `store_true` flags that flip booleans only when present.
- `--val-videos A,B,C` — comma-separated list of video names; overrides `val_fraction`.
- `--compile-encoder`, `--compile-decoder` — enable `torch.compile` on the encoder / decoder respectively.

---

## Function: `main()` — Argument Parsing

```python
parser = argparse.ArgumentParser(...)
subparsers = parser.add_subparsers(dest="command", required=True)
```

**Concept:** Two subcommands: `prepare` and `train`. Like `git commit` vs `git push` — same program, different actions.

**Key change from earlier versions:** all numeric / string CLI defaults are now `None`. This is intentional — it means an unspecified flag falls through to `config.py`. Booleans default to `False` (as `store_true` flags) and only flip when present.

```python
p_train.add_argument("--epochs", type=int, default=None)
p_train.add_argument("--batch-size", type=int, default=None)
p_train.add_argument("--lora-rank", type=int, default=None)
p_train.add_argument("--lora-alpha", type=float, default=None)
p_train.add_argument("--lora-lr", type=float, default=None)
p_train.add_argument("--lora-b-lr-ratio", type=float, default=None)
p_train.add_argument("--base-lr", type=float, default=None)
p_train.add_argument("--warmup-epochs", type=int, default=None)
# ... etc ...

p_train.add_argument("--val-videos", type=str, default=None,
    help="Comma-separated video names to use as the val split")
p_train.add_argument("--compile-encoder", action="store_true")
p_train.add_argument("--compile-decoder", action="store_true")
```

**Three learning rates (LoRA+) come from `config.py`:**
- `lora_lr = 1e-4` — for LoRA A matrices (down-projection, Kaiming init)
- `lora_b_lr_ratio = 16.0` — B matrices get `lora_lr × 16 = 1.6e-3` (B starts at zeros, needs higher LR)
- `base_lr = 1e-5` — for embeddings (lower, they're already pretrained)

---

# File 2: `config.py` - Configuration

All hyperparameters organized as Python dataclasses.

## LoRAConfig

```python
@dataclass
class LoRAConfig:
    rank: int = 16              # Main LoRA rank — decoder + memory_attention
    alpha: float = 32.0          # α scaling factor (kept α/r = 2.0)
    dropout: float = 0.10        # Regularization

    # Run 6 addition — LoRA on Hiera encoder's last stage
    hiera_last_stage: bool = True
    hiera_rank: int = 8          # Smaller rank for the 768-dim Hiera layers
    hiera_alpha: float = 16.0    # Scale = α/r = 2.0 (consistent)
```

**Mathematical background:**

The LoRA forward pass is:
```
y = W₀·x + (α/r)·B·A·x
```

- `rank = 16`: Matrix A has shape (16 × input_dim), B has shape (output_dim × 16)
- `alpha = 32.0`: Scaling factor
- `scale = alpha/rank = 32/16 = 2.0`: How much LoRA influences output
- `dropout = 0.10`: 10% of inputs are randomly zeroed during training. Tuned: 0.25 over-regularized, 0.05 underfit, 0.10 is the sweet spot.

**Why rank 16 for main LoRA (was 8 through Run 4):** With the dataset grown to 16 videos, the overfit risk from a bigger adapter is lower, and Runs 1–4 saw the model behaving very similarly to the base SAM 2 — suggesting more adapter capacity could help. Trainable parameter count roughly doubles from rank 8.

**Why Hiera last-stage LoRA in Run 6:** Run 5 (Path B + rank 16) reached val IoU 0.8295 with train_loss=0.395 > val_loss=0.282 — a wide reversed gap that signals capacity headroom, not overfit. Per-video performance was uniform (0.822/0.844/0.823), suggesting a representational ceiling rather than isolated hard cases. The frozen Hiera encoder was pretrained on natural images; its deepest features had never seen phase-contrast microscopy. Adapting the last 2 trunk blocks gives the model permission to adjust its visual representation to microscopy without fully unfreezing the encoder.

**Why rank 8 (smaller than main) for Hiera:** Hiera's last-stage layers are 768-dim (vs 256 in the decoder/memory_attention). The same rank gives more absolute capacity per layer, and encoder edits are more impactful — so a smaller LoRA is the safer add given only 16 training videos. Coverage is exactly 8 layers: `attn.qkv`, `attn.proj`, `mlp.layers.0`, `mlp.layers.1` per block × 2 blocks.

---

## DataConfig

```python
@dataclass
class DataConfig:
    training_data_dir: str = "training_data"
    num_frames: int = 8        # Frames per clip
    image_size: int = 1024     # SAM 2 input size (fixed!)
    max_objects: int = 3       # Max neurons tracked simultaneously
    val_fraction: float = 0.25 # 25% for validation (5 of 20 videos at current dataset size)
    val_videos: Optional[List[str]] = None  # Explicit override
    clip_stride: int = 4       # Skip overlapping clips (1=all clips, 4=every 4th)
    cache_in_memory: bool = True       # Cache all frames/masks in RAM at init
    precompute_features: bool = False  # Off — full pixel-level augmentation enabled
```

**Why `val_fraction = 0.25` (was 0.2)?**
With only 12 videos, the previous 80/20 split gave 2 val videos. Mean of 2 is noisy — a single video's IoU dominates the headline number. Bumping to 0.25 gives 9 train / 3 val and a more stable mean of 3.

**Why `val_videos` (override field)?**
Lets you pin a specific val split for split-rotation experiments. When None (default), the seed-based shuffle picks 3 videos. When set (e.g. `["A11_C1", "A52_C1", "A91_C1_1"]`), those exact 3 go to val and the rest train. CLI flag: `--val-videos A11_C1,A52_C1,A91_C1_1`.

**Why `precompute_features = False` (was True)?**
With precompute on, all training augmentation is replaced by feature-level h-flip / v-flip — rotation, shear, brightness, contrast all silently drop out. On a small dataset that effectively gives the model only ~4 distinct views per frame. Precompute is faster per step but the augmentation loss made the model overfit. Now off by default; turn back on with no other changes if you want the speedup at the cost of augmentation diversity.

**Why `num_frames = 8`?**

Your 72-frame video is split into 8-frame "clips" with stride:
```
Frames:  [0][1][2][3][4][5][6][7][8]...[71]
Clip 1 (stride=4): [0][1][2][3][4][5][6][7]
Clip 2 (stride=4): [4][5][6][7][8][9][10][11]
...
```

Number of clips with stride=4 = (72 - 8 + 1) / 4 ≈ 17 clips per video (vs 65 with stride=1).

**Why 8 and not more?**
- 8 frames × 1024×1024 × 3 channels = ~100 MB per clip
- Model activations add ~4-8 GB
- Batch size 1 with 8 frames fits in ~10 GB VRAM

**Why `image_size = 1024`?**
SAM 2 was pretrained at 1024×1024. Using a different size would break the positional encodings.

**Why `clip_stride = 4`?**
With stride=1, consecutive clips overlap by 7/8 frames (87.5% overlap). This wastes compute on nearly identical data. Stride=4 reduces steps/epoch by 4x with minimal information loss.

**Why `precompute_features = True`?**
The frozen image encoder output is deterministic — recomputing it every step wastes ~60-70% of per-step compute. Precomputing once gives a **10x speedup**.

**Why `cache_in_memory = True`?**
All 876 frames + masks fit in ~2.4 GB RAM. Caching eliminates disk I/O from NFS, which can add 1-5s/step on cluster storage.

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
    base_lr: float = 1e-5      # For embeddings (pretrained)
    lora_lr: float = 1e-4      # For LoRA A matrices (down-projection)
    lora_b_lr_ratio: float = 16.0  # LoRA+ ratio: lr_B = lora_lr × ratio
    weight_decay: float = 0.05  # Decoupled weight decay (AdamW)
    max_grad_norm: float = 1.0
    warmup_epochs: int = 2     # Linear warmup duration
    use_cosine_schedule: bool = True
    min_lr_ratio: float = 0.05  # LR floor at 5% of peak
```

**Why `warmup_epochs = 2` (was 1)?**
LoRA-B is zero-initialized; the first gradient steps push it out of a sharp init region and benefit from a gentler LR ramp. 2 epochs ≈ 3.3% of a 60-epoch run.

**Why `min_lr_ratio = 0.05` (was 0.10)?**
GPT-3 used 10% as the floor; in earlier runs the cosine bottomed out ~10 epochs before training ended and the flat 10% tail produced no improvements. Halving the floor lets the late phase run at finer LR steps.

**Why `weight_decay = 0.05` (was 0.1)?**
Tuned alongside dropout — when we lowered dropout from 0.25 to 0.10, weight_decay 0.2 was over-regularising. 0.05 paired with dropout 0.10 gives the right total regularization budget for our dataset size.

**Why three learning rates (LoRA+)?**

LoRA+ (Hayou et al., ICML 2024) uses asymmetric learning rates for A and B matrices:

- **A matrices (lr=1e-4):** Down-projection, initialized with Kaiming uniform. Produces reasonable gradients from the start.
- **B matrices (lr=1.6e-3, 16× higher):** Up-projection, initialized to zeros. Because B starts at zero, gradients for B are much smaller than for A — they need a higher learning rate to catch up. This is the key insight of LoRA+.
- **Embeddings (lr=1e-5):** Already pretrained with useful values. Large steps would destroy them.

**Why `max_grad_norm = 1.0`?**

Gradient clipping. If gradients get too large (exploding gradients), we scale them down:
```
if ||gradients|| > 1.0:
    gradients = gradients × (1.0 / ||gradients||)
```

This prevents training instability.

---

## LossConfig

```python
@dataclass
class LossConfig:
    loss_mask_weight: float = 5.0       # Focal loss weight
    loss_dice_weight: float = 2.0       # Dice loss weight
    loss_cbdice_weight: float = 5.0     # cbDice for skeleton topology (bumped from 3.0 in Run 5)
    loss_supervoxel_weight: float = 0.3 # Supervoxel weight — small, re-enabled in Run 5
    loss_iou_weight: float = 1.0        # IoU prediction loss
    loss_class_weight: float = 1.0      # Object presence loss
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    use_cbdice: bool = True
    use_supervoxel: bool = True         # Re-enabled in Run 5 — see note below
    topology_resolution: int = 256
```

**Why `loss_cbdice_weight = 5.0` (was 3.0 in Run 4, 1.0 originally)?**
Neurons are tree-like — topology fidelity matters more than bulk pixel overlap. Diagnostic per-component logging showed the cbDice value drop from 0.056 (weight 1.0) to 0.016 (weight 3.0) to ~0.008 (weight 3.0, more training). Run 5 bumps to 5.0 because Run 4 predictions still showed broken thin branches in user inspection. The user-reported "lack of connectivity" issue is what this weight targets.

**Why `use_supervoxel = True` again?**
Was disabled in Run 3 after sitting flat at 0.30 for many epochs. Re-enabled in Run 5 with a smaller weight (0.3 instead of 1.0) for two reasons:
1. The dataset grew from 12 to 16 videos — the previous behaviour may have been an undersampling artefact.
2. Path B (real `memory_attention`) means the forward pass now produces temporally coherent predictions; the connectivity signal supervoxel measures might finally be learnable.

Diagnostics will show within ~5 epochs whether the per-component value drops below 0.20. If it stays flat at 0.30, disable again.

**Why `topology_resolution = 256`?**

Topology losses (cbDice skeleton/EDT and supervoxel connectivity analysis) are computed at this resolution instead of the full 1024×1024. This is a 16× pixel reduction that gives a ~2000× speedup for topology losses with minimal quality impact — topology (connectivity, skeleton structure) is a coarse property that doesn't require full resolution. The supervoxel loss detects critical components at 256×256 and then upscales the critical masks back to 1024×1024 for accurate loss gradients.

**Combined loss:**
```
L_total = 5·L_focal + 2·L_dice + 1·L_cbDice + 1·L_supervoxel + 1·L_iou + 1·L_class
```

**Why these weights?**
- **Focal (5.0)**: Pixel accuracy is the primary objective, but reduced from 20 to balance with topology losses
- **Dice (2.0)**: Region-level overlap is important for shape matching
- **cbDice + Supervoxel (1.0 each)**: Topology preservation for thin neuron branches

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

# Run 6 addition — Hiera encoder's last stage (deepest visual features)
HIERA_LAST_STAGE_LORA_TARGETS = [
    r"image_encoder\.trunk\.blocks\.(10|11)\.attn\.(qkv|proj)",
    r"image_encoder\.trunk\.blocks\.(10|11)\.mlp\.layers\.\d+",
]
```

The Hiera target list matches the last 2 of the trunk's 12 blocks (stages `[1, 2, 7, 2]` → indices 10 and 11 are the last stage). Each `MultiScaleBlock` has `attn.qkv` (fused QKV projection), `attn.proj` (output projection), and `mlp.layers.{0,1}` (two MLP linears).

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

## Functions: `save_train_state()` and `load_train_state()`

`save_lora_weights` saves only the bytes you need for **inference**. For **resuming a training run** under a SLURM wall-clock limit, the model weights alone aren't enough — you also need the optimiser moments, the LR scheduler position, and the early-stopping counter. That's what `save_train_state` writes:

```python
def save_train_state(path, *, model, optimizer, scheduler, epoch,
                     best_val_iou, epochs_without_improvement,
                     num_epochs, steps_per_epoch) -> None:
    torch.save({
        # Model side — same payload as save_lora_weights
        "lora_state_dict":    {...},  # all LoRA A/B params
        "other_state_dict":   {...},  # unfrozen embeddings
        "config":             {"rank": ..., "alpha": ...},
        # Training-continuation side
        "optimizer_state_dict":         optimizer.state_dict(),
        "scheduler_state_dict":         scheduler.state_dict(),
        "epoch":                        epoch,
        "best_val_iou":                 best_val_iou,
        "epochs_without_improvement":   epochs_without_improvement,
        "num_epochs":                   num_epochs,
        "steps_per_epoch":              steps_per_epoch,
    }, path)
```

Written at the end of every epoch to `<checkpoint_dir>/train_state.pt`.

`load_train_state(path, *, model, optimizer, scheduler)` reverses it: restores model params, optimiser, scheduler, and returns the saved counters so the train loop can pick up at `epoch + 1`.

**Workflow:** an sbatch script auto-detects the file and adds `--resume` accordingly:

```bash
STATE_FILE=checkpoints/train_state.pt
if [ -f "$STATE_FILE" ]; then
    python -u cli.py train --epochs 90 --resume "$STATE_FILE"
else
    python -u cli.py train --epochs 90
fi
```

Submitting the script repeatedly carries one logical training run across multiple SLURM jobs. To start a NEW training run later, delete `train_state.pt` manually — otherwise the smart sbatch silently picks up the old one.

**Size.** Heavier than `lora_weights.pt` because of the Adam moments (~5M trainable params × 2 buffers × 4 bytes ≈ 40 MB), but still trivial vs the dataset.

**Mid-epoch interrupt cost.** Save happens at end-of-epoch only, so a SIGTERM mid-epoch costs about one epoch's worth of GPU time (~3.5 min at current step times). Acceptable.

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

## Class: `SoftCBDiceLoss` (Topology-Preserving with Boundary Weighting)

cbDice (Shi et al., MICCAI 2024) extends clDice (Shit et al., CVPR 2021) by adding distance-transform-based boundary weighting. The key improvement: thinner structures get higher weight via inverse-radius weighting from the Euclidean distance transform (EDT).

**Formula:**
```
cbTprec = |W_t · skel(pred) · target| / |W_t · skel(pred)|   (weighted skeleton precision)
cbTsens = |W_p · skel(target) · pred| / |W_p · skel(target)| (weighted skeleton sensitivity)
cbDice = 2 × cbTprec × cbTsens / (cbTprec + cbTsens)
```

Where W_t and W_p are inverse-radius distance transform weights. For a structure with local radius r, the weight is 1/(r+1) — thin dendrites (r≈1) get weight ≈0.5, thick soma (r≈10) gets weight ≈0.09.

**Why this matters for neurons:**
- Neuron branches are thin tubular structures with varying radius
- Their skeleton represents the centerline
- Breaking a thin branch gets heavily penalized (both skeleton loss AND high weight from small radius)
- cbDice automatically upweights errors on thin branches that standard clDice would treat equally

**Performance optimization:** cbDice downsamples masks to `topology_resolution` (default 256×256) before computing skeletons and EDT. This is a 16× pixel reduction that gives ~100× speedup for the EDT computation and ~16× for soft_skel, with minimal quality loss since topology is a coarse property. Gradients flow through the bilinear interpolation used for downsampling.

---

## Class: `SupervoxelLoss2D` (Connectivity-Preserving)

**NEW!** Supervoxel loss specifically targets split/merge errors that cbDice may miss.

```python
class SupervoxelLoss2D(nn.Module):
    """Connectivity-preserving loss that penalizes topology changes."""

    def __init__(self, alpha=0.5, beta=0.5, threshold=0.5, topology_resolution=256):
        self.alpha = alpha    # Balance: topology vs standard loss
        self.beta = beta      # Balance: split vs merge errors
        self.topology_resolution = topology_resolution  # Downsample for speed

    def forward(self, pred, target):
        # 1. Compute base BCE loss at FULL resolution (for accurate gradients)
        base_loss = BCE(pred, target)

        # 2. Downsample to topology_resolution for connectivity detection
        pred_binary_ds = downsample(sigmoid(pred) > threshold, topology_resolution)
        target_binary_ds = downsample(target, topology_resolution)

        # 3. Detect CRITICAL components using scipy.ndimage.label (C-compiled)
        neg_critical, pos_critical = detect_critical_components(pred_binary_ds, target_binary_ds)

        # 4. Upscale critical masks back to full resolution
        neg_critical = upsample(neg_critical, original_size)
        pos_critical = upsample(pos_critical, original_size)

        # 5. Apply higher loss to critical regions at full resolution
        loss = (1-α)·base_loss + α·β·loss(neg_critical) + α·(1-β)·loss(pos_critical)
```

**Mathematical formulation:**
```
L = (1−α)L₀(y,ŷ) + αβ∑L₀(yc,ŷc) + α(1−β)∑L₀(yc,ŷc)
```

Where:
- L₀ is the base loss (BCE)
- α controls topology-aware vs standard loss balance (default: 0.5)
- β balances negatively vs positively critical components (default: 0.5)
- yc are pixels in critical supervoxels

**Two types of critical components:**

1. **Negatively Critical (False Negatives):**
   - Missing pixels that cause **split errors**
   - If added to prediction, would connect previously disconnected regions
   - Example: A 1-pixel gap in a neuron branch

2. **Positively Critical (False Positives):**
   - Extra pixels that cause **merge errors**
   - If removed, would split incorrectly merged regions
   - Example: A false bridge connecting two separate neurons

**Algorithm (O(n) complexity via scipy.ndimage.label):**
```
1. Downsample masks to 256×256 (16× fewer pixels)
2. Label connected components using scipy.ndimage.label (C-compiled, ~100× faster than Python BFS)
3. For each FN component: use binary_dilation to find adjacent prediction labels
4. For each FP component: remove, relabel, check if neighbors split
5. Upscale critical masks back to full resolution for loss application
```

**Performance:** The original implementation used pure Python BFS with nested pixel loops over 1024×1024 (1M pixels), which took ~230s/step. Replacing with scipy.ndimage.label + downsampling to 256×256 reduces this to <0.1s/step (~2000× speedup).

**Why this complements cbDice:**
- cbDice: "Does the skeleton match?" (weighted centerline overlap)
- Supervoxel: "Does connectivity match?" (topology preservation)

A 1-pixel gap may not significantly affect cbDice (skeleton mostly intact), but Supervoxel loss heavily penalizes it because it causes a split.

**Reference:**
Grim et al., "Efficient Connectivity-Preserving Instance Segmentation with Supervoxel-Based Loss Function", AAAI 2025

---

## Class: `MultiStepMultiMasksAndIous`

```python
class MultiStepMultiMasksAndIous(nn.Module):
    """Combined loss for SAM 2 training."""

    def forward(self, outputs, targets):
        pred_masks = outputs["pred_masks"]

        # Compute individual losses
        loss_mask = self.focal(pred_masks, targets)        # Pixel accuracy
        loss_dice = self.dice(pred_masks, targets)         # Shape quality
        loss_cbdice = self.cbdice(pred_masks, targets)     # Skeleton topology (with boundary weighting)
        loss_supervoxel = self.supervoxel(pred_masks, targets)  # Connectivity
        loss_iou = ...  # IoU prediction accuracy
        loss_class = ...  # Object presence

        # Weighted sum
        core_loss = (
            5.0 * loss_mask +        # Focal loss
            2.0 * loss_dice +        # Dice loss
            1.0 * loss_cbdice +      # cbDice for skeleton
            1.0 * loss_supervoxel +  # Supervoxel for connectivity
            1.0 * loss_iou +
            1.0 * loss_class
        )

        return {"core_loss": core_loss, ...}
```

**Combined loss formula:**
```
L_total = 5×L_focal + 2×L_dice + 1×L_cbDice + 1×L_supervoxel + 1×L_iou + 1×L_class
```

**Fast training mode (CPU/MPS):**
```bash
# Disable both topology losses for faster training
python cli.py train --no-topology
```
This sets both `use_cbdice=False` and `use_supervoxel=False`, reducing the loss to:
```
L_total = 5×L_focal + 2×L_dice + 1×L_iou + 1×L_class
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
from src.data_utils import (
    BatchedVideoDatapoint, Frame, Object, VideoDatapoint, collate_fn,
)
```

Note: We import from our local `data_utils.py`, not from Meta's internal modules.

## Class: `NeuronVideoDataset`

### Constructor

```python
def __init__(self, data_dir, video_names, num_frames=8, max_objects=3,
             transform=None, clip_stride=1, cache_in_memory=False):
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

    # Optionally cache all frames/masks in RAM (eliminates disk I/O)
    if cache_in_memory:
        for vid in self.videos:
            for idx in range(vid["num_frames"]):
                self._frame_cache[(vid["name"], idx)] = self._load_frame_from_disk(...)
                self._mask_cache[(vid["name"], idx)] = self._load_mask_from_disk(...)
```

**Building the clip index with stride:**

```python
    self.clips = []
    for vid_idx, vid in enumerate(self.videos):
        n_clips = vid["num_frames"] - num_frames + 1
        for start in range(0, n_clips, clip_stride):  # stride=4 skips overlapping clips
            self.clips.append((vid_idx, start))
```

**Example with stride=4:**
- Video has 72 frames, we want 8-frame clips
- Clip starts: 0, 4, 8, 12, ..., 64 → 17 clips (vs 65 with stride=1)
- **4x fewer steps per epoch** with minimal information loss

**What `self.clips` looks like (stride=4):**
```python
[
    (0, 0),   # Video 0, starting at frame 0
    (0, 4),   # Video 0, starting at frame 4
    (0, 8),   # Video 0, starting at frame 8
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
def neuron_collate_fn(batch):
    """Each item in batch is a tuple of (VideoDatapoint, frame_ids)."""
    video_datapoints = [item[0] for item in batch]
    frame_id_lists = [item[1] for item in batch]
    batched = collate_fn(video_datapoints, dict_key="neuron_train")
    return batched, frame_id_lists
```

**What is a collate function?**

PyTorch DataLoader loads multiple items and needs to combine them into a batch. The collate function does this.

The collate function now returns a tuple `(BatchedVideoDatapoint, frame_id_lists)` where `frame_id_lists` is a list of lists of `(video_name, frame_idx)` tuples. These frame IDs enable the training loop to look up precomputed features for each frame in the batch.

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

# File 8: `src/trainer.py` - Training Loop

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

Since Meta's `SAM2Train` class is not in the pip package, we wrap the standard SAM 2 model for training. **As of Run 5**, the wrapper dispatches to one of two forward paths:

```python
class SAM2TrainingWrapper(nn.Module):
    """Wrapper that adapts SAM2Base for training with video clips.

    Has two forward paths:
    - _forward_memory (default): uses SAM2Base.track_step → memory_attention
      conditioning + SAM heads + memory_encoder. All 68 LoRA adapters
      receive gradients including the 40 in memory_attention.
    - _forward_simple (legacy): feeds the previous predicted mask as a
      dense prompt to the SAM decoder each step. Faster but the
      memory_attention LoRA adapters never receive gradients.
    """

    def __init__(self, sam2_model, num_frames=8, use_memory_path=True):
        super().__init__()
        self.model = sam2_model
        self.num_frames = num_frames
        self.use_memory_path = use_memory_path
        if use_memory_path:
            # Force the SAM decoder to be exercised on the conditioning frame:
            # by default SAM2 has use_mask_input_as_output_without_sam=True
            # which routes the GT mask straight to output on frame 0.
            self.model.use_mask_input_as_output_without_sam = False

    def forward(self, batch, precomputed_backbone_out=None):
        if self.use_memory_path:
            return self._forward_memory(batch, precomputed_backbone_out)
        return self._forward_simple(batch, precomputed_backbone_out)
```

### Path B forward (`_forward_memory`, default since Run 5)

For each frame `t = 0..T-1`:
1. **Slice features** for frame `t` out of the B*T-batched encoder output.
2. **Expand to (clip × object)** via `repeat_interleave`. Each `(clip, object)` entry gets its own memory bank slot.
3. **Call `model.track_step(...)`** with `is_init_cond_frame=(t==0)`. On frame 0 the GT mask is the dense prompt; on later frames `mask_inputs=None` and memory_attention provides the temporal context from the bank.
4. **Store the live output dict** under `cond_frame_outputs` (t=0) or `non_cond_frame_outputs` (t>0). Gradients must flow through the bank, so we do NOT detach.

The result `out_t["pred_masks_high_res"]` has shape `[B*N_obj, 1, H, W]`. We stack T frames and reshape to `[B, T, N_obj, H, W]`.

### Path A forward (`_forward_simple`, legacy)

Retained for A/B comparison. Activated by `use_memory_attention_path=False` or `--no-memory-attention`. Loops over objects then frames; on frame 0 uses the GT mask as a dense prompt, on later frames interpolates the previously predicted mask as the prompt. This is what Runs 1–4 used.

**Why both paths exist:** the simplified path is faster (no memory bank, no temporal cross-attention) and was easier to get correct first. It also gives a clean baseline to measure the value-add of memory_attention on this dataset.

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

    # 3. Apply main LoRA — mask decoder + memory_attention (rank 16)
    apply_lora(model, target_patterns=SAM2_LORA_TARGETS, ...)

    # 3b. (Run 6) Apply second LoRA pass on the last 2 Hiera trunk
    #     blocks. The encoder base weights stay frozen; LoRALinear's A/B
    #     are NEW parameters that default to requires_grad=True, so the
    #     adapters train while the base Hiera weights stay frozen.
    if config.lora.hiera_last_stage:
        apply_lora(
            model,
            target_patterns=HIERA_LAST_STAGE_LORA_TARGETS,
            rank=config.lora.hiera_rank,  # 8 — smaller than main LoRA
            alpha=config.lora.hiera_alpha,
        )

    # 4. Wrap for training (Path B by default — uses SAM 2's full
    #    track_step API so memory_attention LoRA adapters actually train)
    training_model = SAM2TrainingWrapper(
        model,
        num_frames=config.data.num_frames,
        use_memory_path=config.use_memory_attention_path,
    )

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

## Function: `build_optimizer()` — LoRA+ Asymmetric Learning Rates

```python
def build_optimizer(model: nn.Module, config: TrainingConfig):
    param_groups = get_trainable_parameters(model)
    # Returns: {"lora_a": [...], "lora_b": [...], "other": [...]}

    lora_b_lr = config.optimizer.lora_lr * config.optimizer.lora_b_lr_ratio

    optimizer = torch.optim.AdamW([
        {
            "params": param_groups["lora_a"],
            "lr": config.optimizer.lora_lr,           # 1e-4
            "weight_decay": config.optimizer.weight_decay,  # 0.1
        },
        {
            "params": param_groups["lora_b"],
            "lr": lora_b_lr,                          # 1.6e-3 (16× higher)
            "weight_decay": config.optimizer.weight_decay,  # 0.1
        },
        {
            "params": param_groups["other"],
            "lr": config.optimizer.base_lr,           # 1e-5
            "weight_decay": 0.0,                      # No decay for embeddings!
        },
    ])
    return optimizer
```

**Three parameter groups with LoRA+ (Hayou et al., ICML 2024):**

| Group | LR | Weight Decay | Reason |
|-------|-----|--------------|--------|
| LoRA A | 1e-4 | 0.1 | Down-projection, Kaiming init, reasonable gradients |
| LoRA B | 1.6e-3 | 0.1 | Up-projection, zeros init, needs 16× higher LR (LoRA+) |
| Embeddings | 1e-5 | 0.0 | Already pretrained, need small steps; no decay to preserve info |

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
│  Warmup (1 epoch)
│ ╱
│╱   Cosine annealing
│ ╲
│   ╲
│     ╲
│       ╲____
│             ╲
└─────────────────────── epochs
  0 1            55  60
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

### Training Loop

```python
    for epoch in range(config.num_epochs):
        model.train()

        for step, (batch, frame_id_lists) in enumerate(train_loader):
            batch = batch.to(device)

            with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_amp):
                if feature_cache is not None:
                    # Feature-level flip augmentation
                    do_hflip = random.random() < config.augmentation.horizontal_flip_p
                    do_vflip = random.random() < config.augmentation.vertical_flip_p
                    backbone_out = assemble_cached_backbone(
                        feature_cache, pos_enc, frame_id_lists, device,
                        do_hflip=do_hflip, do_vflip=do_vflip,
                    )
                    # Flip masks to match flipped features
                    if do_hflip:
                        batch.masks = batch.masks.flip(-1)
                    if do_vflip:
                        batch.masks = batch.masks.flip(-2)
                    outputs = model(batch, precomputed_backbone_out=backbone_out)
                else:
                    outputs = model(batch)

                targets = batch.masks.float()
                losses = loss_fn(outputs, targets)
                loss = losses["core_loss"]

            # Backward with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
            optimizer.step()
            scheduler.step()
```

**Step by step:**

1. `model.train()` — Enable training mode (dropout active, etc.)
2. `(batch, frame_id_lists) = ...` — Unpack batch and frame IDs from dataloader
3. `batch = batch.to(device)` — Move data to GPU
4. If precomputed features available:
   a. Random flip decision (feature-level augmentation)
   b. Assemble cached backbone_out from frame IDs
   c. Flip masks to match flipped features
   d. `outputs = model(batch, precomputed_backbone_out=backbone_out)` — Skip encoder
5. Otherwise: `outputs = model(batch)` — Full forward with encoder
6. `losses = loss_fn(outputs, targets)` — Compute all loss terms
7. `loss = losses["core_loss"]` — Get combined loss
8. `optimizer.zero_grad()` — Clear old gradients
9. `loss.backward()` — Compute gradients (backpropagation)
10. `clip_grad_norm_(..., 1.0)` — Prevent exploding gradients
11. `optimizer.step()` — Update weights
12. `scheduler.step()` — Update learning rate

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

# B200 / GPU Performance Optimizations

These are applied automatically when running on CUDA. None of them change training quality — they only change how the math is launched on the GPU.

| Optimization | Where | What it does |
|---|---|---|
| **bfloat16 AMP** | `trainer.py` autocast block | 2× memory savings, 2× throughput on Tensor Cores. No GradScaler — bf16 has fp32 exponent range, no underflow. |
| **Channels-last memory format** | `build_training_model()` | `model.to(memory_format=torch.channels_last)` — only touches 4D Conv weights. Encoder input is converted in the wrapper forward. cuDNN picks fast Tensor Core conv kernels. |
| **Pinned memory + non_blocking H2D** | DataLoader + `BatchedVideoDatapoint.to()` | Async copy from CPU to GPU, overlaps with subsequent GPU compute. |
| **Deferred `.item()` syncs** | `trainer.py` step loop | Loss tensors stay on-device during the epoch; `.item()` only runs at logging boundaries (every `log_every_n_steps`). Each `.item()` is a `cudaStreamSynchronize`, so this removes per-step stalls. |
| **Parallel scipy distance transforms** | `loss_fns.py::_compute_distance_weights` | cbDice's per-batch CPU stall split across 4 threads via `ThreadPoolExecutor`. `scipy.ndimage.distance_transform_edt` releases the GIL, so threads genuinely parallelise. Bit-identical output. |
| **Fused AdamW** | `build_optimizer()` | `fused=True` packs the entire optimizer step (m/v/weight/wd) into one CUDA kernel. With LoRA's 136 tiny adapter tensors, this collapses dozens of kernel launches per step into one. |
| **cuDNN SDPA backend** | `train()` startup | `torch.backends.cuda.enable_cudnn_sdp(True)` — picks NVIDIA's FlashAttention-3 kernel for `F.scaled_dot_product_attention` on Blackwell. SAM2 already uses SDPA at three sites (Hiera self-attn + decoder self/cross attn). |

## Optional opt-in flags

| Flag | What it does |
|---|---|
| `--compile-encoder` | `torch.compile(model.image_encoder)` — adds 30-90s first-call compile cost, then ~15-30% faster encoder forward |
| `--compile-decoder` | `torch.compile(model.sam_mask_decoder)` — SAM2's own video predictor compiles this same module |

Compile is opt-in because the first-call overhead and occasional graph-break warnings make A/B testing harder if it's always on.

---

# Diagnostics in the Training Loop

Each validation epoch (`evaluate.py::validate_epoch`) prints in addition to the headline `val_loss / val_iou / val_dice`:

- **`val_loss_components`** — per-loss-function values (mask, dice, cbdice, supervoxel, iou, class). Shows whether overfitting is per-component or global.
- **`per_frame_iou`** — IoU at each frame index `t=0..7`. The `t0 → t7` drop quantifies the propagation chain error (t0 has the GT prompt and is the ceiling).
- **`per_video_iou`** — IoU per validation video. Catches OOD videos that drag down the mean.
- **Worst-IoU PNG dumps** — saved to `checkpoints/val_samples/epoch_NNN/`. The 6 worst-IoU `(batch, frame, object=0)` slices across the entire val set, with filenames like `e040_rank00_iou0.42_<video>_t6_f47.png`. Each PNG is `image | GT | prediction | overlay (red=pred, green=GT, yellow=both)`. Open them to see actual failure modes — a single number can hide whether the model is failing because of label noise, OOD data, or genuine prediction errors.

The training loop also prints **`train_loss_components`** at end of each epoch. Comparing train vs val components per-component reveals whether a specific loss is over- or under-fitting independently of the others.

---

# Summary: What You Should Know

After reading this document, you should be able to explain:

1. **Why LoRA?** — Parameter-efficient fine-tuning that avoids overfitting with small datasets
2. **LoRA math:** `y = W₀·x + (α/r)·B·A·x`, B initialized to zeros
3. **Why those specific layers?** — Mask decoder (what to segment), memory_attention (temporal propagation), and **as of Run 6**, the last 2 blocks of the Hiera trunk (the deepest encoder features — domain adaptation for microscopy). The decoder + memory_attention are LoRA-adapted at rank 16; Hiera last-stage at rank 8. The encoder's base weights still stay frozen; only the new LoRA adapters train. In Runs 1–4 the wrapper used a simplified previous-mask-as-prompt propagation that bypassed memory_attention entirely.
4. **Why freeze image encoder?** — General features work, would overfit with 10 videos
5. **Data pipeline:** TIF → PNG (lossless), per-frame normalization, dataset stats
6. **Augmentations:** Temporal consistency (same transform for all frames in clip); full pixel-level pipeline because `precompute_features=False`
7. **Loss function:** Focal (handles class imbalance) + Dice (shape quality) + cbDice (skeleton topology, weight 5) + Supervoxel (connectivity, weight 0.3) + IoU pred + Object score.
8. **Three learning rates (LoRA+):** LoRA A (1e-4) vs LoRA B (1.6e-3, 16× higher) vs embeddings (1e-5)
9. **Cosine schedule with floor:** 2-epoch linear warmup → cosine decay → floor at 5% of peak (GPT-3 style, half their floor)
10. **`config.py` is the source of truth:** CLI flags only override when explicitly passed
11. **B200 perf:** bf16 (no GradScaler), channels-last, FlashAttention-3 via cuDNN SDPA, fused AdamW, parallel cbDice scipy, pinned memory + non_blocking copies, deferred `.item()` syncs
12. **Diagnostics:** Per-component loss + per-frame IoU + per-video IoU + worst-IoU PNG dumps — every validation epoch
13. **Chained training:** `--resume train_state.pt` lets one logical training run span multiple SLURM jobs under a wall-clock limit (e.g. 90 epochs across two 4-hour B200 jobs)

Good luck with your paper and presentation!
