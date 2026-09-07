"""
LoRA+ (Low-Rank Adaptation with asymmetric learning rates) for SAM 2 fine-tuning.

Wraps nn.Linear layers with low-rank adapters so that:
    output = original_linear(x) + (scale) * B @ A @ x

Only the A and B matrices are trainable, keeping the original weights frozen.

LoRA+ improvement (Hayou et al., ICML 2024): Uses different learning rates
for A and B matrices. B starts at zeros and needs a higher LR (ratio ~16x)
to overcome gradient imbalance, yielding faster convergence at zero extra cost.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with a low-rank adapter.

    forward(x) = original(x) + scale * dropout(B(A(x)))
    where scale = alpha / rank
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # Low-rank matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Dropout on the low-rank path
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize A with Kaiming uniform and B with zeros
        # so the adapter starts as identity (no change to original output)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        # Freeze the original linear
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return original_out + self.scale * lora_out


# Default LoRA target patterns for SAM 2
# These patterns work for both SAM2Train (model.xxx) and video predictor (xxx) structures
SAM2_LORA_TARGETS = [
    # Mask decoder transformer attention layers (with optional model. prefix)
    r"(model\.)?sam_mask_decoder\.transformer\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)",
    r"(model\.)?sam_mask_decoder\.transformer\.layers\.\d+\.cross_attn_token_to_image\.(q_proj|k_proj|v_proj|out_proj)",
    r"(model\.)?sam_mask_decoder\.transformer\.layers\.\d+\.cross_attn_image_to_token\.(q_proj|k_proj|v_proj|out_proj)",
    # Mask decoder final attention
    r"(model\.)?sam_mask_decoder\.transformer\.final_attn_token_to_image\.(q_proj|k_proj|v_proj|out_proj)",
    # Memory attention layers
    r"(model\.)?memory_attention\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)",
    r"(model\.)?memory_attention\.layers\.\d+\.cross_attn_image\.(q_proj|k_proj|v_proj|out_proj)",
    r"(model\.)?memory_attention\.layers\.\d+\.(linear1|linear2)",
]


# Hiera image-encoder last-stage targets. Hiera-tiny has stages [1, 2, 7, 2]
# so 12 blocks total (indices 0..11). The last stage is blocks 10 and 11 —
# the deepest, highest-dim (768-channel) features. These are the most
# domain-specific representations in the encoder and the part most worth
# adapting for microscopy (vs the early stages, which capture low-level
# edges/textures that transfer fine from natural images).
#
# Each MultiScaleBlock has:
#   .attn.qkv      — fused Q+K+V projection (in_dim -> 3 * out_dim)
#   .attn.proj     — output projection
#   .mlp.layers.0  — first MLP linear
#   .mlp.layers.1  — second MLP linear
# We LoRA-adapt all four per block (8 layers across the last 2 blocks).
HIERA_LAST_STAGE_LORA_TARGETS = [
    r"(model\.)?image_encoder\.trunk\.blocks\.(10|11)\.attn\.(qkv|proj)",
    r"(model\.)?image_encoder\.trunk\.blocks\.(10|11)\.mlp\.layers\.\d+",
]


def hiera_trailing_block_targets(num_blocks_total: int, num_adapt: int) -> List[str]:
    """LoRA patterns for the LAST ``num_adapt`` Hiera trunk blocks.

    The block indices are derived from the trunk actually loaded rather than
    hardcoded, so this stays correct if the backbone is swapped. For
    sam2.1-hiera-tiny the trunk is ``stages: [1, 2, 7, 2]`` = 12 blocks:

        stage 0 -> block 0       (96-dim)
        stage 1 -> blocks 1-2    (192-dim)
        stage 2 -> blocks 3-9    (384-dim)   global attention at 5, 7, 9
        stage 3 -> blocks 10-11  (768-dim)

    num_adapt=2 reproduces HIERA_LAST_STAGE_LORA_TARGETS exactly (stage 3
    only). num_adapt=4 reaches back into stage 2 and picks up block 9, which
    is a GLOBAL-attention block -- it attends over the whole image rather than
    a local window, which is the computation that has to localise a neuron
    when no ground-truth prompt is available. Note the two stages differ in
    width (384 vs 768), so a given rank buys proportionally more capacity in
    stage 3 than in stage 2.

    Four Linear layers are matched per block (attn.qkv, attn.proj,
    mlp.layers.0, mlp.layers.1). The block-level ``proj`` that Hiera inserts
    at a stage transition is deliberately NOT matched, matching the previous
    behaviour.
    """
    if num_adapt <= 0:
        return []
    lo = max(0, num_blocks_total - num_adapt)
    idx = "|".join(str(i) for i in range(lo, num_blocks_total))
    return [
        rf"(model\.)?image_encoder\.trunk\.blocks\.({idx})\.attn\.(qkv|proj)",
        rf"(model\.)?image_encoder\.trunk\.blocks\.({idx})\.mlp\.layers\.\d+",
    ]


def _matches_any_pattern(name: str, patterns: List[str]) -> bool:
    """Check if a module name matches any of the target patterns."""
    return any(re.fullmatch(p, name) for p in patterns)


def apply_lora(
    model: nn.Module,
    target_patterns: Optional[List[str]] = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> Set[str]:
    """
    Apply LoRA adapters to matching nn.Linear layers in the model.

    Args:
        model: The SAM 2 model (SAM2Base or SAM2Train).
        target_patterns: Regex patterns for module names to wrap.
                        Defaults to SAM2_LORA_TARGETS.
        rank: LoRA rank (low-rank dimension).
        alpha: LoRA scaling factor.
        dropout: Dropout on the LoRA path.

    Returns:
        Set of module names that were wrapped with LoRA.
    """
    if target_patterns is None:
        target_patterns = SAM2_LORA_TARGETS

    wrapped_names = set()

    # Build a flat dict of all named modules for parent lookup
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


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get only the LoRA adapter parameters (A and B matrices)."""
    params = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            params.extend(module.lora_A.parameters())
            params.extend(module.lora_B.parameters())
    return params


def get_trainable_parameters(model: nn.Module) -> Dict[str, List[nn.Parameter]]:
    """
    Get all trainable parameters grouped by type (LoRA+ compatible).

    LoRA+ (Hayou et al., ICML 2024) uses different learning rates for the
    A (down-projection) and B (up-projection) matrices. B starts at zeros
    and needs a higher LR to overcome the gradient imbalance.

    Returns dict with keys:
        'lora_a': LoRA A (down-projection) parameters — standard LR
        'lora_b': LoRA B (up-projection) parameters — higher LR (LoRA+)
        'other': Other unfrozen parameters (embeddings, small projections)
    """
    lora_a_params = []
    lora_b_params = []
    other_params = []

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_module_names.add(name)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Check if this param belongs to a LoRA module
        is_lora = any(name.startswith(ln) for ln in lora_module_names)
        if is_lora:
            if ".lora_A." in name:
                lora_a_params.append(param)
            elif ".lora_B." in name:
                lora_b_params.append(param)
            else:
                other_params.append(param)  # Shouldn't happen, but safe
        else:
            other_params.append(param)

    return {"lora_a": lora_a_params, "lora_b": lora_b_params, "other": other_params}


def _normalize_key(key: str) -> str:
    """Remove 'model.' prefix from key for consistent storage."""
    if key.startswith("model."):
        return key[6:]
    return key


def save_train_state(
    path: str,
    *,
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_val_iou: float,
    epochs_without_improvement: int,
    num_epochs: int,
    steps_per_epoch: int,
    topo_state: Optional[dict] = None,
) -> None:
    """Save a full training-state checkpoint so a run can resume mid-schedule.

    Unlike save_lora_weights (which saves only the model parameters needed
    for inference), this saves everything required to continue training
    from the exact point we left off:
        - LoRA + other trainable model params (same as save_lora_weights)
        - Optimizer state (Adam moments)
        - LR scheduler state (preserves cosine schedule position)
        - Counters: epoch, best_val_iou, epochs_without_improvement
        - num_epochs and steps_per_epoch the original schedule was built with
          (so a resume can detect a schedule mismatch and warn)

    The resulting file is bigger than lora_weights.pt because of the Adam
    moments, but at ~5M trainable params × 2 (m + v) × 4 bytes (fp32) it
    is still only ~40 MB per checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Reuse save_lora_weights' layout for the model side so that the
    # resulting file is recognisable to load_lora_weights too if needed.
    lora_state = {}
    other_state = {}
    cfg = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            prefix = _normalize_key(name)
            lora_state[f"{prefix}.lora_A.weight"] = module.lora_A.weight.data.cpu()
            lora_state[f"{prefix}.lora_B.weight"] = module.lora_B.weight.data.cpu()
            if not cfg:
                cfg = {"rank": module.rank, "alpha": module.alpha}
    for name, param in model.named_parameters():
        if param.requires_grad:
            is_lora = any(
                name.startswith(n)
                for n, m in model.named_modules()
                if isinstance(m, LoRALinear)
            )
            if not is_lora:
                other_state[_normalize_key(name)] = param.data.cpu()

    torch.save(
        {
            "lora_state_dict": lora_state,
            "other_state_dict": other_state,
            "config": cfg,
            # Training-continuation state below.
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": int(epoch),
            "best_val_iou": float(best_val_iou),
            "epochs_without_improvement": int(epochs_without_improvement),
            # Tie-break bookkeeping for lora_weights_topo.pt. Optional and
            # defaulted on load, so checkpoints written before this existed
            # still resume cleanly.
            "topo_state": topo_state or {},
            "num_epochs": int(num_epochs),
            "steps_per_epoch": int(steps_per_epoch),
        },
        path,
    )


def load_train_state(
    path: str,
    *,
    model: nn.Module,
    optimizer,
    scheduler,
    device: str = "cpu",
) -> dict:
    """Restore a full training state from `save_train_state` output.

    Loads model params (via load_lora_weights logic), then restores the
    optimizer and scheduler in-place. Returns a dict with the resume
    counters:
        {"start_epoch", "best_val_iou", "epochs_without_improvement",
         "num_epochs", "steps_per_epoch"}

    start_epoch is the epoch number to BEGIN with on resume (= saved
    epoch + 1, because the saved epoch was already completed).
    """
    state = torch.load(path, map_location=device, weights_only=False)

    # Restore model params using the same key-matching logic as
    # load_lora_weights — we reuse it for consistency.
    load_lora_weights(model, str(path), device=device)

    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])

    return {
        "start_epoch": int(state["epoch"]) + 1,
        "best_val_iou": float(state["best_val_iou"]),
        "epochs_without_improvement": int(state["epochs_without_improvement"]),
        "num_epochs": int(state.get("num_epochs", 0)),
        "steps_per_epoch": int(state.get("steps_per_epoch", 0)),
        "topo_state": state.get("topo_state") or {},
    }


def save_lora_weights(model: nn.Module, path: str):
    """
    Save only the LoRA adapter weights and any other trainable parameters.

    The saved file contains:
        'lora_state_dict': dict of LoRA A/B weights keyed by full module path
        'other_state_dict': dict of other trainable parameters
        'config': LoRA configuration (rank, alpha)

    Keys are normalized to remove 'model.' prefix for compatibility between
    training (SAM2Train) and inference (video predictor) models.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lora_state = {}
    other_state = {}
    config = {}

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Normalize the prefix to remove 'model.' if present
            prefix = _normalize_key(name)
            lora_state[f"{prefix}.lora_A.weight"] = module.lora_A.weight.data.cpu()
            lora_state[f"{prefix}.lora_B.weight"] = module.lora_B.weight.data.cpu()
            if not config:
                config = {
                    "rank": module.rank,
                    "alpha": module.alpha,
                }

    # Save other trainable params (embeddings etc.)
    for name, param in model.named_parameters():
        if param.requires_grad:
            is_lora = any(
                name.startswith(n)
                for n, m in model.named_modules()
                if isinstance(m, LoRALinear)
            )
            if not is_lora:
                # Normalize the key
                normalized_name = _normalize_key(name)
                other_state[normalized_name] = param.data.cpu()

    torch.save(
        {
            "lora_state_dict": lora_state,
            "other_state_dict": other_state,
            "config": config,
        },
        path,
    )
    total_params = sum(p.numel() for p in lora_state.values()) + sum(
        p.numel() for p in other_state.values()
    )
    print(f"Saved LoRA weights to {path} ({total_params:,} parameters)")


def load_lora_weights(model: nn.Module, path: str, device: str = "cpu"):
    """
    Load LoRA weights into a model that already has LoRA applied.

    Args:
        model: Model with LoRA adapters already applied via apply_lora().
        path: Path to the saved LoRA weights file.
        device: Device to load weights onto.

    Handles both old checkpoints (with 'model.' prefix) and new checkpoints
    (normalized without prefix) for compatibility.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    lora_state = checkpoint["lora_state_dict"]
    other_state = checkpoint.get("other_state_dict", {})

    # Build a mapping from LoRA module names to their modules
    # This ensures we move the actual module objects to the device
    lora_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_modules[name] = module
            # Move entire LoRA module to device
            module.to(device)

    # Get current model state after moving modules
    model_state = model.state_dict()

    # Build a normalized key mapping for the model state
    # This maps both "xxx" and "model.xxx" to the actual key in model_state
    model_key_map = {}
    for key in model_state.keys():
        model_key_map[key] = key
        # Also map normalized version
        normalized = _normalize_key(key)
        if normalized != key:
            model_key_map[normalized] = key

    # Try to match keys, handling potential "model." prefix mismatch
    loaded_count = 0
    for ckpt_key, value in lora_state.items():
        # Normalize the checkpoint key
        normalized_ckpt_key = _normalize_key(ckpt_key)

        # Try to find matching model key
        model_key = None
        if ckpt_key in model_key_map:
            model_key = model_key_map[ckpt_key]
        elif normalized_ckpt_key in model_key_map:
            model_key = model_key_map[normalized_ckpt_key]

        if model_key is not None:
            model_state[model_key].copy_(value.to(device))
            loaded_count += 1
        else:
            # Only warn for LoRA weights
            if "lora_" in ckpt_key:
                print(f"Warning: LoRA key {ckpt_key} not found in model")

    # Load other trainable weights with same handling
    for ckpt_key, value in other_state.items():
        normalized_ckpt_key = _normalize_key(ckpt_key)

        model_key = None
        if ckpt_key in model_key_map:
            model_key = model_key_map[ckpt_key]
        elif normalized_ckpt_key in model_key_map:
            model_key = model_key_map[normalized_ckpt_key]

        if model_key is not None:
            model_state[model_key].copy_(value.to(device))
        # Don't warn for other state - may have different structure

    print(f"Loaded LoRA weights from {path} ({loaded_count} LoRA tensors)")
