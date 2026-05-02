"""
Standalone training script for SAM 2 LoRA fine-tuning on neuron data.

This bypasses Meta's complex distributed Trainer and uses a simple PyTorch
training loop suitable for single-GPU training.

Key differences from Meta's internal training:
- Uses standard build_sam2() instead of SAM2Train (which isn't in pip package)
- Custom forward pass that handles video frames and mask propagation
- Local loss function implementation (loss_fns.py)

Performance optimizations:
- Feature precomputation: frozen image encoder output cached once, skipped
  during training (~10x speedup)
- In-memory caching: all frames/masks loaded into RAM at init
- Clip stride: skip overlapping clips to reduce steps/epoch
- Feature-level flip augmentation: flip cached feature maps instead of pixels
"""

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from sam2.build_sam import build_sam2, _hf_download

from src.config import TrainingConfig
from src.lora import (
    apply_lora,
    get_trainable_parameters,
    save_lora_weights,
    SAM2_LORA_TARGETS,
)
from src.neuron_dataset import (
    NeuronVideoDataset,
    neuron_collate_fn,
    get_video_names,
    train_val_split,
)
from src.augmentations import VideoAugmentation, VideoValidationTransform
from src.evaluate import validate_epoch
from src.loss_fns import MultiStepMultiMasksAndIous
from src.data_utils import BatchedVideoDatapoint


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Feature precomputation (10x speedup for frozen image encoder)
# ---------------------------------------------------------------------------

def precompute_all_features(
    model: "SAM2TrainingWrapper",
    train_dataset: NeuronVideoDataset,
    val_dataset: NeuronVideoDataset,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    image_size: int,
    norm_mean: List[float],
    norm_std: List[float],
) -> Tuple[Dict, List[torch.Tensor]]:
    """Precompute frozen image encoder features for all frames.

    Since the image encoder is frozen (no LoRA on Conv2d layers), its output
    is deterministic for each input frame. Computing it once and caching
    eliminates ~60-70% of per-step compute.

    Args:
        model: SAM2TrainingWrapper (uses model.model for the inner SAM2).
        train_dataset: Training dataset (to iterate over videos).
        val_dataset: Validation dataset (to iterate over videos).
        device: Target device for encoder forward pass.
        use_amp: Whether to use automatic mixed precision.
        amp_dtype: AMP data type (bfloat16 on CUDA, float32 otherwise).
        image_size: Image size (1024 for SAM2).
        norm_mean: Normalization mean (per-channel).
        norm_std: Normalization std (per-channel).

    Returns:
        feature_cache: Dict mapping (video_name, frame_idx) to list of 3 FPN
                       feature tensors in float16 on CPU.
        pos_enc: List of 3 positional encoding tensors on CPU (spatial-only,
                 same for all frames — computed once).
    """
    sam2 = model.model  # Inner SAM2Base model
    was_training = sam2.training
    sam2.eval()

    feature_cache: Dict[Tuple[str, int], List[torch.Tensor]] = {}
    pos_enc: Optional[List[torch.Tensor]] = None

    # Collect all unique videos from both datasets
    all_videos = []
    seen_videos = set()
    for dataset in [train_dataset, val_dataset]:
        for vid in dataset.videos:
            if vid["name"] not in seen_videos:
                all_videos.append((vid, dataset))
                seen_videos.add(vid["name"])

    total_frames = sum(vid["num_frames"] for vid, _ in all_videos)
    processed = 0
    start_time = time.time()

    print(f"  {len(all_videos)} videos, {total_frames} frames total")

    with torch.no_grad():
        for vid, dataset in all_videos:
            for idx in range(vid["num_frames"]):
                # Load raw frame (from cache or disk)
                frame = dataset._load_frame(vid["name"], vid["img_dir"], idx)

                # Apply same preprocessing as VideoValidationTransform:
                # resize to image_size and normalize
                frame = TF.resize(
                    frame,
                    [image_size, image_size],
                    interpolation=TF.InterpolationMode.BILINEAR,
                    antialias=True,
                )
                frame = TF.normalize(frame, norm_mean, norm_std)

                frame_batch = frame.unsqueeze(0).to(device)
                if device == "cuda":
                    frame_batch = frame_batch.contiguous(memory_format=torch.channels_last)

                with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_amp):
                    backbone_out = sam2.forward_image(frame_batch)

                key = (vid["name"], idx)
                # Cache backbone_fpn as float16 on CPU (saves ~50% memory)
                feature_cache[key] = [
                    f.squeeze(0).cpu().half()
                    for f in backbone_out["backbone_fpn"]
                ]

                # Store positional encodings once (they're purely spatial,
                # independent of image content — same for all frames)
                if pos_enc is None:
                    pos_enc = [
                        pe.squeeze(0).cpu()
                        for pe in backbone_out["vision_pos_enc"]
                    ]

                processed += 1
                if processed % 100 == 0 or processed == total_frames:
                    print(f"  Processed {processed}/{total_frames} frames...")

    elapsed = time.time() - start_time
    total_mb = sum(
        sum(f.nelement() * f.element_size() for f in feats)
        for feats in feature_cache.values()
    ) / 1e6
    print(f"  Feature precomputation done in {elapsed:.1f}s ({total_mb:.0f} MB cached)")

    # Clean up GPU memory
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    if was_training:
        sam2.train()

    return feature_cache, pos_enc


def assemble_cached_backbone(
    feature_cache: Dict[Tuple[str, int], List[torch.Tensor]],
    pos_enc: List[torch.Tensor],
    frame_id_lists: List[List[Tuple[str, int]]],
    device: str,
    do_hflip: bool = False,
    do_vflip: bool = False,
) -> Dict[str, list]:
    """Assemble backbone_out dict from cached per-frame features.

    Reconstructs the same dict that model.forward_image() would return,
    but from pre-cached features instead of running the encoder.

    Optionally applies flip augmentation at the feature level (valid because
    the frozen encoder has no learned spatial asymmetry).

    Args:
        feature_cache: Dict mapping (video_name, frame_idx) to list of 3
                       FPN feature tensors (float16, CPU).
        pos_enc: List of 3 positional encoding tensors (CPU).
        frame_id_lists: List (one per batch item) of lists of
                        (video_name, frame_idx) tuples.
        device: Target device.
        do_hflip: Apply horizontal flip to feature maps.
        do_vflip: Apply vertical flip to feature maps.

    Returns:
        backbone_out dict with 'backbone_fpn' and 'vision_pos_enc', compatible
        with model._prepare_backbone_features().
    """
    # Flatten frame_id_lists to match [B*T] ordering:
    # [vid0_f0, vid0_f1, ..., vid0_fT-1, vid1_f0, ..., vid1_fT-1]
    flat_ids = []
    for frame_ids in frame_id_lists:
        flat_ids.extend(frame_ids)

    num_levels = len(feature_cache[flat_ids[0]])
    backbone_fpn = []
    for level in range(num_levels):
        level_feats = [feature_cache[fid][level] for fid in flat_ids]
        stacked = torch.stack(level_feats, dim=0).to(device).float()

        # Feature-level flip augmentation
        if do_hflip:
            stacked = stacked.flip(-1)
        if do_vflip:
            stacked = stacked.flip(-2)

        backbone_fpn.append(stacked)

    # Expand pos_enc to match batch size (PE is spatial-only, same for all)
    BT = len(flat_ids)
    vision_pos_enc = [
        pe.unsqueeze(0).expand(BT, -1, -1, -1).to(device)
        for pe in pos_enc
    ]

    return {
        "backbone_fpn": backbone_fpn,
        "vision_pos_enc": vision_pos_enc,
    }


# ---------------------------------------------------------------------------
# SAM2 Training Wrapper
# ---------------------------------------------------------------------------

class SAM2TrainingWrapper(nn.Module):
    """Wrapper that adapts SAM2Base for training with video clips.

    SAM2 models from the pip package don't have a training-specific forward.
    This wrapper:
    1. Takes batched video data (images + masks)
    2. Processes frames through the image encoder (or uses precomputed features)
    3. Simulates mask propagation using the memory mechanism
    4. Returns predicted masks for loss computation
    """

    def __init__(self, sam2_model: nn.Module, num_frames: int = 8):
        super().__init__()
        self.model = sam2_model
        self.num_frames = num_frames

    def forward(
        self,
        batch: BatchedVideoDatapoint,
        precomputed_backbone_out: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            batch: BatchedVideoDatapoint with:
                - images: [B, T, C, H, W] video frames
                - masks: [B, T, N_obj, H, W] ground truth masks
            precomputed_backbone_out: Optional dict from assemble_cached_backbone().
                If provided, skips the image encoder entirely (10x faster).

        Returns:
            Dict with:
                - pred_masks: [B, T, N_obj, H, W] predicted mask logits
        """
        images = batch.images  # [B, T, C, H, W]
        gt_masks = batch.masks  # [B, T, N_obj, H, W]

        B, T, C, H, W = images.shape
        N_obj = gt_masks.shape[2]

        # Encode all frames through image encoder (or use precomputed features)
        if precomputed_backbone_out is not None:
            backbone_out = precomputed_backbone_out
        else:
            # Reshape to [B*T, C, H, W] for batch processing.
            # Convert to channels-last so cuDNN picks the fast tensor-core
            # conv kernels in the Hiera encoder. Weights were already
            # converted in build_training_model().
            images_flat = images.view(B * T, C, H, W)
            if images_flat.is_cuda:
                images_flat = images_flat.contiguous(memory_format=torch.channels_last)
            with torch.set_grad_enabled(self.training):
                backbone_out = self.model.forward_image(images_flat)

        # Get the image features (handle different SAM2 versions)
        if "backbone_fpn" in backbone_out:
            # SAM 2.1 format - _prepare_backbone_features returns features
            # in format that may vary depending on model configuration
            _, vision_feats, vision_pos_embeds, feat_sizes = self.model._prepare_backbone_features(backbone_out)

            # Store all feature levels for high-res features
            self._current_vision_feats = vision_feats
            self._current_feat_sizes = feat_sizes

            # vision_feats is a list of tensors at different resolutions
            # Take the highest resolution (last element) for main features
            feat = vision_feats[-1]

            # Handle different possible shapes from SAM2:
            # - [B*T, C, H, W] - standard 4D format
            # - [B*T, C, H*W] - flattened spatial (common in transformers)
            # - [HW, B*T, C] - transformer sequence format (HW tokens, batch, channels)
            if feat.dim() == 3:
                # 3D tensor - need to determine format and reshape
                dim0, dim1, dim2 = feat.shape

                # Check for transformer format: [HW, BT, C] where HW=4096, BT=4, C=256
                # Total elements = 4194304 = 4096 * 4 * 256
                if dim0 == 4096 and dim2 == 256:
                    # Format: [H*W, B*T, C] -> [B*T, C, H, W]
                    HW, BT, C_feat = feat.shape
                    H_feat = W_feat = int(HW ** 0.5)  # 64
                    feat = feat.permute(1, 2, 0).view(BT, C_feat, H_feat, W_feat)
                elif dim1 == 256:
                    # Format: [BT, C, H*W] -> [BT, C, H, W]
                    BT, C_feat, HW = feat.shape
                    H_feat = W_feat = int(HW ** 0.5)
                    feat = feat.view(BT, C_feat, H_feat, W_feat)
                elif dim2 == 256:
                    # Format: [BT, H*W, C] -> [BT, C, H, W]
                    BT, HW, C_feat = feat.shape
                    H_feat = W_feat = int(HW ** 0.5)
                    feat = feat.permute(0, 2, 1).view(BT, C_feat, H_feat, W_feat)
                else:
                    raise ValueError(f"Unknown 3D feature format: {feat.shape}")
            # Now feat should be [B*T, C, H, W]
        else:
            # Fallback: direct feature access
            feat = backbone_out["vision_features"]
            self._current_vision_feats = None

        # Reshape features back to video format
        _, C_feat, H_feat, W_feat = feat.shape
        feat = feat.view(B, T, C_feat, H_feat, W_feat)

        # For each object, propagate masks through time
        # We use the ground truth mask from frame 0 as the prompt
        pred_masks_all = []

        for obj_idx in range(N_obj):
            # Get GT mask for this object at frame 0 as prompt
            prompt_mask = gt_masks[:, 0, obj_idx]  # [B, H, W]

            # Collect predictions for all frames
            obj_pred_masks = []

            for t in range(T):
                # Get features for this frame
                frame_feat = feat[:, t]  # [B, C_feat, H_feat, W_feat]

                if t == 0:
                    # First frame: use the GT mask as prompt
                    # Resize mask to feature resolution for conditioning
                    mask_input = F.interpolate(
                        prompt_mask.unsqueeze(1).float(),
                        size=(H_feat, W_feat),
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    # Subsequent frames: use previous prediction as memory
                    # (simplified - in full SAM2 this uses memory attention)
                    mask_input = F.interpolate(
                        obj_pred_masks[-1].unsqueeze(1),
                        size=(H_feat, W_feat),
                        mode="bilinear",
                        align_corners=False,
                    )

                # Predict mask for this frame
                # We use the mask decoder with dense embeddings
                pred_mask = self._decode_mask(frame_feat, mask_input)
                obj_pred_masks.append(pred_mask)

            # Stack predictions for this object: [B, T, H, W]
            obj_pred_masks = torch.stack(obj_pred_masks, dim=1)
            pred_masks_all.append(obj_pred_masks)

        # Stack all objects: [B, T, N_obj, H, W]
        pred_masks = torch.stack(pred_masks_all, dim=2)

        return {"pred_masks": pred_masks}

    def _decode_mask(
        self, image_features: torch.Tensor, mask_input: torch.Tensor, batch_offset: int = 0
    ) -> torch.Tensor:
        """Decode a mask prediction from image features and mask prompt.

        Args:
            image_features: [B, C, H_feat, W_feat] image encoder output
            mask_input: [B, 1, H_feat, W_feat] mask prompt (GT or previous pred)
            batch_offset: Offset into stored FPN features for this frame

        Returns:
            pred_mask: [B, H, W] predicted mask logits at original resolution
        """
        B, C, H_feat, W_feat = image_features.shape

        # Get high-res features from stored FPN outputs
        # vision_feats[0] and [1] are higher resolution features needed by mask decoder
        high_res_features = None
        if self._current_vision_feats is not None and len(self._current_vision_feats) >= 2:
            # Extract the two highest resolution feature maps
            # vision_feats[0]: highest res (e.g., 256x256), vision_feats[1]: mid res (e.g., 128x128)
            feat_s0 = self._current_vision_feats[0]  # [HW, BT, C] format
            feat_s1 = self._current_vision_feats[1]  # [HW, BT, C] format

            # Get the correct batch slice and reshape to [B, C, H, W]
            # For now, extract features for batch_offset
            if feat_s0.dim() == 3:
                # [HW, BT, C] -> need to extract per-frame and reshape
                HW0, BT, C0 = feat_s0.shape
                H0 = W0 = int(HW0 ** 0.5)
                feat_s0_reshaped = feat_s0.permute(1, 2, 0)[:B].reshape(B, C0, H0, W0)

                HW1, _, C1 = feat_s1.shape
                H1 = W1 = int(HW1 ** 0.5)
                feat_s1_reshaped = feat_s1.permute(1, 2, 0)[:B].reshape(B, C1, H1, W1)

                high_res_features = (feat_s0_reshaped, feat_s1_reshaped)

        # Use the model's mask decoder
        sparse_embeddings = self.model.sam_prompt_encoder.not_a_point_embed.weight
        sparse_embeddings = sparse_embeddings.unsqueeze(0).expand(B, -1, -1)

        # Encode mask_input as dense prompt embeddings through SAM2's mask downscaling
        # mask_input is [B, 1, H_feat, W_feat] - resize to 256x256 as SAM2 expects
        mask_for_encoder = F.interpolate(
            mask_input,
            size=(256, 256),  # SAM2 expects 4x the 64x64 embedding resolution
            mode="bilinear",
            align_corners=False,
        )
        # Encode through the prompt encoder's convolutional downscaling network
        dense_prompt = self.model.sam_prompt_encoder.mask_downscaling(mask_for_encoder)

        # Call mask decoder with high_res_features
        # SAM2 mask decoder returns: (masks, iou_predictions, mask_tokens_out, object_score_logits)
        decoder_output = self.model.sam_mask_decoder(
            image_embeddings=image_features,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_prompt,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        low_res_masks = decoder_output[0]  # First element is the masks

        # Upscale to original resolution
        # low_res_masks: [B, 1, H_low, W_low]
        high_res_masks = F.interpolate(
            low_res_masks,
            size=(1024, 1024),  # SAM 2 default size
            mode="bilinear",
            align_corners=False,
        )

        return high_res_masks.squeeze(1)  # [B, H, W]


def build_training_model(config: TrainingConfig, device: str):
    """
    Build SAM2 model with LoRA adapters for training.

    Uses the standard pip-installable SAM2 model wrapped for training,
    instead of the internal SAM2Train class.

    Returns:
        model: SAM2TrainingWrapper with LoRA applied
        loss_fn: MultiStepMultiMasksAndIous loss function
    """
    # Download checkpoint
    config_name, ckpt_path = _hf_download(config.sam2_model_id)

    # Build the standard SAM2 model
    model = build_sam2(
        config_file=config_name,
        ckpt_path=ckpt_path,
        device="cpu",  # Load on CPU first
        apply_postprocessing=False,
    )

    # Freeze image encoder if configured
    if config.freeze_image_encoder:
        for p in model.image_encoder.parameters():
            p.requires_grad = False

    # Apply LoRA to target modules
    print("\nApplying LoRA adapters...")
    wrapped = apply_lora(
        model,
        target_patterns=SAM2_LORA_TARGETS,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
    )
    print(f"  Wrapped {len(wrapped)} Linear layers with LoRA (rank={config.lora.rank})")
    for name in sorted(wrapped):
        print(f"    - {name}")

    # Unfreeze small learnable embeddings
    unfrozen_embeddings = []
    for name, param in model.named_parameters():
        if any(
            key in name
            for key in [
                "maskmem_tpos_enc",
                "no_mem_embed",
                "no_mem_pos_enc",
                "no_obj_ptr",
                "obj_ptr_proj",
                "obj_ptr_tpos_proj",
            ]
        ):
            param.requires_grad = True
            unfrozen_embeddings.append(name)

    if unfrozen_embeddings:
        print(f"\n  Also unfroze {len(unfrozen_embeddings)} embedding parameters:")
        for name in unfrozen_embeddings:
            print(f"    - {name}")

    # Wrap for training
    training_model = SAM2TrainingWrapper(model, num_frames=config.data.num_frames)

    # Count parameters
    total_params = sum(p.numel() for p in training_model.parameters())
    trainable_params = sum(p.numel() for p in training_model.parameters() if p.requires_grad)
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    # Build loss function with topology-preserving losses
    loss_fn = MultiStepMultiMasksAndIous(
        weight_dict=config.loss_weight_dict,
        focal_alpha=config.loss.focal_alpha,
        focal_gamma=config.loss.focal_gamma,
        pred_obj_scores=True,
        use_cbdice=config.loss.use_cbdice,
        cbdice_num_iter=config.loss.cbdice_num_iter,
        use_supervoxel=config.loss.use_supervoxel,
        topology_resolution=config.loss.topology_resolution,
    )
    if config.loss.use_cbdice:
        print(f"  Using cbDice loss for skeleton topology (weight={config.loss.loss_cbdice_weight})")
    if config.loss.use_supervoxel:
        print(f"  Using supervoxel loss for connectivity (weight={config.loss.loss_supervoxel_weight})")

    training_model = training_model.to(device)

    # Channels-last memory format on CUDA. .to(memory_format=...) only
    # touches 4D Conv weights — Linear / LayerNorm / Embedding stay as-is.
    # Pair with channels_last input in the wrapper forward (below) so
    # cuDNN picks the fast tensor-core conv kernels on Hopper/Blackwell.
    # Numerically equivalent to bf16 precision — different reduction
    # order in cuDNN may shift values by ≤1 ulp but no systematic bias.
    if device == "cuda":
        training_model = training_model.to(memory_format=torch.channels_last)

    return training_model, loss_fn


def build_optimizer(model: nn.Module, config: TrainingConfig):
    """Build AdamW optimizer with LoRA+ asymmetric A/B learning rates.

    LoRA+ (Hayou et al., ICML 2024) uses different learning rates for A and B:
    - A (down-projection): standard lora_lr
    - B (up-projection): lora_lr × lora_b_lr_ratio (default 16×)

    B starts at zeros and needs a higher LR to overcome gradient imbalance.
    """
    param_groups = get_trainable_parameters(model)

    lora_b_lr = config.optimizer.lora_lr * config.optimizer.lora_b_lr_ratio

    optimizer = torch.optim.AdamW(
        [
            {
                "params": param_groups["lora_a"],
                "lr": config.optimizer.lora_lr,
                "weight_decay": config.optimizer.weight_decay,
            },
            {
                "params": param_groups["lora_b"],
                "lr": lora_b_lr,
                "weight_decay": config.optimizer.weight_decay,
            },
            {
                "params": param_groups["other"],
                "lr": config.optimizer.base_lr,
                "weight_decay": 0.0,  # No weight decay for embeddings
            },
        ],
    )

    print(f"  LoRA+ optimizer: A lr={config.optimizer.lora_lr:.1e}, "
          f"B lr={lora_b_lr:.1e} (ratio={config.optimizer.lora_b_lr_ratio}), "
          f"embeddings lr={config.optimizer.base_lr:.1e}")
    return optimizer


def build_scheduler(optimizer, config: TrainingConfig, steps_per_epoch: int):
    """Build LR scheduler with warmup + cosine annealing with minimum LR floor."""
    total_steps = config.num_epochs * steps_per_epoch
    warmup_steps = config.optimizer.warmup_epochs * steps_per_epoch
    min_lr_ratio = config.optimizer.min_lr_ratio  # Never let LR drop below this

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup from min_lr_ratio to 1.0
            warmup_progress = step / max(warmup_steps, 1)
            return min_lr_ratio + (1.0 - min_lr_ratio) * warmup_progress
        if config.optimizer.use_cosine_schedule:
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            # Cosine annealing with minimum floor
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr_ratio, cosine_decay)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _load_normalization_stats(config: TrainingConfig):
    """
    Resolve normalization stats for training.

    If config.augmentation.normalize_mean/std are already set, use them.
    Otherwise, try to load dataset-specific stats from stats.json.
    Falls back to ImageNet stats if stats file is not found.
    """
    mean = config.augmentation.normalize_mean
    std = config.augmentation.normalize_std

    if mean is not None and std is not None:
        return mean, std

    # Try loading from stats.json
    stats_path = Path(config.data.training_data_dir) / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        norm = stats.get("normalization", {})
        ds_mean = norm.get("dataset_mean")
        ds_std = norm.get("dataset_std")
        if ds_mean and ds_std:
            print(f"  Using dataset normalization stats from {stats_path}")
            print(f"    mean={ds_mean}, std={ds_std}")
            return ds_mean, ds_std

    # Fall back to ImageNet
    fallback_mean = [0.485, 0.456, 0.406]
    fallback_std = [0.229, 0.224, 0.225]
    print(f"  No dataset stats found, falling back to ImageNet normalization")
    print(f"    mean={fallback_mean}, std={fallback_std}")
    return fallback_mean, fallback_std


def train(config: TrainingConfig = None):
    """Main training function."""
    if config is None:
        config = TrainingConfig()

    device = detect_device()
    print(f"Using device: {device}")

    # Seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # --- Data ---
    print("\n=== Data Setup ===")
    all_videos = get_video_names(config.data.training_data_dir)
    if not all_videos:
        raise RuntimeError(
            f"No training videos found in {config.data.training_data_dir}/images/. "
            f"Run prepare_data.py first."
        )

    if config.data.val_videos:
        # Explicit override — pin val to these videos, train on the rest.
        all_set = set(all_videos)
        requested = list(config.data.val_videos)
        missing = [v for v in requested if v not in all_set]
        if missing:
            raise ValueError(
                f"--val-videos requested {missing}, but they are not in the "
                f"dataset. Available: {sorted(all_videos)}"
            )
        val_videos = requested
        train_videos = [v for v in all_videos if v not in set(val_videos)]
        print(f"Using explicit val split (val_fraction ignored)")
    else:
        train_videos, val_videos = train_val_split(
            all_videos, val_fraction=config.data.val_fraction, seed=config.seed
        )
    print(f"Training videos ({len(train_videos)}): {train_videos}")
    print(f"Validation videos ({len(val_videos)}): {val_videos}")

    # Resolve normalization stats (dataset-specific or ImageNet fallback)
    norm_mean, norm_std = _load_normalization_stats(config)

    # When precomputing features, use validation transform for training data
    # (no pixel-level augmentation — augmentation happens at feature level via flips)
    if config.data.precompute_features:
        train_transform = VideoValidationTransform(
            image_size=config.data.image_size,
            normalize_mean=norm_mean,
            normalize_std=norm_std,
        )
        print("  Using precomputed features → train augmentation at feature level (flips only)")
    else:
        train_transform = VideoAugmentation(
            image_size=config.data.image_size,
            horizontal_flip_p=config.augmentation.horizontal_flip_p,
            vertical_flip_p=config.augmentation.vertical_flip_p,
            rotation_degrees=config.augmentation.rotation_degrees,
            shear=config.augmentation.shear,
            brightness=config.augmentation.brightness,
            contrast=config.augmentation.contrast,
            normalize_mean=norm_mean,
            normalize_std=norm_std,
        )

    val_transform = VideoValidationTransform(
        image_size=config.data.image_size,
        normalize_mean=norm_mean,
        normalize_std=norm_std,
    )

    # Datasets with performance options
    train_dataset = NeuronVideoDataset(
        data_dir=config.data.training_data_dir,
        video_names=train_videos,
        num_frames=config.data.num_frames,
        max_objects=config.data.max_objects,
        transform=train_transform,
        clip_stride=config.data.clip_stride,
        cache_in_memory=config.data.cache_in_memory,
    )
    val_dataset = NeuronVideoDataset(
        data_dir=config.data.training_data_dir,
        video_names=val_videos,
        num_frames=config.data.num_frames,
        max_objects=config.data.max_objects,
        transform=val_transform,
        clip_stride=config.data.clip_stride,
        cache_in_memory=config.data.cache_in_memory,
    )

    # When data is cached in memory, workers just add pickle overhead
    effective_num_workers = 0 if config.data.cache_in_memory else config.num_workers

    # Always pin memory on CUDA — even when frames are RAM-cached, the cached
    # tensors are unpinned, so the H2D copy blocks the GPU. Pinning the
    # collated batch is cheap and lets us use non_blocking=True at the .to()
    # call to overlap copy with GPU compute.
    use_pin_memory = (device == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=effective_num_workers,
        collate_fn=neuron_collate_fn,
        pin_memory=use_pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        collate_fn=neuron_collate_fn,
        pin_memory=use_pin_memory,
    )

    # --- Model ---
    print("\n=== Model Setup ===")
    model, loss_fn = build_training_model(config, device)

    # --- Feature Precomputation ---
    feature_cache = None
    pos_enc = None

    # AMP setup (needed for both precomputation and training)
    use_amp = config.use_amp and device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    if config.data.precompute_features:
        print("\n=== Precomputing Image Encoder Features ===")
        feature_cache, pos_enc = precompute_all_features(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            image_size=config.data.image_size,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

    # --- Optimizer & Scheduler ---
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))

    # --- AMP ---
    # No GradScaler: we use bfloat16, which has fp32 exponent range and
    # therefore no underflow risk. GradScaler is only useful for fp16.
    # Keeping it would add per-step unscale/step/update syncs for no benefit.

    # --- Training loop ---
    precompute_label = " (precomputed features)" if feature_cache else ""
    print(f"\n=== Training ({config.num_epochs} epochs, "
          f"{len(train_loader)} steps/epoch{precompute_label}) ===")
    if feature_cache:
        print("  Using precomputed features (encoder skipped during training)")

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_iou = 0.0
    epochs_without_improvement = 0
    training_log = []

    for epoch in range(config.num_epochs):
        model.train()
        # Loss tensors stay on-device — no .item() inside the step loop.
        # We only sync at logging boundaries (every log_every_n_steps and at
        # end of epoch). Each .item() is a cudaStreamSynchronize, so per-step
        # syncs were the dominant CPU stall on B200.
        epoch_losses: List[torch.Tensor] = []
        epoch_components: Dict[str, List[torch.Tensor]] = {}
        step_times = []
        epoch_start = time.time()

        for step, (batch, frame_id_lists) in enumerate(train_loader):
            step_start = time.time()
            batch = batch.to(device, non_blocking=True)

            with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_amp):
                if feature_cache is not None:
                    # Feature-level flip augmentation (replaces pixel-level augmentation)
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

                    # Forward with precomputed features (skips image encoder)
                    outputs = model(batch, precomputed_backbone_out=backbone_out)
                else:
                    # Standard forward (runs image encoder every step)
                    outputs = model(batch)

                # Compute loss
                targets = batch.masks.float()
                losses = loss_fn(outputs, targets)
                loss = losses["core_loss"]

            # Per-component loss tracking (Hypothesis A: irreducible floor).
            # Store *detached tensors* — no sync. We aggregate at log boundaries.
            for k, v in losses.items():
                if k == "core_loss":
                    continue
                if isinstance(v, torch.Tensor):
                    epoch_components.setdefault(k, []).append(v.detach())

            # Backward — bf16 doesn't need GradScaler (no underflow risk like
            # fp16). Plain backward avoids the per-step unscale/step/update
            # syncs that GradScaler.step performs internally.
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.optimizer.max_grad_norm
            )
            optimizer.step()

            scheduler.step()
            epoch_losses.append(loss.detach())
            step_times.append(time.time() - step_start)

            # Memory cleanup for MPS to prevent OOM
            del outputs, losses, loss
            if device == "mps":
                torch.mps.empty_cache()

            # Logging — one sync per log_every_n_steps instead of per step.
            if (step + 1) % config.log_every_n_steps == 0:
                recent = epoch_losses[-config.log_every_n_steps:]
                avg_loss = float(torch.stack(recent).mean().item())
                avg_step_time = np.mean(step_times[-config.log_every_n_steps:])
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch+1}/{config.num_epochs}, "
                    f"Step {step+1}/{len(train_loader)}, "
                    f"Loss: {avg_loss:.4f}, "
                    f"LR: {lr:.2e}, "
                    f"{avg_step_time:.1f}s/step"
                )

        epoch_time = time.time() - epoch_start
        # Now sync once per metric: the loss/components lists hold detached
        # GPU tensors collected during the epoch. Stack + reduce + .item()
        # gives one cudaStreamSynchronize per epoch, not per step.
        if epoch_losses:
            avg_epoch_loss = float(torch.stack(epoch_losses).mean().item())
        else:
            avg_epoch_loss = 0.0
        avg_step_time_epoch = np.mean(step_times) if step_times else 0
        train_components = {
            k: float(torch.stack(v).mean().item())
            for k, v in epoch_components.items()
            if v
        }
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_epoch_loss,
            "time_seconds": epoch_time,
            "sec_per_step": avg_step_time_epoch,
            "train_loss_components": train_components,
        }

        print(
            f"Epoch {epoch+1}/{config.num_epochs}: "
            f"train_loss={avg_epoch_loss:.4f}, "
            f"time={epoch_time:.1f}s, "
            f"{avg_step_time_epoch:.1f}s/step"
        )
        if train_components:
            comp_str = ", ".join(
                f"{k.replace('loss_','')}={v:.3f}"
                for k, v in train_components.items()
            )
            print(f"  train_loss_components: {comp_str}")

        # --- Validation ---
        if (epoch + 1) % config.val_every_n_epochs == 0 and len(val_dataset) > 0:
            print("  Validating...")
            sample_dir = checkpoint_dir / "val_samples" / f"epoch_{epoch+1:03d}"
            val_metrics = validate_epoch(
                model, val_loader, loss_fn, device,
                max_batches=config.val_max_batches,
                feature_cache=feature_cache,
                pos_enc=pos_enc,
                sample_dump_dir=sample_dir,
                sample_dump_tag=f"e{epoch+1:03d}",
            )
            log_entry.update(val_metrics)
            print(
                f"  val_loss={val_metrics['val_loss']:.4f}, "
                f"val_iou={val_metrics['val_iou']:.4f}, "
                f"val_dice={val_metrics['val_dice']:.4f}"
            )
            # Hypothesis A: per-component val loss (compare floors vs. train).
            if val_metrics.get("loss_components"):
                comp_str = ", ".join(
                    f"{k.replace('loss_','')}={v:.3f}"
                    for k, v in val_metrics["loss_components"].items()
                )
                print(f"  val_loss_components: {comp_str}")
            # Hypothesis B: chain error along the propagation.
            pf_iou = val_metrics.get("per_frame_iou", [])
            if pf_iou:
                fmt = ", ".join(
                    f"t{t}={v:.3f}" if not np.isnan(v) else f"t{t}=nan"
                    for t, v in enumerate(pf_iou)
                )
                print(f"  per_frame_iou: {fmt}")
            # Hypothesis C: variance across val videos.
            pv_iou = val_metrics.get("per_video_iou", {})
            if pv_iou:
                fmt = ", ".join(f"{n}={v:.3f}" for n, v in pv_iou.items())
                print(f"  per_video_iou: {fmt}")

            # Save best checkpoint and track early stopping
            if val_metrics["val_iou"] > best_val_iou:
                best_val_iou = val_metrics["val_iou"]
                epochs_without_improvement = 0
                save_lora_weights(model, checkpoint_dir / "lora_weights.pt")
                print(f"  New best IoU: {best_val_iou:.4f} - saved checkpoint")
            else:
                epochs_without_improvement += 1
                print(f"  No improvement for {epochs_without_improvement} validation(s)")

        training_log.append(log_entry)

        # Save training log
        with open(checkpoint_dir / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2, default=str)

        # Early stopping check
        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"No improvement in val IoU for {config.early_stopping_patience} validation checks")
            break

    # Save final checkpoint if not saving best only
    if not config.save_best_only:
        save_lora_weights(model, checkpoint_dir / "lora_weights_final.pt")

    # Save config
    import dataclasses

    with open(checkpoint_dir / "training_config.json", "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)

    print(f"\nTraining complete! Best val IoU: {best_val_iou:.4f}")
    print(f"Checkpoint saved to: {checkpoint_dir / 'lora_weights.pt'}")

    return best_val_iou
