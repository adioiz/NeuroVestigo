"""
Standalone training script for SAM 2 LoRA fine-tuning on neuron data.

This bypasses Meta's complex distributed Trainer and uses a simple PyTorch
training loop suitable for single-GPU training.

Key differences from Meta's internal training:
- Uses standard build_sam2() instead of SAM2Train (which isn't in pip package)
- Custom forward pass that handles video frames and mask propagation
- Local loss function implementation (loss_fns.py)
"""

import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sam2.build_sam import build_sam2, _hf_download

from src.training.config import TrainingConfig
from src.training.lora import (
    apply_lora,
    get_trainable_parameters,
    save_lora_weights,
    SAM2_LORA_TARGETS,
)
from src.training.neuron_dataset import (
    NeuronVideoDataset,
    neuron_collate_fn,
    get_video_names,
    train_val_split,
)
from src.training.augmentations import VideoAugmentation, VideoValidationTransform
from src.training.evaluate import validate_epoch
from src.training.loss_fns import MultiStepMultiMasksAndIous
from src.training.data_utils import BatchedVideoDatapoint


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SAM2TrainingWrapper(nn.Module):
    """Wrapper that adapts SAM2Base for training with video clips.

    SAM2 models from the pip package don't have a training-specific forward.
    This wrapper:
    1. Takes batched video data (images + masks)
    2. Processes frames through the image encoder
    3. Simulates mask propagation using the memory mechanism
    4. Returns predicted masks for loss computation
    """

    def __init__(self, sam2_model: nn.Module, num_frames: int = 8):
        super().__init__()
        self.model = sam2_model
        self.num_frames = num_frames

    def forward(self, batch: BatchedVideoDatapoint) -> Dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            batch: BatchedVideoDatapoint with:
                - images: [B, T, C, H, W] video frames
                - masks: [B, T, N_obj, H, W] ground truth masks

        Returns:
            Dict with:
                - pred_masks: [B, T, N_obj, H, W] predicted mask logits
        """
        images = batch.images  # [B, T, C, H, W]
        gt_masks = batch.masks  # [B, T, N_obj, H, W]

        B, T, C, H, W = images.shape
        N_obj = gt_masks.shape[2]

        # Encode all frames through image encoder
        # Reshape to [B*T, C, H, W] for batch processing
        images_flat = images.view(B * T, C, H, W)

        # Get image embeddings
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
        use_cldice=config.loss.use_cldice,
        cldice_num_iter=config.loss.cldice_num_iter,
        use_supervoxel=config.loss.use_supervoxel,
    )
    if config.loss.use_cldice:
        print(f"  Using clDice loss for skeleton topology (weight={config.loss.loss_cldice_weight})")
    if config.loss.use_supervoxel:
        print(f"  Using supervoxel loss for connectivity (weight={config.loss.loss_supervoxel_weight})")

    training_model = training_model.to(device)
    return training_model, loss_fn


def build_optimizer(model: nn.Module, config: TrainingConfig):
    """Build AdamW optimizer with separate LR for LoRA and other params."""
    param_groups = get_trainable_parameters(model)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": param_groups["lora"],
                "lr": config.optimizer.lora_lr,
                "weight_decay": config.optimizer.weight_decay,
            },
            {
                "params": param_groups["other"],
                "lr": config.optimizer.base_lr,
                "weight_decay": 0.0,  # No weight decay for embeddings
            },
        ],
    )
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
    Otherwise, try to load dataset-specific stats from manifest.json.
    Falls back to ImageNet stats if manifest is not found.
    """
    mean = config.augmentation.normalize_mean
    std = config.augmentation.normalize_std

    if mean is not None and std is not None:
        return mean, std

    # Try loading from manifest
    manifest_path = Path(config.data.training_data_dir) / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        norm = manifest.get("normalization", {})
        ds_mean = norm.get("dataset_mean")
        ds_std = norm.get("dataset_std")
        if ds_mean and ds_std:
            print(f"  Using dataset normalization stats from {manifest_path}")
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

    train_videos, val_videos = train_val_split(
        all_videos, val_fraction=config.data.val_fraction, seed=config.seed
    )
    print(f"Training videos ({len(train_videos)}): {train_videos}")
    print(f"Validation videos ({len(val_videos)}): {val_videos}")

    # Resolve normalization stats (dataset-specific or ImageNet fallback)
    norm_mean, norm_std = _load_normalization_stats(config)

    # Augmentations
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

    # Datasets (label_id removed - always uses all labels)
    train_dataset = NeuronVideoDataset(
        data_dir=config.data.training_data_dir,
        video_names=train_videos,
        num_frames=config.data.num_frames,
        max_objects=config.data.max_objects,
        transform=train_transform,
    )
    val_dataset = NeuronVideoDataset(
        data_dir=config.data.training_data_dir,
        video_names=val_videos,
        num_frames=config.data.num_frames,
        max_objects=config.data.max_objects,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=neuron_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=neuron_collate_fn,
        pin_memory=True,
    )

    # --- Model ---
    print("\n=== Model Setup ===")
    model, loss_fn = build_training_model(config, device)

    # --- Optimizer & Scheduler ---
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))

    # --- AMP ---
    use_amp = config.use_amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # --- Training loop ---
    print(f"\n=== Training ({config.num_epochs} epochs) ===")
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_iou = 0.0
    epochs_without_improvement = 0
    training_log = []

    for epoch in range(config.num_epochs):
        model.train()
        epoch_losses = []
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            batch = batch.to(device)

            with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_amp):
                # Forward
                outputs = model(batch)

                # Compute loss
                targets = batch.masks.float()
                losses = loss_fn(outputs, targets)
                loss = losses["core_loss"]

            # Backward
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optimizer.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optimizer.max_grad_norm
                )
                optimizer.step()

            scheduler.step()
            epoch_losses.append(loss.item())

            # Memory cleanup for MPS to prevent OOM
            del outputs, losses, loss
            if device == "mps":
                torch.mps.empty_cache()

            # Logging
            if (step + 1) % config.log_every_n_steps == 0:
                avg_loss = np.mean(epoch_losses[-config.log_every_n_steps :])
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch+1}/{config.num_epochs}, "
                    f"Step {step+1}/{len(train_loader)}, "
                    f"Loss: {avg_loss:.4f}, "
                    f"LR: {lr:.2e}"
                )

        epoch_time = time.time() - epoch_start
        avg_epoch_loss = np.mean(epoch_losses)
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_epoch_loss,
            "time_seconds": epoch_time,
        }

        print(
            f"Epoch {epoch+1}/{config.num_epochs}: "
            f"train_loss={avg_epoch_loss:.4f}, "
            f"time={epoch_time:.1f}s"
        )

        # --- Validation ---
        if (epoch + 1) % config.val_every_n_epochs == 0 and len(val_dataset) > 0:
            print("  Validating...")
            val_metrics = validate_epoch(
                model, val_loader, loss_fn, device,
                max_batches=config.val_max_batches
            )
            log_entry.update(val_metrics)
            print(
                f"  val_loss={val_metrics['val_loss']:.4f}, "
                f"val_iou={val_metrics['val_iou']:.4f}, "
                f"val_dice={val_metrics['val_dice']:.4f}"
            )

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
