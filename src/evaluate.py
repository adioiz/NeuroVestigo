"""Evaluation utilities for SAM 2 fine-tuning on neuron data."""

from typing import Dict, List, Optional, Tuple

import torch
import numpy as np


def compute_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute IoU between two binary masks.

    Args:
        pred: Predicted mask (any shape, will be flattened).
        target: Ground truth mask (same shape as pred).

    Returns:
        IoU score (float). Returns 1.0 if both masks are empty, 0.0 if only one is empty.
    """
    pred = pred.bool().flatten()
    target = target.bool().flatten()
    intersection = (pred & target).sum().float()
    union = (pred | target).sum().float()

    if union == 0:
        return 1.0  # Both masks are empty - perfect match
    return (intersection / union).item()


def compute_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Dice coefficient between two binary masks.

    Args:
        pred: Predicted mask.
        target: Ground truth mask.

    Returns:
        Dice score (float). Returns 1.0 if both masks are empty.
    """
    pred = pred.bool().flatten()
    target = target.bool().flatten()
    intersection = (pred & target).sum().float()
    total = pred.sum().float() + target.sum().float()

    if total == 0:
        return 1.0  # Both masks are empty - perfect match
    return (2 * intersection / total).item()


def evaluate_batch(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Evaluate a batch of SAM2TrainingWrapper outputs against ground truth.

    Args:
        outputs: Dict with 'pred_masks' of shape [B, T, N_obj, H, W].
        targets: Ground truth masks [B, T, N_obj, H, W] (bool).

    Returns:
        Dict with 'mean_iou' and 'mean_dice' averaged over all frames and objects.
    """
    all_ious = []
    all_dices = []

    if "pred_masks" not in outputs:
        return {"mean_iou": 0.0, "mean_dice": 0.0}

    pred_masks = outputs["pred_masks"]  # [B, T, N_obj, H, W]
    pred_binary = (pred_masks > 0)  # Threshold at 0 for logits

    # Resize GT to match pred if needed
    if pred_binary.shape[-2:] != targets.shape[-2:]:
        B, T, N_obj, H_gt, W_gt = targets.shape
        _, _, _, H_pred, W_pred = pred_binary.shape
        targets_resized = torch.nn.functional.interpolate(
            targets.float().view(B * T * N_obj, 1, H_gt, W_gt),
            size=(H_pred, W_pred),
            mode="nearest",
        ).view(B, T, N_obj, H_pred, W_pred).bool()
    else:
        targets_resized = targets.bool()

    # Compute metrics for each sample, frame, and object
    B, T, N_obj = pred_binary.shape[:3]
    for b in range(B):
        for t in range(T):
            for obj in range(N_obj):
                pred = pred_binary[b, t, obj]
                gt = targets_resized[b, t, obj]

                # Skip if both are empty
                if not pred.any() and not gt.any():
                    continue

                iou = compute_iou(pred, gt)
                dice = compute_dice(pred, gt)

                # Handle NaN from empty masks
                if not torch.isnan(torch.tensor(iou)):
                    all_ious.append(iou)
                if not torch.isnan(torch.tensor(dice)):
                    all_dices.append(dice)

    if not all_ious:
        return {"mean_iou": 0.0, "mean_dice": 0.0}

    return {
        "mean_iou": float(np.mean(all_ious)),
        "mean_dice": float(np.mean(all_dices)),
    }


@torch.no_grad()
def validate_epoch(
    model,
    val_loader,
    loss_fn,
    device: str,
    max_batches: int = 0,
    feature_cache: Optional[Dict[Tuple[str, int], List[torch.Tensor]]] = None,
    pos_enc: Optional[List[torch.Tensor]] = None,
) -> Dict[str, float]:
    """
    Run validation on the validation set.

    Args:
        model: SAM2TrainingWrapper model.
        val_loader: Validation DataLoader (yields (batch, frame_id_lists) tuples).
        loss_fn: MultiStepMultiMasksAndIous loss function.
        device: Device string.
        max_batches: Maximum batches to validate (0 = all batches for full validation).
        feature_cache: Optional precomputed feature cache. If provided, uses cached
                       backbone features instead of running the image encoder.
        pos_enc: Optional positional encodings (required if feature_cache is set).

    Returns:
        Dict with 'val_loss', 'val_iou', 'val_dice'.
    """
    model.eval()

    total_loss = 0.0
    all_ious = []
    all_dices = []
    num_batches = 0

    for batch, frame_id_lists in val_loader:
        # Early exit if max_batches limit reached
        if max_batches > 0 and num_batches >= max_batches:
            break
        batch = batch.to(device)

        # Use precomputed features if available (no flip for validation)
        if feature_cache is not None and pos_enc is not None:
            from src.trainer import assemble_cached_backbone
            backbone_out = assemble_cached_backbone(
                feature_cache, pos_enc, frame_id_lists, device,
                do_hflip=False, do_vflip=False,
            )
            outputs = model(batch, precomputed_backbone_out=backbone_out)
        else:
            outputs = model(batch)

        # Compute loss
        targets = batch.masks.float()  # [B, T, N_obj, H, W]
        losses = loss_fn(outputs, targets)
        total_loss += losses["core_loss"].item()

        # Compute metrics - targets should be [B, T, N_obj, H, W]
        metrics = evaluate_batch(outputs, batch.masks)
        all_ious.append(metrics["mean_iou"])
        all_dices.append(metrics["mean_dice"])
        num_batches += 1

    model.train()

    if num_batches == 0:
        return {"val_loss": float("inf"), "val_iou": 0.0, "val_dice": 0.0}

    return {
        "val_loss": total_loss / num_batches,
        "val_iou": float(np.mean(all_ious)),
        "val_dice": float(np.mean(all_dices)),
    }
