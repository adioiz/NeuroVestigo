"""Evaluation utilities for SAM 2 fine-tuning on neuron data."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import ndimage

# Structuring elements for Betti computation. Digital topology requires a
# complementary pair (Jordan-curve consistency): 8-connectivity for the
# foreground, 4-connectivity for the background.
_CONN8 = np.ones((3, 3), dtype=int)
_CONN4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


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


def compute_betti_numbers(mask) -> Tuple[int, int]:
    """Betti numbers (β0, β1) of a 2-D binary mask.

    β0 = number of connected components (distinct objects).
    β1 = number of holes (enclosed background regions).

    For neurons these are the topology quantities IoU misses: a severed
    branch raises β0 by 1 while barely moving IoU; a spurious enclosed gap
    raises β1. Foreground uses 8-connectivity and background 4-connectivity
    so the counts satisfy the Jordan-curve relation.

    Args:
        mask: 2-D array-like (torch or numpy), truthy = foreground.

    Returns:
        (beta0, beta1) as ints.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"compute_betti_numbers expects a 2-D mask, got {m.ndim}-D")

    _, beta0 = ndimage.label(m, structure=_CONN8)          # foreground components
    labels_bg, n_bg = ndimage.label(~m, structure=_CONN4)  # background components
    if n_bg == 0:  # mask fills the frame — no background, no holes
        return int(beta0), 0
    # Background components touching the image border are "outside"; the rest
    # are enclosed holes.
    border = np.concatenate(
        [labels_bg[0, :], labels_bg[-1, :], labels_bg[:, 0], labels_bg[:, -1]]
    )
    outer = set(np.unique(border)) - {0}
    return int(beta0), int(n_bg - len(outer))


def compute_betti_error(pred, target) -> Tuple[int, int]:
    """Absolute Betti-number errors (|Δβ0|, |Δβ1|) between pred and target.

    Lower is better; 0 means the prediction matches the ground-truth topology
    (same number of components and holes).
    """
    p0, p1 = compute_betti_numbers(pred)
    t0, t1 = compute_betti_numbers(target)
    return abs(p0 - t0), abs(p1 - t1)


def evaluate_batch_per_frame(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Evaluate a batch and return per-(batch_item, frame) IoU/Dice lists.

    Returns:
        ious[b][t]: IoU averaged over objects for batch item b at frame t.
                    Empty entries are omitted (frames where no object exists).
        dices[b][t]: same for Dice.
    """
    aligned = _align_pred_target(outputs, targets)
    if aligned is None:
        return [], []
    pred_binary, targets_resized = aligned

    B, T, N_obj = pred_binary.shape[:3]
    ious_per_bt: List[List[float]] = [[] for _ in range(B)]
    dices_per_bt: List[List[float]] = [[] for _ in range(B)]

    for b in range(B):
        for t in range(T):
            obj_ious, obj_dices = [], []
            for obj in range(N_obj):
                pred = pred_binary[b, t, obj]
                gt = targets_resized[b, t, obj]
                if not pred.any() and not gt.any():
                    continue
                obj_ious.append(compute_iou(pred, gt))
                obj_dices.append(compute_dice(pred, gt))
            # Use NaN sentinel for frames with no valid objects so frame index
            # alignment is preserved across the loop.
            ious_per_bt[b].append(float(np.mean(obj_ious)) if obj_ious else float("nan"))
            dices_per_bt[b].append(float(np.mean(obj_dices)) if obj_dices else float("nan"))

    return ious_per_bt, dices_per_bt


def _align_pred_target(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Binarise predictions and match the target resolution to them.

    Returns (pred_binary, targets_resized), both [B, T, N_obj, H, W] bool, or
    None when the model produced no masks. Shared by the IoU/Dice and Betti
    per-frame evaluators so both score identical (pred, gt) pairs.
    """
    if "pred_masks" not in outputs:
        return None

    pred_binary = outputs["pred_masks"] > 0  # [B, T, N_obj, H, W]
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
    return pred_binary, targets_resized


def betti_error_per_frame(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Per-(batch_item, frame) Betti-0 and Betti-1 errors vs. ground truth.

    Mirrors ``evaluate_batch_per_frame``: object errors are averaged per frame,
    and frames with no object present get a NaN sentinel so frame-index
    alignment is preserved. Betti-0 error counts wrong number of components
    (breaks/merges); Betti-1 error counts wrong number of holes.
    """
    aligned = _align_pred_target(outputs, targets)
    if aligned is None:
        return [], []
    pred_binary, targets_resized = aligned

    B, T, N_obj = pred_binary.shape[:3]
    b0_per_bt: List[List[float]] = [[] for _ in range(B)]
    b1_per_bt: List[List[float]] = [[] for _ in range(B)]

    for b in range(B):
        for t in range(T):
            obj_b0, obj_b1 = [], []
            for obj in range(N_obj):
                pred = pred_binary[b, t, obj]
                gt = targets_resized[b, t, obj]
                if not pred.any() and not gt.any():
                    continue
                e0, e1 = compute_betti_error(pred, gt)
                obj_b0.append(float(e0))
                obj_b1.append(float(e1))
            b0_per_bt[b].append(float(np.mean(obj_b0)) if obj_b0 else float("nan"))
            b1_per_bt[b].append(float(np.mean(obj_b1)) if obj_b1 else float("nan"))

    return b0_per_bt, b1_per_bt


def evaluate_batch(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
) -> Dict[str, float]:
    """Aggregate IoU/Dice over a batch (legacy interface)."""
    ious_per_bt, dices_per_bt = evaluate_batch_per_frame(outputs, targets)
    flat_ious = [v for row in ious_per_bt for v in row if not np.isnan(v)]
    flat_dices = [v for row in dices_per_bt for v in row if not np.isnan(v)]
    if not flat_ious:
        return {"mean_iou": 0.0, "mean_dice": 0.0}
    return {
        "mean_iou": float(np.mean(flat_ious)),
        "mean_dice": float(np.mean(flat_dices)),
    }


def _build_sample_record(
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
    images: torch.Tensor,
    b: int,
    t: int,
    obj_idx: int,
    iou: float,
    video_name: str,
    frame_idx_in_video: Optional[int],
) -> Dict:
    """Pull out CPU tensors for one (b, t, obj) slice so they can be saved later.

    Held in a buffer while validation runs; cheaper than keeping the whole
    batch around. We resize the prediction to GT resolution here so the
    saved PNGs are aligned.
    """
    H_gt, W_gt = target_masks.shape[-2:]
    pred_slice = pred_masks[b, t, obj_idx].float().unsqueeze(0).unsqueeze(0)
    pred_slice = torch.nn.functional.interpolate(
        pred_slice, size=(H_gt, W_gt), mode="bilinear", align_corners=False,
    )[0, 0]

    return {
        "iou": iou,
        "video": video_name,
        "t": t,
        "frame_idx": frame_idx_in_video,
        "obj": obj_idx,
        "image": images[b, t].detach().float().cpu().clone(),
        "gt": target_masks[b, t, obj_idx].bool().cpu().clone(),
        "pred": pred_slice.detach().cpu().clone(),
    }


def save_selected_samples(
    samples: List[Dict],
    out_dir: Path,
    tag: str,
):
    """Save one PNG per record in `samples`. Each PNG is image | GT | pred | overlay.

    Filenames embed video name and IoU so the worst cases are easy to spot.
    """
    if not samples:
        return
    try:
        from PIL import Image
    except ImportError:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for rank, s in enumerate(samples):
        gt = s["gt"].numpy()
        pred = (s["pred"] > 0).numpy()
        img = s["image"].numpy()

        img_min = img.min()
        img_max = img.max()
        img_norm = (img - img_min) / max(float(img_max - img_min), 1e-6)
        img_uint8 = (img_norm * 255).clip(0, 255).astype(np.uint8)
        if img_uint8.shape[0] == 1:
            rgb = np.repeat(img_uint8, 3, axis=0).transpose(1, 2, 0)
        else:
            rgb = img_uint8[:3].transpose(1, 2, 0)

        gt_rgb = (np.stack([gt, gt, gt], axis=-1) * 255).astype(np.uint8)
        pr_rgb = (np.stack([pred, pred, pred], axis=-1) * 255).astype(np.uint8)
        overlay = rgb.copy()
        overlay[..., 0] = np.maximum(overlay[..., 0], pred * 255)  # red = pred
        overlay[..., 1] = np.maximum(overlay[..., 1], gt * 255)    # green = GT
        composite = np.concatenate([rgb, gt_rgb, pr_rgb, overlay], axis=1)

        frame_id = s["frame_idx"] if s["frame_idx"] is not None else s["t"]
        fname = (
            f"{tag}_rank{rank:02d}_iou{s['iou']:.2f}_"
            f"{s['video']}_t{s['t']}_f{frame_id}.png"
        )
        Image.fromarray(composite).save(out_dir / fname)


@torch.no_grad()
def validate_epoch(
    model,
    val_loader,
    loss_fn,
    device: str,
    max_batches: int = 0,
    feature_cache: Optional[Dict[Tuple[str, int], List[torch.Tensor]]] = None,
    pos_enc: Optional[List[torch.Tensor]] = None,
    sample_dump_dir: Optional[Path] = None,
    sample_dump_tag: str = "val",
) -> Dict[str, float]:
    """
    Run validation on the validation set.

    Args:
        model: SAM2TrainingWrapper model.
        val_loader: Validation DataLoader (yields (batch, frame_id_lists) tuples).
        loss_fn: MultiStepMultiMasksAndIous loss function.
        device: Device string.
        max_batches: Maximum batches to validate (0 = all batches for full validation).
        feature_cache: Optional precomputed feature cache.
        pos_enc: Optional positional encodings (required if feature_cache is set).
        sample_dump_dir: If set, save side-by-side PNGs from the first batch here.
        sample_dump_tag: Filename prefix for saved samples.

    Returns:
        Dict with 'val_loss', 'val_iou', 'val_dice' (legacy aggregates) plus:
        'per_frame_iou'    list[T] — mean IoU at each frame index across val set
        'per_frame_dice'   list[T] — mean Dice at each frame index across val set
        'per_video_iou'    dict[video_name -> mean IoU]
        'loss_components'  dict[component_name -> mean value]
    """
    model.eval()

    total_loss = 0.0
    component_sums: Dict[str, float] = defaultdict(float)
    component_count = 0

    per_frame_ious: Dict[int, List[float]] = defaultdict(list)
    per_frame_dices: Dict[int, List[float]] = defaultdict(list)
    per_video_ious: Dict[str, List[float]] = defaultdict(list)
    flat_ious: List[float] = []
    flat_dices: List[float] = []
    per_frame_betti0: Dict[int, List[float]] = defaultdict(list)
    per_frame_betti1: Dict[int, List[float]] = defaultdict(list)
    flat_betti0: List[float] = []
    flat_betti1: List[float] = []

    # Buffer of the K worst (lowest-IoU) frames seen so far across the val set.
    # We keep a strict cap of `worst_buffer_cap` records — when full, only
    # frames that beat the current max IoU make it in. This keeps memory
    # bounded regardless of val set size.
    worst_buffer: List[Dict] = []
    worst_buffer_cap = 6 if sample_dump_dir is not None else 0

    num_batches = 0

    for batch, frame_id_lists in val_loader:
        if max_batches > 0 and num_batches >= max_batches:
            break
        batch = batch.to(device, non_blocking=True)

        if feature_cache is not None and pos_enc is not None:
            from src.trainer import assemble_cached_backbone
            backbone_out = assemble_cached_backbone(
                feature_cache, pos_enc, frame_id_lists, device,
                do_hflip=False, do_vflip=False,
            )
            outputs = model(batch, precomputed_backbone_out=backbone_out)
        else:
            outputs = model(batch)

        targets = batch.masks.float()
        losses = loss_fn(outputs, targets)
        total_loss += losses["core_loss"].item()
        for k, v in losses.items():
            if k == "core_loss":
                continue
            if isinstance(v, torch.Tensor):
                component_sums[k] += float(v.item())
        component_count += 1

        ious_per_bt, dices_per_bt = evaluate_batch_per_frame(outputs, batch.masks)
        # Topology metric — shares the same alignment, so its NaN pattern
        # matches ious_per_bt and (b, t) indices line up.
        betti0_per_bt, betti1_per_bt = betti_error_per_frame(outputs, batch.masks)
        for b, (ious, dices) in enumerate(zip(ious_per_bt, dices_per_bt)):
            video_name = frame_id_lists[b][0][0] if frame_id_lists[b] else "?"
            for t, (iou, dice) in enumerate(zip(ious, dices)):
                if np.isnan(iou):
                    continue
                per_frame_ious[t].append(iou)
                per_frame_dices[t].append(dice)
                per_video_ious[video_name].append(iou)
                flat_ious.append(iou)
                flat_dices.append(dice)
                if betti0_per_bt and not np.isnan(betti0_per_bt[b][t]):
                    per_frame_betti0[t].append(betti0_per_bt[b][t])
                    per_frame_betti1[t].append(betti1_per_bt[b][t])
                    flat_betti0.append(betti0_per_bt[b][t])
                    flat_betti1.append(betti1_per_bt[b][t])

                # Worst-IoU sample tracking. Skip t==0: it always uses the GT
                # mask as the prompt so the prediction is trivially perfect.
                if worst_buffer_cap > 0 and t > 0:
                    current_max = max((s["iou"] for s in worst_buffer), default=1.0)
                    if (
                        len(worst_buffer) < worst_buffer_cap
                        or iou < current_max
                    ):
                        frame_idx_in_video = (
                            frame_id_lists[b][t][1]
                            if t < len(frame_id_lists[b])
                            else None
                        )
                        record = _build_sample_record(
                            outputs["pred_masks"], batch.masks, batch.images,
                            b=b, t=t, obj_idx=0,
                            iou=iou, video_name=video_name,
                            frame_idx_in_video=frame_idx_in_video,
                        )
                        worst_buffer.append(record)
                        if len(worst_buffer) > worst_buffer_cap:
                            # Drop the highest-IoU record (we want the worst).
                            worst_buffer.sort(key=lambda s: s["iou"])
                            worst_buffer.pop()

        num_batches += 1

    model.train()

    # Save the worst-IoU samples now that we've seen the full val set.
    if sample_dump_dir is not None and worst_buffer:
        try:
            worst_buffer.sort(key=lambda s: s["iou"])
            save_selected_samples(worst_buffer, sample_dump_dir, sample_dump_tag)
        except Exception as e:
            print(f"  [warn] sample dump failed: {e}")

    if num_batches == 0:
        return {"val_loss": float("inf"), "val_iou": 0.0, "val_dice": 0.0}

    T_max = max(per_frame_ious) + 1 if per_frame_ious else 0
    per_frame_iou = [
        float(np.mean(per_frame_ious[t])) if per_frame_ious.get(t) else float("nan")
        for t in range(T_max)
    ]
    per_frame_dice = [
        float(np.mean(per_frame_dices[t])) if per_frame_dices.get(t) else float("nan")
        for t in range(T_max)
    ]
    per_video_iou = {
        name: float(np.mean(vals)) for name, vals in per_video_ious.items()
    }
    per_frame_betti0_mean = [
        float(np.mean(per_frame_betti0[t])) if per_frame_betti0.get(t) else float("nan")
        for t in range(T_max)
    ]
    per_frame_betti1_mean = [
        float(np.mean(per_frame_betti1[t])) if per_frame_betti1.get(t) else float("nan")
        for t in range(T_max)
    ]
    loss_components = {
        k: v / component_count for k, v in component_sums.items()
    }

    return {
        "val_loss": total_loss / num_batches,
        "val_iou": float(np.mean(flat_ious)) if flat_ious else 0.0,
        "val_dice": float(np.mean(flat_dices)) if flat_dices else 0.0,
        # Topology: mean absolute Betti-number error vs. GT (lower = better).
        # val_betti0 = component-count errors (breaks/merges); val_betti1 = hole
        # errors. These capture connectivity that IoU/Dice are blind to.
        "val_betti0": float(np.mean(flat_betti0)) if flat_betti0 else float("nan"),
        "val_betti1": float(np.mean(flat_betti1)) if flat_betti1 else float("nan"),
        "per_frame_iou": per_frame_iou,
        "per_frame_dice": per_frame_dice,
        "per_frame_betti0": per_frame_betti0_mean,
        "per_frame_betti1": per_frame_betti1_mean,
        "per_video_iou": per_video_iou,
        "loss_components": loss_components,
    }
