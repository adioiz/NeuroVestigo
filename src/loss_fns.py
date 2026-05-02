"""Loss functions for SAM 2 fine-tuning.

This module implements the loss functions needed for training SAM 2,
replacing the Meta-internal training.loss_fns module.

Loss Components:
    - FocalLoss: Handles class imbalance (background >> neuron pixels)
    - DiceLoss: Optimizes for shape/overlap (IoU-like)
    - SoftCBDiceLoss: Topology-preserving loss with boundary weighting (neurons)
    - SupervoxelLoss2D: Connectivity-preserving loss for split/merge errors
    - MultiStepMultiMasksAndIous: Combines all losses with configurable weights

References:
    - cbDice: Shi et al., "centerline boundary Dice loss for vascular
      segmentation", MICCAI 2024 (extends clDice with distance-transform
      boundary weighting)
      https://github.com/PengchengShi1220/cbDice
    - clDice (predecessor): Shit et al., "clDice - A Novel Topology-Preserving
      Loss Function for Tubular Structure Segmentation", CVPR 2021
      https://arxiv.org/abs/2003.07311
    - Supervoxel Loss: Grim et al., "Efficient Connectivity-Preserving Instance
      Segmentation with Supervoxel-Based Loss Function", AAAI 2025
      https://arxiv.org/abs/2501.01022
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label as ndimage_label
from scipy.ndimage import binary_dilation as ndi_binary_dilation


class FocalLoss(nn.Module):
    """Focal Loss for handling severe class imbalance.

    In microscopy segmentation, background pixels vastly outnumber neuron pixels
    (often 99% vs 1%). Standard cross-entropy would be dominated by the easy
    background pixels. Focal loss down-weights easy examples.

    Formula:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Where:
        - p_t = p if y=1, else (1-p): probability of correct class
        - α: balances positive/negative classes (default 0.25 for positives)
        - γ: focusing parameter, reduces loss for well-classified examples

    Args:
        alpha: Weight for positive class (neurons). Default 0.25.
        gamma: Focusing parameter. Higher = more focus on hard examples. Default 2.0.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            pred: Predicted logits, any shape (will be flattened).
            target: Binary ground truth, same shape as pred.

        Returns:
            Scalar focal loss value.
        """
        # Sigmoid to get probabilities
        pred_sigmoid = torch.sigmoid(pred)

        # Standard binary cross-entropy (unreduced)
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # p_t: probability of being correct
        # If target=1: p_t = pred_sigmoid (want this high)
        # If target=0: p_t = 1 - pred_sigmoid (want this high)
        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)

        # Focal weight: (1 - p_t)^gamma
        # Easy examples (p_t close to 1) get weight close to 0
        # Hard examples (p_t close to 0) get weight close to 1
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting for class balance
        # alpha for positive class, (1-alpha) for negative
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        # Combine: alpha * focal_weight * cross_entropy
        focal_loss = alpha_t * focal_weight * ce_loss

        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for shape/overlap optimization.

    Dice coefficient measures overlap between prediction and ground truth.
    Unlike pixel-wise losses, it treats the mask as a whole, which helps
    with class imbalance and encourages connected predictions.

    Formula:
        Dice = 2 * |A ∩ B| / (|A| + |B|)
        DiceLoss = 1 - Dice

    The smooth parameter prevents division by zero when both masks are empty.

    Args:
        smooth: Smoothing factor to prevent division by zero. Default 1.0.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute dice loss.

        Args:
            pred: Predicted logits, any shape.
            target: Binary ground truth, same shape as pred.

        Returns:
            Scalar dice loss value (1 - dice_coefficient).
        """
        # Flatten for global computation
        pred_sigmoid = torch.sigmoid(pred).view(-1)
        target_flat = target.view(-1)

        # Soft intersection: element-wise product
        intersection = (pred_sigmoid * target_flat).sum()

        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (
            pred_sigmoid.sum() + target_flat.sum() + self.smooth
        )

        # Return 1 - dice (so minimizing loss maximizes dice)
        return 1 - dice


# =============================================================================
# cbDice Loss - Topology-Preserving Loss with Boundary Weighting
# =============================================================================
# cbDice (Shi et al., MICCAI 2024) extends clDice (Shit et al., CVPR 2021)
# with distance-transform-based boundary weighting. Thinner structures get
# higher weight via inverse-radius weighting from the EDT.
#
# Morphological helpers (soft_erode, soft_dilate, soft_open, soft_skel) are
# shared between clDice and cbDice.
#
# References:
#   cbDice: https://github.com/PengchengShi1220/cbDice
#   clDice: https://github.com/jocpae/clDice
# =============================================================================


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Soft morphological erosion using min-pooling.

    Erosion shrinks bright regions. For binary masks, this removes thin
    protrusions and shrinks the mask boundary inward.

    Uses negative max-pooling as a proxy for min-pooling (more efficient).
    Applies 1D pooling in each dimension separately for better results.

    Args:
        img: Input tensor [B, C, H, W] with values in [0, 1]

    Returns:
        Eroded tensor, same shape as input.
    """
    if len(img.shape) == 4:
        # 2D case: apply 1D erosion in each direction
        p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        return torch.min(p1, p2)
    elif len(img.shape) == 5:
        # 3D case
        p1 = -F.max_pool3d(-img, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        p2 = -F.max_pool3d(-img, kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0))
        p3 = -F.max_pool3d(-img, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {len(img.shape)}D")


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """Soft morphological dilation using max-pooling.

    Dilation expands bright regions. For binary masks, this fills small
    holes and expands the mask boundary outward.

    Args:
        img: Input tensor [B, C, H, W] with values in [0, 1]

    Returns:
        Dilated tensor, same shape as input.
    """
    if len(img.shape) == 4:
        return F.max_pool2d(img, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {len(img.shape)}D")


def soft_open(img: torch.Tensor) -> torch.Tensor:
    """Soft morphological opening: erosion followed by dilation.

    Opening removes thin bright regions (noise, thin connections)
    while preserving the overall shape of larger regions.

    Args:
        img: Input tensor [B, C, H, W] with values in [0, 1]

    Returns:
        Opened tensor, same shape as input.
    """
    return soft_dilate(soft_erode(img))


def soft_skel(img: torch.Tensor, num_iter: int = 5) -> torch.Tensor:
    """Compute soft skeleton through iterative morphological operations.

    The skeleton (or medial axis) is the set of points equidistant from
    the boundary. For tubular structures like neurons, the skeleton
    represents the centerline.

    Algorithm:
        1. Iteratively erode the image
        2. At each iteration, compute: eroded - opened(eroded)
        3. This gives points that would be removed by opening (thin parts)
        4. Accumulate these points across iterations

    Args:
        img: Input tensor [B, C, H, W] with values in [0, 1]
        num_iter: Number of erosion iterations. More = thinner skeleton.
                  Default 5 works well for most neuron images (reduced from 10
                  for better memory efficiency on MPS).

    Returns:
        Soft skeleton tensor, same shape as input.
    """
    img_eroded = img.clone()
    skeleton = torch.zeros_like(img)

    for _ in range(num_iter):
        img_eroded = soft_erode(img_eroded)
        img_opened = soft_open(img_eroded)
        # Points removed by opening = thin parts = skeleton candidates
        delta = torch.clamp(img_eroded - img_opened, min=0)
        # Use in-place operations to reduce memory
        skeleton.add_(delta).clamp_(max=1)
        # Clear intermediate to free memory
        del delta, img_opened

    return skeleton


# Module-level thread pool for parallel distance-transform computation.
# scipy.ndimage.distance_transform_edt is a C extension that releases the
# GIL, so threads (not processes) actually parallelize. Created once and
# reused across cbDice forward passes — submission overhead is ~tens of µs.
from concurrent.futures import ThreadPoolExecutor
_DT_EXECUTOR = ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="cbdice-dt",
)


def _compute_distance_weights(binary_mask: torch.Tensor) -> torch.Tensor:
    """Compute inverse-radius distance weights for cbDice.

    Uses Euclidean distance transform (EDT) to compute weights that give
    higher importance to thinner structures (smaller local radius). This is
    the key addition of cbDice over clDice.

    For a neuron branch with radius r, the EDT value at the centerline is r.
    We invert this: weight = 1/(r+1), so thin branches (r≈1) get weight ≈0.5
    while thick soma regions (r≈10) get weight ≈0.09.

    Implementation note: at B*T*N=16 with 1-2 ms per scipy call, the serial
    loop was ~30 ms of CPU stall per cbDice forward. Parallelising the per-
    sample calls across 4 threads (scipy releases the GIL) cuts this to
    ~8-10 ms while producing bit-identical output (same algorithm, same
    arguments, deterministic write order on the way out).

    Args:
        binary_mask: [B, 1, H, W] binary mask (values 0 or 1)

    Returns:
        weights: [B, 1, H, W] distance-based weights (0 for background)
    """
    from scipy.ndimage import distance_transform_edt

    mask_np = binary_mask.detach().cpu().numpy()
    B, C, H, W = mask_np.shape
    weights = np.zeros_like(mask_np, dtype=np.float32)

    def _dt_one(b: int, c: int):
        m = mask_np[b, c]
        if m.sum() == 0:
            return b, c, None
        dt = distance_transform_edt(m)
        # Inverse radius weighting: thinner structures get higher weight.
        # Add 1 to avoid division by zero, then invert.
        dt_inv = 1.0 / (dt + 1.0)
        # Normalize so weights sum to number of foreground pixels.
        fg_count = m.sum()
        dt_inv = dt_inv * fg_count / (dt_inv[m > 0].sum() + 1e-8)
        return b, c, dt_inv * m  # Zero out background

    # Submit all (b, c) tasks; collect in a deterministic order so the final
    # `weights` tensor is identical across runs.
    futures = [
        _DT_EXECUTOR.submit(_dt_one, b, c)
        for b in range(B) for c in range(C)
    ]
    for f in futures:
        b, c, w = f.result()
        if w is not None:
            weights[b, c] = w

    return torch.from_numpy(weights).to(binary_mask.device)


class SoftCBDiceLoss(nn.Module):
    """Centerline Boundary Dice (cbDice) Loss for topology-preserving segmentation.

    cbDice extends clDice (Shit et al., CVPR 2021) by adding distance-transform-
    based boundary weighting via the Euclidean distance transform (EDT). This
    gives higher weight to thinner structures, better preserving thin neuron
    branches that standard clDice may underweight.

    The key improvement over clDice:
    - clDice: All skeleton pixels are weighted equally
    - cbDice: Skeleton pixels in thin structures (small local radius) receive
      higher weight via inverse-radius weighting from the EDT

    Formula:
        cbTprec = |W_t · skel(pred) · target| / |W_t · skel(pred)|
        cbTsens = |W_p · skel(target) · pred| / |W_p · skel(target)|
        cbDice = 2 · cbTprec · cbTsens / (cbTprec + cbTsens)

    Where W_t and W_p are inverse-radius distance transform weights computed
    from the target and prediction masks respectively.

    Why this works better for neurons:
    - Neuron branches are thin tubular structures with varying radius
    - Thin dendrites (radius ≈ 1-2 pixels) are harder to segment and more
      important biologically than the thick soma
    - cbDice automatically upweights errors on thin branches
    - Breaking a thin branch gets heavily penalized (both skeleton loss AND
      high weight from small radius)

    Args:
        num_iter: Iterations for soft skeletonization. Default 5.
        smooth: Smoothing factor to prevent division by zero. Default 1.0.

    Reference:
        Shi et al., "centerline boundary Dice loss for vascular segmentation",
        MICCAI 2024. https://github.com/PengchengShi1220/cbDice
    """

    def __init__(self, num_iter: int = 5, smooth: float = 1.0, topology_resolution: int = 256):
        super().__init__()
        self.num_iter = num_iter
        self.smooth = smooth
        self.topology_resolution = topology_resolution

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute cbDice loss.

        Args:
            pred: Predicted logits, shape [B, ...] or [..., H, W]
            target: Binary ground truth, same shape as pred.

        Returns:
            Scalar cbDice loss value (1 - cbDice).
        """
        # Convert logits to probabilities
        pred_prob = torch.sigmoid(pred)

        # Ensure 4D shape [B, C, H, W] for morphological operations
        original_shape = pred_prob.shape
        if len(original_shape) == 2:
            # [H, W] -> [1, 1, H, W]
            pred_prob = pred_prob.unsqueeze(0).unsqueeze(0)
            target = target.unsqueeze(0).unsqueeze(0)
        elif len(original_shape) == 3:
            # [B, H, W] -> [B, 1, H, W]
            pred_prob = pred_prob.unsqueeze(1)
            target = target.unsqueeze(1)
        elif len(original_shape) == 5:
            # [B, T, N, H, W] -> flatten to [B*T*N, 1, H, W]
            B, T, N, H, W = original_shape
            pred_prob = pred_prob.view(B * T * N, 1, H, W)
            target = target.view(B * T * N, 1, H, W)

        # Downsample for efficient topology computation.
        # Topology (skeleton connectivity) is a coarse property — 256×256
        # captures the same structure as 1024×1024 but 16× faster.
        H_cur, W_cur = pred_prob.shape[-2:]
        topo_res = self.topology_resolution
        if H_cur > topo_res or W_cur > topo_res:
            pred_prob = F.interpolate(
                pred_prob, size=(topo_res, topo_res),
                mode='bilinear', align_corners=False,
            )
            target = F.interpolate(
                target.float(), size=(topo_res, topo_res),
                mode='nearest',
            )

        # Compute soft skeleton for prediction (needs gradients)
        skel_pred = soft_skel(pred_prob, self.num_iter)

        # Compute soft skeleton for target (no gradients needed - saves memory)
        with torch.no_grad():
            skel_target = soft_skel(target.float(), self.num_iter)

        # Compute distance-transform weights (cbDice addition over clDice)
        with torch.no_grad():
            # Weights from target mask (for precision: weighting skel_pred)
            weights_target = _compute_distance_weights(target)
            # Weights from prediction mask (for sensitivity: weighting skel_target)
            pred_binary = (pred_prob > 0.5).float()
            weights_pred = _compute_distance_weights(pred_binary)

        # Weighted topology precision: how much of predicted skeleton overlaps
        # with target, weighted by inverse radius (thin structures count more)
        tprec_num = (weights_target * skel_pred * target).sum() + self.smooth
        tprec_den = (weights_target * skel_pred).sum() + self.smooth
        tprec = tprec_num / tprec_den

        # Weighted topology sensitivity: how much of target skeleton overlaps
        # with prediction, weighted by inverse radius
        tsens_num = (weights_pred * skel_target * pred_prob).sum() + self.smooth
        tsens_den = (weights_pred * skel_target).sum() + self.smooth
        tsens = tsens_num / tsens_den

        # cbDice: harmonic mean of weighted precision and sensitivity
        cb_dice = 2.0 * tprec * tsens / (tprec + tsens + 1e-8)

        # Clean up intermediate tensors
        del skel_pred, skel_target, weights_target, weights_pred

        return 1.0 - cb_dice


class SoftDiceCBDiceLoss(nn.Module):
    """Combined Dice + cbDice loss for tubular structure segmentation.

    This combines:
    - Standard Dice: Optimizes volumetric overlap
    - cbDice: Preserves topology/connectivity with boundary weighting

    The alpha parameter controls the balance:
    - alpha=0: Pure Dice loss (no topology awareness)
    - alpha=1: Pure cbDice loss (only topology)
    - alpha=0.5: Equal weight (recommended starting point)

    For neurons, we recommend alpha=0.3-0.5 to balance shape and connectivity.

    Args:
        alpha: Weight for cbDice. (1-alpha) for Dice. Default 0.5.
        num_iter: Iterations for soft skeletonization. Default 10.
        smooth: Smoothing factor. Default 1.0.
    """

    def __init__(self, alpha: float = 0.5, num_iter: int = 10, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.dice = DiceLoss(smooth=smooth)
        self.cbdice = SoftCBDiceLoss(num_iter=num_iter, smooth=smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined Dice + cbDice loss.

        Args:
            pred: Predicted logits.
            target: Binary ground truth.

        Returns:
            Weighted combination: (1-alpha)*Dice + alpha*cbDice
        """
        dice_loss = self.dice(pred, target)
        cbdice_loss = self.cbdice(pred, target)
        return (1.0 - self.alpha) * dice_loss + self.alpha * cbdice_loss


# =============================================================================
# Supervoxel Loss - Connectivity-Preserving Loss for Instance Segmentation
# =============================================================================
# Reference: Grim et al., "Efficient Connectivity-Preserving Instance
# Segmentation with Supervoxel-Based Loss Function", AAAI 2025
# https://arxiv.org/abs/2501.01022
# Implementation adapted from: https://github.com/AllenNeuralDynamics/supervoxel-loss
# =============================================================================


def _detect_critical_components_2d(
    pred_binary: np.ndarray,
    target_binary: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect critical supervoxels that affect topology.

    Uses scipy.ndimage.label (C-compiled) instead of Python BFS for
    ~100× faster connected component analysis.

    A supervoxel (connected component) is critical if adding or removing it
    changes the number of connected components in the segmentation.

    Two types:
    1. Negatively critical (in false negatives): If added, would merge
       disconnected ground truth components → their removal caused a split error
    2. Positively critical (in false positives): If removed, would separate
       incorrectly merged prediction components → they cause a merge error

    Args:
        pred_binary: Binary prediction mask (H, W), uint8
        target_binary: Binary ground truth mask (H, W), uint8

    Returns:
        neg_critical_mask: Mask of negatively critical false negative pixels
        pos_critical_mask: Mask of positively critical false positive pixels
    """
    h, w = pred_binary.shape

    # Compute error masks
    false_neg = (target_binary > 0) & (pred_binary == 0)  # Should be 1, predicted 0
    false_pos = (pred_binary > 0) & (target_binary == 0)  # Should be 0, predicted 1

    neg_critical_mask = np.zeros((h, w), dtype=np.float32)
    pos_critical_mask = np.zeros((h, w), dtype=np.float32)

    # 4-connectivity structuring element
    struct_4conn = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    # Label prediction components once (C-compiled, fast)
    pred_labels, _ = ndimage_label(pred_binary, structure=struct_4conn)

    # ===== Negatively Critical Components (False Negatives) =====
    # For each FN connected component, check if adding it to the prediction
    # would bridge 2+ distinct prediction components (causing a split error).
    fn_labels, n_fn = ndimage_label(false_neg, structure=struct_4conn)

    for comp_id in range(1, n_fn + 1):
        comp_mask = (fn_labels == comp_id)

        # Dilate by 1 pixel to find adjacent prediction components
        dilated = ndi_binary_dilation(comp_mask, structure=struct_4conn)
        neighbor_region = dilated & ~comp_mask

        # Find distinct prediction labels adjacent to this FN component
        adjacent_labels = np.unique(pred_labels[neighbor_region])
        adjacent_labels = adjacent_labels[adjacent_labels > 0]  # Remove background

        if len(adjacent_labels) >= 2:
            # This FN component bridges 2+ prediction components → critical
            neg_critical_mask[comp_mask] = 1.0

    # ===== Positively Critical Components (False Positives) =====
    # For each FP connected component, check if removing it would disconnect
    # the prediction (i.e., it's a bridge between prediction components).
    fp_labels, n_fp = ndimage_label(false_pos, structure=struct_4conn)

    for comp_id in range(1, n_fp + 1):
        comp_mask = (fp_labels == comp_id)

        # Find prediction pixels adjacent to this FP component
        dilated = ndi_binary_dilation(comp_mask, structure=struct_4conn)
        neighbor_pred = dilated & ~comp_mask & (pred_binary > 0)

        if neighbor_pred.sum() < 2:
            continue  # Need at least 2 neighbors to potentially disconnect

        # Remove this FP component and relabel prediction
        pred_without = pred_binary.copy()
        pred_without[comp_mask] = 0
        labels_without, _ = ndimage_label(pred_without, structure=struct_4conn)

        # Check if neighbors that were connected (through FP) are now split
        neighbor_new_labels = np.unique(labels_without[neighbor_pred])
        neighbor_new_labels = neighbor_new_labels[neighbor_new_labels > 0]

        if len(neighbor_new_labels) > 1:
            # Removing this FP disconnects the prediction → critical
            pos_critical_mask[comp_mask] = 1.0

    return neg_critical_mask, pos_critical_mask


class SupervoxelLoss2D(nn.Module):
    """Supervoxel-based connectivity-preserving loss.

    This loss identifies "critical supervoxels" - connected components whose
    addition/removal changes the number of connected components (topology)
    in the segmentation. These regions receive higher loss weight.

    The key insight is that small pixel errors in critical regions (like a
    1-pixel gap that splits a neuron) have huge topological impact, but
    standard losses weight them the same as any other pixel error.

    Two types of critical components:
    1. Negatively critical (false negatives): Missing pixels that cause splits
    2. Positively critical (false positives): Extra pixels that cause merges

    Loss formulation:
        L = (1-α)L₀(y,ŷ) + αβ∑L₀(yc,ŷc) + α(1-β)∑L₀(yc,ŷc)

    Where:
        - L₀ is the base loss (BCE)
        - α controls topology-aware vs standard loss balance
        - β balances negatively vs positively critical components
        - yc are pixels in critical supervoxels

    Computational complexity: O(n) using BFS, much lighter than clDice.

    Reference:
        Grim et al., "Efficient Connectivity-Preserving Instance Segmentation
        with Supervoxel-Based Loss Function", AAAI 2025

    Args:
        alpha: Weight for topology-aware terms vs standard loss (0-1). Default 0.5.
        beta: Balance between negatively and positively critical. Default 0.5.
        threshold: Threshold for binarizing predictions. Default 0.5.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        threshold: float = 0.5,
        topology_resolution: int = 256,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.topology_resolution = topology_resolution
        self.base_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute supervoxel connectivity loss.

        Critical region detection runs at reduced resolution for speed,
        while the actual loss gradients are computed at full resolution.

        Args:
            pred: Predicted logits, any shape (will process per-sample)
            target: Binary ground truth, same shape as pred

        Returns:
            Scalar supervoxel loss value
        """
        # Ensure we have batch dimension
        pred_flat = pred.view(-1, pred.shape[-2], pred.shape[-1])  # (N, H, W)
        target_flat = target.view(-1, target.shape[-2], target.shape[-1])

        batch_size = pred_flat.shape[0]
        device = pred.device
        H, W = pred_flat.shape[-2:]

        # Compute base BCE loss at FULL resolution (for accurate gradients)
        base_loss_per_pixel = self.base_loss(pred_flat, target_flat.float())

        # Binarize predictions
        pred_sigmoid = torch.sigmoid(pred_flat)
        pred_binary_full = (pred_sigmoid > self.threshold)
        target_binary_full = (target_flat > 0.5)

        # Downsample for topology detection (connectivity is a coarse property)
        topo_res = self.topology_resolution
        if H > topo_res or W > topo_res:
            pred_binary_ds = F.interpolate(
                pred_binary_full.unsqueeze(1).float(),
                size=(topo_res, topo_res), mode='nearest',
            ).squeeze(1)
            target_binary_ds = F.interpolate(
                target_binary_full.unsqueeze(1).float(),
                size=(topo_res, topo_res), mode='nearest',
            ).squeeze(1)
        else:
            pred_binary_ds = pred_binary_full.float()
            target_binary_ds = target_binary_full.float()

        pred_binary_batch = pred_binary_ds.cpu().numpy()
        target_binary_batch = target_binary_ds.cpu().numpy()

        total_loss = torch.tensor(0.0, device=device)

        for i in range(batch_size):
            pred_binary = pred_binary_batch[i]
            target_binary = target_binary_batch[i]

            # Skip if either mask is empty
            if pred_binary.sum() == 0 or target_binary.sum() == 0:
                # Just use standard loss for this sample
                total_loss = total_loss + base_loss_per_pixel[i].mean()
                continue

            # Detect critical components at LOW resolution (fast)
            neg_critical, pos_critical = _detect_critical_components_2d(
                pred_binary.astype(np.uint8),
                target_binary.astype(np.uint8)
            )

            # Convert to tensors
            neg_critical_t = torch.from_numpy(neg_critical).to(device)
            pos_critical_t = torch.from_numpy(pos_critical).to(device)

            # Upscale critical masks to FULL resolution for loss application
            if H > topo_res or W > topo_res:
                neg_critical_t = F.interpolate(
                    neg_critical_t.unsqueeze(0).unsqueeze(0),
                    size=(H, W), mode='nearest',
                ).squeeze(0).squeeze(0)
                pos_critical_t = F.interpolate(
                    pos_critical_t.unsqueeze(0).unsqueeze(0),
                    size=(H, W), mode='nearest',
                ).squeeze(0).squeeze(0)

            # Compute weighted loss at full resolution
            # Standard loss (non-critical regions)
            non_critical_mask = 1.0 - torch.clamp(neg_critical_t + pos_critical_t, 0, 1)
            standard_loss = (base_loss_per_pixel[i] * non_critical_mask).sum()

            # Critical region losses (higher weight)
            neg_critical_loss = (base_loss_per_pixel[i] * neg_critical_t).sum()
            pos_critical_loss = (base_loss_per_pixel[i] * pos_critical_t).sum()

            # Count pixels for normalization
            n_non_critical = non_critical_mask.sum().item() + 1e-6
            n_neg_critical = neg_critical_t.sum().item() + 1e-6
            n_pos_critical = pos_critical_t.sum().item() + 1e-6

            # Normalize each component
            standard_loss_norm = standard_loss / n_non_critical
            neg_critical_loss_norm = neg_critical_loss / n_neg_critical
            pos_critical_loss_norm = pos_critical_loss / n_pos_critical

            # Combined loss with alpha/beta weighting
            sample_loss = (
                (1 - self.alpha) * standard_loss_norm +
                self.alpha * self.beta * neg_critical_loss_norm +
                self.alpha * (1 - self.beta) * pos_critical_loss_norm
            )

            total_loss = total_loss + sample_loss

        return total_loss / batch_size


class MultiStepMultiMasksAndIous(nn.Module):
    """Combined loss function for SAM 2 training.

    This loss combines multiple objectives:
    1. Focal loss (loss_mask): Handles class imbalance
    2. Dice loss (loss_dice): Optimizes shape/overlap
    3. cbDice loss (loss_cbdice): Preserves topology via skeleton matching with
       boundary weighting
    4. Supervoxel loss (loss_supervoxel): Preserves connectivity via critical regions
    5. IoU loss (loss_iou): Trains IoU prediction head
    6. Object score loss (loss_class): Trains object presence prediction

    Both cbDice and Supervoxel losses are important for neuron segmentation:
    - cbDice: Ensures predicted skeleton matches ground truth skeleton, with
      higher weight on thin structures (via inverse-radius EDT weighting)
    - Supervoxel: Penalizes errors that cause split/merge topology changes

    The losses are weighted and summed into a single "core_loss" for backprop.

    Args:
        weight_dict: Dictionary mapping loss names to their weights.
        focal_alpha: Alpha parameter for focal loss. Default 0.25.
        focal_gamma: Gamma parameter for focal loss. Default 2.0.
        pred_obj_scores: Whether to compute object score loss. Default True.
        use_cbdice: Whether to use cbDice loss. Default True.
        cbdice_num_iter: Iterations for soft skeletonization. Default 5.
        use_supervoxel: Whether to use supervoxel loss. Default True.
    """

    def __init__(
        self,
        weight_dict: dict = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pred_obj_scores: bool = True,
        use_cbdice: bool = True,
        cbdice_num_iter: int = 5,
        use_supervoxel: bool = True,
        topology_resolution: int = 256,
    ):
        super().__init__()
        self.weight_dict = weight_dict or {
            "loss_mask": 5.0,        # Focal loss
            "loss_dice": 2.0,        # Standard Dice
            "loss_cbdice": 1.0,      # cbDice for skeleton topology
            "loss_supervoxel": 1.0,  # Supervoxel for connectivity
            "loss_iou": 1.0,
            "loss_class": 1.0,
        }
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss()
        self.use_cbdice = use_cbdice
        if use_cbdice:
            self.cbdice = SoftCBDiceLoss(
                num_iter=cbdice_num_iter,
                topology_resolution=topology_resolution,
            )
        self.use_supervoxel = use_supervoxel
        if use_supervoxel:
            self.supervoxel = SupervoxelLoss2D(
                alpha=0.5, beta=0.5,
                topology_resolution=topology_resolution,
            )
        self.pred_obj_scores = pred_obj_scores

    def forward(self, outputs: dict, targets: torch.Tensor) -> dict:
        """Compute combined loss.

        Args:
            outputs: Model outputs dict containing:
                - "pred_masks": Predicted mask logits [B, T, N, H, W] or similar
                - "pred_ious" (optional): Predicted IoU scores
                - "pred_obj_scores" (optional): Predicted object presence scores
            targets: Ground truth masks, same shape as pred_masks.

        Returns:
            Dictionary with individual losses and combined "core_loss":
                - "core_loss": Weighted sum of all losses (for backprop)
                - "loss_mask": Focal loss value
                - "loss_dice": Dice loss value
                - "loss_cbdice": cbDice loss value (skeleton topology)
                - "loss_supervoxel": Supervoxel loss value (connectivity)
                - "loss_iou": IoU prediction loss (or 0 if not available)
                - "loss_class": Object score loss (or 0 if not available)
        """
        pred_masks = outputs["pred_masks"]

        # Primary segmentation losses
        loss_mask = self.focal(pred_masks, targets)
        loss_dice = self.dice(pred_masks, targets)

        # cbDice loss for skeleton topology preservation (with boundary weighting)
        loss_cbdice = torch.tensor(0.0, device=pred_masks.device)
        if self.use_cbdice:
            loss_cbdice = self.cbdice(pred_masks, targets)

        # Supervoxel loss for connectivity preservation
        loss_supervoxel = torch.tensor(0.0, device=pred_masks.device)
        if self.use_supervoxel:
            loss_supervoxel = self.supervoxel(pred_masks, targets)

        # IoU prediction loss: train the model to predict its own accuracy
        loss_iou = torch.tensor(0.0, device=pred_masks.device)
        if "pred_ious" in outputs:
            pred_sigmoid = torch.sigmoid(pred_masks)
            # Compute actual IoU per mask
            intersection = (pred_sigmoid * targets).sum(dim=(-2, -1))
            union = pred_sigmoid.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1)) - intersection
            actual_iou = intersection / (union + 1e-6)
            # MSE between predicted and actual IoU
            loss_iou = F.mse_loss(outputs["pred_ious"], actual_iou)

        # Object score loss: train the model to know when there's an object
        loss_class = torch.tensor(0.0, device=pred_masks.device)
        if self.pred_obj_scores and "pred_obj_scores" in outputs:
            # Target: 1 if mask has any positive pixels, 0 otherwise
            has_object = (targets.sum(dim=(-2, -1)) > 0).float()
            loss_class = F.binary_cross_entropy_with_logits(
                outputs["pred_obj_scores"], has_object
            )

        # Weighted sum of all losses
        core_loss = (
            self.weight_dict.get("loss_mask", 5.0) * loss_mask +
            self.weight_dict.get("loss_dice", 2.0) * loss_dice +
            self.weight_dict.get("loss_cbdice", 1.0) * loss_cbdice +
            self.weight_dict.get("loss_supervoxel", 1.0) * loss_supervoxel +
            self.weight_dict.get("loss_iou", 1.0) * loss_iou +
            self.weight_dict.get("loss_class", 1.0) * loss_class
        )

        return {
            "core_loss": core_loss,
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
            "loss_cbdice": loss_cbdice,
            "loss_supervoxel": loss_supervoxel,
            "loss_iou": loss_iou,
            "loss_class": loss_class,
        }
