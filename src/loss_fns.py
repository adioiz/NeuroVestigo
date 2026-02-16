"""Loss functions for SAM 2 fine-tuning.

This module implements the loss functions needed for training SAM 2,
replacing the Meta-internal training.loss_fns module.

Loss Components:
    - FocalLoss: Handles class imbalance (background >> neuron pixels)
    - DiceLoss: Optimizes for shape/overlap (IoU-like)
    - SoftCLDiceLoss: Topology-preserving loss for tubular structures (neurons)
    - SupervoxelLoss2D: Connectivity-preserving loss for split/merge errors
    - MultiStepMultiMasksAndIous: Combines all losses with configurable weights

References:
    - clDice: Shit et al., "clDice - A Novel Topology-Preserving Loss Function
      for Tubular Structure Segmentation", CVPR 2021
      https://arxiv.org/abs/2003.07311
    - Supervoxel Loss: Grim et al., "Efficient Connectivity-Preserving Instance
      Segmentation with Supervoxel-Based Loss Function", AAAI 2025
      https://arxiv.org/abs/2501.01022
"""

from collections import deque
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
# clDice Loss - Topology-Preserving Loss for Tubular Structures
# =============================================================================
# Reference: Shit et al., "clDice - A Novel Topology-Preserving Loss Function
# for Tubular Structure Segmentation", CVPR 2021
# Implementation adapted from: https://github.com/jocpae/clDice
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


class SoftCLDiceLoss(nn.Module):
    """Centerline Dice (clDice) Loss for topology-preserving segmentation.

    Standard Dice loss treats all pixels equally, which means breaking a
    thin connection (few pixels) has minimal impact on the loss. This is
    problematic for tubular structures like neurons, where connectivity
    is crucial.

    clDice addresses this by computing Dice on the skeletonized masks:
    - Skeleton of prediction vs. ground truth mask
    - Skeleton of ground truth vs. predicted mask

    This heavily penalizes predictions that break the centerline topology.

    Formula:
        Tprec = |skel(pred) ∩ target| / |skel(pred)|  (precision on skeleton)
        Tsens = |skel(target) ∩ pred| / |skel(target)|  (sensitivity on skeleton)
        clDice = 2 * Tprec * Tsens / (Tprec + Tsens)

    Why this works for neurons:
    - Neuron branches are thin tubular structures
    - Their skeleton is the centerline/medial axis
    - Breaking a branch removes its skeleton → low Tsens
    - Adding spurious branches → low Tprec

    Args:
        num_iter: Iterations for soft skeletonization. Default 10.
        smooth: Smoothing factor to prevent division by zero. Default 1.0.

    Reference:
        Shit et al., "clDice - A Novel Topology-Preserving Loss Function
        for Tubular Structure Segmentation", CVPR 2021
    """

    def __init__(self, num_iter: int = 5, smooth: float = 1.0):
        super().__init__()
        self.num_iter = num_iter
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute clDice loss.

        Args:
            pred: Predicted logits, shape [B, ...] or [..., H, W]
            target: Binary ground truth, same shape as pred.

        Returns:
            Scalar clDice loss value (1 - clDice).
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

        # Compute soft skeleton for prediction (needs gradients)
        skel_pred = soft_skel(pred_prob, self.num_iter)

        # Compute soft skeleton for target (no gradients needed - saves memory)
        with torch.no_grad():
            skel_target = soft_skel(target.float(), self.num_iter)

        # Topology precision: how much of predicted skeleton overlaps with target
        # High when predicted centerlines are in the right places
        tprec_num = (skel_pred * target).sum() + self.smooth
        tprec_den = skel_pred.sum() + self.smooth
        tprec = tprec_num / tprec_den

        # Topology sensitivity: how much of target skeleton overlaps with prediction
        # High when true centerlines are captured by prediction
        tsens_num = (skel_target * pred_prob).sum() + self.smooth
        tsens_den = skel_target.sum() + self.smooth
        tsens = tsens_num / tsens_den

        # clDice: harmonic mean of precision and sensitivity
        cl_dice = 2.0 * tprec * tsens / (tprec + tsens + 1e-8)

        # Clean up intermediate tensors
        del skel_pred, skel_target

        return 1.0 - cl_dice


class SoftDiceCLDiceLoss(nn.Module):
    """Combined Dice + clDice loss for tubular structure segmentation.

    This combines:
    - Standard Dice: Optimizes volumetric overlap
    - clDice: Preserves topology/connectivity

    The alpha parameter controls the balance:
    - alpha=0: Pure Dice loss (no topology awareness)
    - alpha=1: Pure clDice loss (only topology)
    - alpha=0.5: Equal weight (recommended starting point)

    For neurons, we recommend alpha=0.3-0.5 to balance shape and connectivity.

    Args:
        alpha: Weight for clDice. (1-alpha) for Dice. Default 0.5.
        num_iter: Iterations for soft skeletonization. Default 10.
        smooth: Smoothing factor. Default 1.0.
    """

    def __init__(self, alpha: float = 0.5, num_iter: int = 10, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.dice = DiceLoss(smooth=smooth)
        self.cldice = SoftCLDiceLoss(num_iter=num_iter, smooth=smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined Dice + clDice loss.

        Args:
            pred: Predicted logits.
            target: Binary ground truth.

        Returns:
            Weighted combination: (1-alpha)*Dice + alpha*clDice
        """
        dice_loss = self.dice(pred, target)
        cldice_loss = self.cldice(pred, target)
        return (1.0 - self.alpha) * dice_loss + self.alpha * cldice_loss


# =============================================================================
# Supervoxel Loss - Connectivity-Preserving Loss for Instance Segmentation
# =============================================================================
# Reference: Grim et al., "Efficient Connectivity-Preserving Instance
# Segmentation with Supervoxel-Based Loss Function", AAAI 2025
# https://arxiv.org/abs/2501.01022
# Implementation adapted from: https://github.com/AllenNeuralDynamics/supervoxel-loss
# =============================================================================


def _get_neighbors_2d(y: int, x: int, h: int, w: int) -> list:
    """Get 4-connected neighbors of a pixel."""
    neighbors = []
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            neighbors.append((ny, nx))
    return neighbors


def _bfs_component(
    mask: np.ndarray,
    start_y: int,
    start_x: int,
    visited: np.ndarray
) -> set:
    """Extract connected component using BFS.

    Args:
        mask: Binary mask (H, W)
        start_y, start_x: Starting pixel coordinates
        visited: Array tracking visited pixels

    Returns:
        Set of (y, x) coordinates in the component
    """
    h, w = mask.shape
    component = set()
    queue = deque([(start_y, start_x)])
    visited[start_y, start_x] = True

    while queue:
        y, x = queue.popleft()
        component.add((y, x))

        for ny, nx in _get_neighbors_2d(y, x, h, w):
            if mask[ny, nx] > 0 and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))

    return component


def _count_components(mask: np.ndarray) -> int:
    """Count number of connected components in binary mask."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    count = 0

    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0 and not visited[y, x]:
                _bfs_component(mask, y, x, visited)
                count += 1

    return count


def _detect_critical_components_2d(
    pred_binary: np.ndarray,
    target_binary: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect critical supervoxels that affect topology.

    A supervoxel (connected component) is critical if adding or removing it
    changes the number of connected components in the segmentation.

    Two types:
    1. Negatively critical (in false negatives): If added, would merge
       disconnected ground truth components → their removal caused a split error
    2. Positively critical (in false positives): If removed, would separate
       incorrectly merged prediction components → they cause a merge error

    Args:
        pred_binary: Binary prediction mask (H, W)
        target_binary: Binary ground truth mask (H, W)

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

    # ===== Negatively Critical Components (False Negatives) =====
    # If adding this FN component to prediction would reduce # of GT components visible
    # in prediction, then it's a "bridge" component whose absence causes split errors
    visited_fn = np.zeros_like(false_neg, dtype=bool)

    for y in range(h):
        for x in range(w):
            if false_neg[y, x] and not visited_fn[y, x]:
                # Extract this FN component
                component = _bfs_component(false_neg, y, x, visited_fn)

                # Check if adding this component would change connectivity
                # by checking neighbors of the component in the prediction
                neighbor_labels = set()
                pred_with_component = pred_binary.copy()

                for cy, cx in component:
                    pred_with_component[cy, cx] = 1
                    for ny, nx in _get_neighbors_2d(cy, cx, h, w):
                        if pred_binary[ny, nx] > 0:
                            neighbor_labels.add((ny, nx))

                # If component touches 2+ distinct predicted regions, it's critical
                if len(neighbor_labels) >= 2:
                    # Check if these neighbors are from different components
                    pred_visited = np.zeros_like(pred_binary, dtype=bool)
                    distinct_neighbors = 0
                    for ny, nx in neighbor_labels:
                        if not pred_visited[ny, nx]:
                            _bfs_component(pred_binary, ny, nx, pred_visited)
                            distinct_neighbors += 1

                    if distinct_neighbors >= 2:
                        # This FN component bridges multiple prediction components
                        for cy, cx in component:
                            neg_critical_mask[cy, cx] = 1.0

    # ===== Positively Critical Components (False Positives) =====
    # If removing this FP component would increase # of components in prediction,
    # it's merging things that shouldn't be merged
    visited_fp = np.zeros_like(false_pos, dtype=bool)

    for y in range(h):
        for x in range(w):
            if false_pos[y, x] and not visited_fp[y, x]:
                # Extract this FP component
                component = _bfs_component(false_pos, y, x, visited_fp)

                # Create prediction without this component
                pred_without_component = pred_binary.copy()
                for cy, cx in component:
                    pred_without_component[cy, cx] = 0

                # Check if neighbors in remaining prediction are still connected
                neighbor_pixels = []
                for cy, cx in component:
                    for ny, nx in _get_neighbors_2d(cy, cx, h, w):
                        if pred_without_component[ny, nx] > 0:
                            neighbor_pixels.append((ny, nx))

                if len(neighbor_pixels) >= 2:
                    # Check if removing component disconnects neighbors
                    visited_check = np.zeros_like(pred_without_component, dtype=bool)
                    ny, nx = neighbor_pixels[0]
                    reachable = _bfs_component(pred_without_component, ny, nx, visited_check)

                    # If any neighbor is not reachable, component was critical
                    all_connected = all((y, x) in reachable for y, x in neighbor_pixels)

                    if not all_connected:
                        for cy, cx in component:
                            pos_critical_mask[cy, cx] = 1.0

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
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.base_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute supervoxel connectivity loss.

        Args:
            pred: Predicted logits, any shape (will process per-sample)
            target: Binary ground truth, same shape as pred

        Returns:
            Scalar supervoxel loss value
        """
        # Ensure we have batch dimension
        original_shape = pred.shape
        pred_flat = pred.view(-1, pred.shape[-2], pred.shape[-1])  # (N, H, W)
        target_flat = target.view(-1, target.shape[-2], target.shape[-1])

        batch_size = pred_flat.shape[0]
        device = pred.device

        # Compute base BCE loss for all pixels
        base_loss_per_pixel = self.base_loss(pred_flat, target_flat.float())

        # Binarize predictions
        pred_binary_batch = (torch.sigmoid(pred_flat) > self.threshold).cpu().numpy()
        target_binary_batch = (target_flat > 0.5).cpu().numpy()

        total_loss = torch.tensor(0.0, device=device)

        for i in range(batch_size):
            pred_binary = pred_binary_batch[i]
            target_binary = target_binary_batch[i]

            # Skip if either mask is empty
            if pred_binary.sum() == 0 or target_binary.sum() == 0:
                # Just use standard loss for this sample
                total_loss = total_loss + base_loss_per_pixel[i].mean()
                continue

            # Detect critical components (CPU operation)
            neg_critical, pos_critical = _detect_critical_components_2d(
                pred_binary.astype(np.uint8),
                target_binary.astype(np.uint8)
            )

            # Convert to tensors
            neg_critical_t = torch.from_numpy(neg_critical).to(device)
            pos_critical_t = torch.from_numpy(pos_critical).to(device)

            # Compute weighted loss
            # Standard loss (non-critical regions)
            non_critical_mask = 1.0 - torch.clamp(neg_critical_t + pos_critical_t, 0, 1)
            standard_loss = (base_loss_per_pixel[i] * non_critical_mask).sum()

            # Critical region losses (higher weight)
            neg_critical_loss = (base_loss_per_pixel[i] * neg_critical_t).sum()
            pos_critical_loss = (base_loss_per_pixel[i] * pos_critical_t).sum()

            # Count pixels for normalization
            n_total = pred_flat.shape[-2] * pred_flat.shape[-1]
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
    3. clDice loss (loss_cldice): Preserves topology via skeleton matching
    4. Supervoxel loss (loss_supervoxel): Preserves connectivity via critical regions
    5. IoU loss (loss_iou): Trains IoU prediction head
    6. Object score loss (loss_class): Trains object presence prediction

    Both clDice and Supervoxel losses are important for neuron segmentation:
    - clDice: Ensures predicted skeleton matches ground truth skeleton
    - Supervoxel: Penalizes errors that cause split/merge topology changes

    The losses are weighted and summed into a single "core_loss" for backprop.

    Args:
        weight_dict: Dictionary mapping loss names to their weights.
        focal_alpha: Alpha parameter for focal loss. Default 0.25.
        focal_gamma: Gamma parameter for focal loss. Default 2.0.
        pred_obj_scores: Whether to compute object score loss. Default True.
        use_cldice: Whether to use clDice loss. Default True.
        cldice_num_iter: Iterations for soft skeletonization. Default 5.
        use_supervoxel: Whether to use supervoxel loss. Default True.
    """

    def __init__(
        self,
        weight_dict: dict = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pred_obj_scores: bool = True,
        use_cldice: bool = True,
        cldice_num_iter: int = 5,
        use_supervoxel: bool = True,
    ):
        super().__init__()
        self.weight_dict = weight_dict or {
            "loss_mask": 5.0,        # Focal loss
            "loss_dice": 2.0,        # Standard Dice
            "loss_cldice": 1.0,      # clDice for skeleton topology
            "loss_supervoxel": 1.0,  # Supervoxel for connectivity
            "loss_iou": 1.0,
            "loss_class": 1.0,
        }
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss()
        self.use_cldice = use_cldice
        if use_cldice:
            self.cldice = SoftCLDiceLoss(num_iter=cldice_num_iter)
        self.use_supervoxel = use_supervoxel
        if use_supervoxel:
            self.supervoxel = SupervoxelLoss2D(alpha=0.5, beta=0.5)
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
                - "loss_cldice": clDice loss value (skeleton topology)
                - "loss_supervoxel": Supervoxel loss value (connectivity)
                - "loss_iou": IoU prediction loss (or 0 if not available)
                - "loss_class": Object score loss (or 0 if not available)
        """
        pred_masks = outputs["pred_masks"]

        # Primary segmentation losses
        loss_mask = self.focal(pred_masks, targets)
        loss_dice = self.dice(pred_masks, targets)

        # clDice loss for skeleton topology preservation
        loss_cldice = torch.tensor(0.0, device=pred_masks.device)
        if self.use_cldice:
            loss_cldice = self.cldice(pred_masks, targets)

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
            self.weight_dict.get("loss_cldice", 1.0) * loss_cldice +
            self.weight_dict.get("loss_supervoxel", 1.0) * loss_supervoxel +
            self.weight_dict.get("loss_iou", 1.0) * loss_iou +
            self.weight_dict.get("loss_class", 1.0) * loss_class
        )

        return {
            "core_loss": core_loss,
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
            "loss_cldice": loss_cldice,
            "loss_supervoxel": loss_supervoxel,
            "loss_iou": loss_iou,
            "loss_class": loss_class,
        }
