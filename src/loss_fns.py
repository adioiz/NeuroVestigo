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
from scipy.ndimage import find_objects as ndi_find_objects


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

    Note (k=0 seed): the reference clDice implementation
    (github.com/jocpae/clDice) seeds the skeleton at scale 0 —
    ``skel = relu(img - open(img))`` — *before* the erosion loop. Omitting
    it (as an earlier version of this function did) destroys any structure
    ≤2 px wide on the first erosion, yielding an empty skeleton for the
    thinnest neurites. The seed line below restores that term. Bounding to
    [0, 1] uses the reference's soft-union ``relu(delta - skel*delta)``
    rather than a hard clamp.
    """
    # k=0 seed term (skeleton at the original scale).
    skeleton = F.relu(img - soft_open(img))

    img_eroded = img
    for _ in range(num_iter):
        img_eroded = soft_erode(img_eroded)
        img_opened = soft_open(img_eroded)
        # Points removed by opening = thin parts = skeleton candidates.
        delta = F.relu(img_eroded - img_opened)
        # Soft union (reference clDice): add only the part of delta not
        # already captured by the accumulated skeleton; keeps S in [0, 1].
        skeleton = skeleton + F.relu(delta - skeleton * delta)

    return skeleton


# Module-level thread pool for parallel distance-transform computation.
# scipy.ndimage.distance_transform_edt is a C extension that releases the
# GIL, so threads (not processes) actually parallelize. Created once and
# reused across cbDice forward passes — submission overhead is ~tens of µs.
from concurrent.futures import ThreadPoolExecutor
_DT_EXECUTOR = ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="cbdice-dt",
)


def _edt_batch(mask: torch.Tensor) -> torch.Tensor:
    """Euclidean distance transform of a batch of 2-D masks.

    Args:
        mask: [N, H, W] tensor, hard (0/1). EDT is computed per sample.

    Returns:
        distances: [N, H, W] float tensor on ``mask.device``; each foreground
        pixel holds its distance to the nearest background pixel (0 on
        background, matching scipy's convention).

    Implementation note: scipy's ``distance_transform_edt`` is a C extension
    that releases the GIL, so per-sample calls parallelise across the module
    thread pool. Collected in submission order for deterministic output.
    """
    from scipy.ndimage import distance_transform_edt

    mask_np = mask.detach().cpu().numpy()
    N = mask_np.shape[0]
    out = np.zeros_like(mask_np, dtype=np.float32)

    def _dt_one(i: int):
        m = mask_np[i]
        if m.sum() == 0:
            return i, None
        return i, distance_transform_edt(m).astype(np.float32)

    futures = [_DT_EXECUTOR.submit(_dt_one, i) for i in range(N)]
    for f in futures:
        i, d = f.result()
        if d is not None:
            out[i] = d
    return torch.from_numpy(out).to(mask.device)


def _cbdice_get_weights(
    mask_input: torch.Tensor,
    skel_input: torch.Tensor,
    dim: int = 2,
    prob_flag: bool = True,
):
    """Radius-normalised cbDice weight maps (faithful port of the reference).

    Port of ``get_weights`` from github.com/PengchengShi1220/cbDice
    (loss/cbdice_loss.py). Returns three maps for one side (prediction or
    label):

        q_v   = dist_map_norm * mask   — normalised distance-volume weight
        q_sv  = skel_R_norm  * mask   — skeleton-radius weight over the mask
        q_s   = I_norm       * skel   — inverse-radius weight on the skeleton

    Radii come from the EDT of the *hard* mask; per-sample they are
    normalised by the maximum skeleton radius. When ``prob_flag`` is True the
    inputs are probabilities: hard supports (``>0.5``) drive the geometry
    (under no-grad), and the returned maps are rescaled by the probabilities
    so gradients flow — exactly as the reference does.

    Args:
        mask_input: [N, H, W]. Probabilities if ``prob_flag`` else hard 0/1.
        skel_input: [N, H, W] skeleton, same convention as ``mask_input``.
        dim: 2 or 3 (controls the I_norm exponent). Neurons are 2-D.
        prob_flag: True for the prediction side, False for the label side.
    """
    if prob_flag:
        mask_prob, skel_prob = mask_input, skel_input
        mask = (mask_prob > 0.5).float()
        skel = (skel_prob > 0.5).float()
    else:
        mask = mask_input
        skel = skel_input

    # Radii from the hard mask (non-differentiable geometry).
    with torch.no_grad():
        distances = _edt_batch(mask)          # [N, H, W], 0 on background
        skel_radius = distances * skel        # radius along the skeleton

        dist_map_norm = torch.zeros_like(distances)
        skel_R_norm = torch.zeros_like(distances)
        I_norm = torch.zeros_like(distances)
        for i in range(distances.shape[0]):
            sr = skel_radius[i]
            sr_max = torch.clamp(sr.max(), min=1.0)
            sr_min = torch.clamp(sr.min(), min=1.0)  # skeleton min is 0 -> 1
            dist_map_norm[i] = distances[i].clamp(max=sr_max) / sr_max
            skel_R_norm[i] = sr / sr_max
            ratio = (sr_max - sr + sr_min) / sr_max
            I_norm[i] = ratio if dim == 2 else ratio ** 2
        I_norm = I_norm * skel  # zero off-skeleton (ref: I_norm[skel==0]=0)

    if prob_flag:
        return dist_map_norm * mask_prob, skel_R_norm * mask_prob, I_norm * skel_prob
    return dist_map_norm * mask, skel_R_norm * mask, I_norm * skel


def _combine_tensors(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Reference cbDice ``combine_tensors``: prefer B*C, fall back to A*C.

    D = B*C everywhere, except where A is non-zero and B is zero, where it
    takes A*C. Used to build the cbDice denominators.
    """
    A_C = A * C
    B_C = B * C
    D = B_C.clone()
    mask_AC = (A != 0) & (B == 0)
    D[mask_AC] = A_C[mask_AC]
    return D


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

    Formula (faithful port of PengchengShi1220/cbDice, loss/cbdice_loss.py):
        Let get_weights(mask, skel) -> (q_v, q_sv, q_s) be the radius-
        normalised maps (see _cbdice_get_weights). With subscripts p/l for
        prediction/label:

        w_tprec = Σ(q_sp · q_vl) / Σ combine(q_spvp, q_slvl, q_sp)
        w_tsens = Σ(q_sl · q_vp) / Σ combine(q_slvl, q_spvp, q_sl)
        cbDice  = 2 · w_tprec · w_tsens / (w_tprec + w_tsens)
        loss    = 1 − cbDice

    Precision multiplies the *prediction* skeleton weight (q_sp) by the
    *label* distance-volume (q_vl); sensitivity multiplies the *label*
    skeleton weight (q_sl) by the *prediction* distance-volume (q_vp). The
    denominators use the reference combine_tensors() merge. The reference
    returns −cbDice (its compound wrapper adds the constant); we return the
    equivalent 1 − cbDice so this stays a proper loss in [0, 1].

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
        #
        # Both sides MUST be resampled the same way. An earlier version resized
        # the prediction 'bilinear' and the target 'nearest'; on a bit-identical
        # pair the two grids then disagreed on 100-400+ pixels, which the loss
        # scored as topology error. Measured: a *perfect* prediction cost
        # 0.008-0.085 instead of 0 (and the training log sat at cbdice ≈ 0.14),
        # so a large share of the signal was pure resampling artefact. Resizing
        # the target bilinear and re-binarising at 0.5 drives the
        # perfect-prediction loss to exactly 0 at every resolution.
        H_cur, W_cur = pred_prob.shape[-2:]
        topo_res = self.topology_resolution
        if H_cur > topo_res or W_cur > topo_res:
            pred_prob = F.interpolate(
                pred_prob, size=(topo_res, topo_res),
                mode='bilinear', align_corners=False,
            )
            target = (
                F.interpolate(
                    target.float(), size=(topo_res, topo_res),
                    mode='bilinear', align_corners=False,
                ) > 0.5
            ).float()

        # Collapse channel: [N, 1, H, W] -> [N, H, W].
        y_pred_prob = pred_prob.squeeze(1)
        y_true = (target > 0).float().squeeze(1)

        # Skeletonise the HARD masks (reference behaviour): the skeleton
        # support is non-differentiable; gradients are reinjected below by
        # multiplying the prediction skeleton by the predicted probabilities.
        with torch.no_grad():
            y_pred_hard = (y_pred_prob > 0.5).float()
            skel_pred_hard = soft_skel(y_pred_hard.unsqueeze(1), self.num_iter).squeeze(1)
            skel_true = soft_skel(y_true.unsqueeze(1), self.num_iter).squeeze(1)
        skel_pred_prob = skel_pred_hard * y_pred_prob  # carries gradient

        # Radius-normalised weight maps for each side (dim=2 for image data).
        q_vl, q_slvl, q_sl = _cbdice_get_weights(y_true, skel_true, dim=2, prob_flag=False)
        q_vp, q_spvp, q_sp = _cbdice_get_weights(
            y_pred_prob, skel_pred_prob, dim=2, prob_flag=True,
        )

        # Topology precision: prediction skeleton (q_sp) scored against the
        # label distance-volume (q_vl); denominator merges both sides.
        w_tprec = (torch.sum(q_sp * q_vl) + self.smooth) / (
            torch.sum(_combine_tensors(q_spvp, q_slvl, q_sp)) + self.smooth
        )
        # Topology sensitivity: label skeleton (q_sl) scored against the
        # prediction distance-volume (q_vp).
        w_tsens = (torch.sum(q_sl * q_vp) + self.smooth) / (
            torch.sum(_combine_tensors(q_slvl, q_spvp, q_sl)) + self.smooth
        )

        cb_dice = 2.0 * w_tprec * w_tsens / (w_tprec + w_tsens + 1e-8)
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


def _pad_box(box, shape, pad: int = 1):
    """Grow a ``find_objects`` bounding box by ``pad``, clipped to ``shape``.

    One pixel of padding is exactly what a 1-pixel dilation needs, so a
    per-component test that only looks at itself and its immediate neighbours
    can run inside this window instead of over the whole frame.
    """
    return tuple(
        slice(max(0, s.start - pad), min(n, s.stop + pad))
        for s, n in zip(box, shape)
    )


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
    shape = (h, w)

    # Compute error masks
    false_neg = (target_binary > 0) & (pred_binary == 0)  # Should be 1, predicted 0
    false_pos = (pred_binary > 0) & (target_binary == 0)  # Should be 0, predicted 1

    neg_critical_mask = np.zeros((h, w), dtype=np.float32)
    pos_critical_mask = np.zeros((h, w), dtype=np.float32)

    # 4-connectivity structuring element
    struct_4conn = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    # Label prediction components once (C-compiled, fast)
    pred_labels, _ = ndimage_label(pred_binary, structure=struct_4conn)
    pred_boxes = ndi_find_objects(pred_labels)

    # ===== Negatively Critical Components (False Negatives) =====
    # For each FN connected component, check if adding it to the prediction
    # would bridge 2+ distinct prediction components (causing a split error).
    fn_labels, _ = ndimage_label(false_neg, structure=struct_4conn)

    for comp_id, box in enumerate(ndi_find_objects(fn_labels), start=1):
        if box is None:
            continue
        # A component lies entirely inside its own bounding box, and a
        # 1-pixel dilation cannot leave that box padded by 1 -- so the whole
        # test is local. Working in the padded box instead of the full frame
        # is what turns this loop from O(n_comp * H * W) into O(total area).
        p = _pad_box(box, shape)
        comp_mask = (fn_labels[p] == comp_id)

        # Dilate by 1 pixel to find adjacent prediction components
        dilated = ndi_binary_dilation(comp_mask, structure=struct_4conn)
        neighbor_region = dilated & ~comp_mask

        # Find distinct prediction labels adjacent to this FN component
        adjacent_labels = np.unique(pred_labels[p][neighbor_region])
        adjacent_labels = adjacent_labels[adjacent_labels > 0]  # Remove background

        if len(adjacent_labels) >= 2:
            # This FN component bridges 2+ prediction components → critical
            neg_critical_mask[p][comp_mask] = 1.0

    # ===== Positively Critical Components (False Positives) =====
    # For each FP connected component, check if removing it would disconnect
    # the prediction (i.e., it's a bridge between prediction components).
    fp_labels, _ = ndimage_label(false_pos, structure=struct_4conn)

    for comp_id, box in enumerate(ndi_find_objects(fp_labels), start=1):
        if box is None:
            continue

        # Cheap local reject first. Isolated speckles are the overwhelming
        # majority of FP components (~1000 per mask at 512²) and all of them
        # die here, inside a handful of pixels.
        p = _pad_box(box, shape)
        comp_local = (fp_labels[p] == comp_id)
        dilated = ndi_binary_dilation(comp_local, structure=struct_4conn)
        neighbor_local = dilated & ~comp_local & (pred_binary[p] > 0)

        if neighbor_local.sum() < 2:
            continue  # Need at least 2 neighbors to potentially disconnect

        # Every FP pixel is also a prediction pixel, and the component is
        # connected, so it sits inside exactly one prediction component `pid`;
        # every neighbour pixel is 4-adjacent to it and therefore sits in
        # `pid` too. Connectivity among those pixels is fully determined
        # inside pid's bounding box, so removing the component and relabelling
        # can be done there rather than over the whole frame. Identical result,
        # one relabel of a neuron instead of one relabel of the image.
        pid = int(pred_labels[p][comp_local][0])
        q = _pad_box(pred_boxes[pid - 1], shape)
        comp_mask = (fp_labels[q] == comp_id)
        dilated = ndi_binary_dilation(comp_mask, structure=struct_4conn)
        neighbor_pred = dilated & ~comp_mask & (pred_binary[q] > 0)

        # Remove this FP component and relabel prediction
        pred_without = (pred_labels[q] == pid)
        pred_without[comp_mask] = False
        labels_without, _ = ndimage_label(pred_without, structure=struct_4conn)

        # Check if neighbors that were connected (through FP) are now split
        neighbor_new_labels = np.unique(labels_without[neighbor_pred])
        neighbor_new_labels = neighbor_new_labels[neighbor_new_labels > 0]

        if len(neighbor_new_labels) > 1:
            # Removing this FP disconnects the prediction → critical
            pos_critical_mask[q][comp_mask] = 1.0

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
