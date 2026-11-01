"""
native_res.py -- bring an automatic segmenter's output from its upscaled grid
back to native resolution without shifting it.

Cellpose-SAM, Omnipose and NeuroQuantify run on frames enlarged s times
(s = 2, 3, 4). Their instance maps used to come back with cv2.INTER_NEAREST,
which samples the TOP-LEFT pixel of each s x s block, i.e. a point (s-1)/(2s)
of a native pixel up-left of the pixel's centre. INTER_NEAREST_EXACT fixes
that only for odd s: for even s no single pixel sits at the centre, and it
just moves the offset to the other side.

So the target is chosen where the segmenter worked -- union of instances that
overlap the ground truth, at the upscaled resolution -- and that binary map is
shrunk by AREA: a native pixel is target when at least half of its block is.
Centred for every integer s. Ideal-segmenter test (thin lines, IoU vs GT):
INTER_NEAREST 0.76 / 0.68 / 0.65 at x2 / x3 / x4, this 0.86 / 0.99 / 0.94.
"""
from __future__ import annotations

import numpy as np


def _scale(big_shape, small_shape) -> int:
    (hu, wu), (H, W) = big_shape, small_shape
    if hu % H or wu % W or hu // H != wu // W:
        raise ValueError(f"upscaled grid {big_shape} is not an integer multiple of {small_shape}")
    return hu // H


def target_at_native(inst_up: np.ndarray, gt: np.ndarray, match_to_target) -> np.ndarray:
    """inst_up: instance labels on the upscaled grid. gt: native ground truth.
    match_to_target(labels, gt) is the runner's own union matcher.
    Returns a native-resolution bool mask."""
    H, W = gt.shape
    s = _scale(inst_up.shape, gt.shape)
    gt_up = np.repeat(np.repeat(gt > 0, s, axis=0), s, axis=1)        # exact, centred
    t_up = np.asarray(match_to_target(inst_up, gt_up), dtype=np.float32)
    return t_up.reshape(H, s, W, s).mean(axis=(1, 3)) >= 0.5         # area: block majority


def labels_to_native(labels_up: np.ndarray, H: int, W: int) -> np.ndarray:
    """Label map for the figures' raw stacks: centre-sampled where s is odd,
    within 1/(2s) of a native pixel where it is even. Not used for scoring."""
    import cv2
    return cv2.resize(np.asarray(labels_up).astype(np.int32), (W, H),
                      interpolation=cv2.INTER_NEAREST_EXACT)
