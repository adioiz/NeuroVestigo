"""Tests for the Betti-number topology metric (src/evaluate.py).

Betti-0 = number of connected components, Betti-1 = number of holes. These
capture connectivity errors (a severed neurite raises β0) that IoU/Dice miss.

Run: ``PYTHONPATH=. pytest tests/test_betti.py``  (or run this file directly)
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import (  # noqa: E402
    compute_betti_numbers,
    compute_betti_error,
    betti_error_per_frame,
)


def _disk(H, W, cy, cx, r):
    yy, xx = np.mgrid[:H, :W]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


# --------------------------------------------------------------------------- #
# Betti numbers of known shapes.
# --------------------------------------------------------------------------- #
def test_betti_numbers_known_shapes():
    H = W = 128
    assert compute_betti_numbers(np.zeros((H, W), bool)) == (0, 0)   # empty
    assert compute_betti_numbers(np.ones((H, W), bool)) == (1, 0)    # full
    assert compute_betti_numbers(_disk(H, W, 64, 64, 20)) == (1, 0)  # one blob
    # two disjoint blobs -> beta0 = 2
    two = _disk(H, W, 40, 40, 12) | _disk(H, W, 90, 90, 12)
    assert compute_betti_numbers(two) == (2, 0)
    # annulus -> one component, one hole
    ring = _disk(H, W, 64, 64, 30) & ~_disk(H, W, 64, 64, 12)
    assert compute_betti_numbers(ring) == (1, 1)
    # blob with two holes
    two_holes = (
        _disk(H, W, 64, 64, 40)
        & ~_disk(H, W, 50, 50, 6)
        & ~_disk(H, W, 80, 80, 6)
    )
    assert compute_betti_numbers(two_holes) == (1, 2)


# --------------------------------------------------------------------------- #
# A severed branch is a Betti-0 error of 1 but only a tiny IoU change — the
# whole reason this metric exists.
# --------------------------------------------------------------------------- #
def test_severed_branch_is_betti0_error():
    gt = np.zeros((40, 40), bool)
    gt[18:21, 4:36] = True                 # one connected bar
    pred = gt.copy()
    pred[18:21, 19:23] = False             # cut it into two -> beta0 = 2
    e0, e1 = compute_betti_error(pred, gt)
    assert e0 == 1, e0
    assert e1 == 0, e1
    # perfect prediction -> zero error
    assert compute_betti_error(gt, gt) == (0, 0)


def test_accepts_torch_tensors():
    m = torch.from_numpy(_disk(64, 64, 32, 32, 10))
    assert compute_betti_numbers(m) == (1, 0)


# --------------------------------------------------------------------------- #
# betti_error_per_frame over a fake model output: mirrors IoU's NaN handling
# (empty frames -> NaN) and averages objects per frame.
# --------------------------------------------------------------------------- #
def test_betti_error_per_frame():
    B, T, N, H, W = 1, 2, 1, 40, 40
    # Frame 0: GT bar, prediction severs it -> beta0 error 1.
    # Frame 1: both empty -> NaN sentinel.
    gt = torch.zeros(B, T, N, H, W)
    gt[0, 0, 0, 18:21, 4:36] = 1.0
    pred_logits = torch.full((B, T, N, H, W), -1.0)
    pred_logits[0, 0, 0, 18:21, 4:36] = 1.0
    pred_logits[0, 0, 0, 18:21, 19:23] = -1.0   # cut

    b0, b1 = betti_error_per_frame({"pred_masks": pred_logits}, gt)
    assert b0[0][0] == 1.0, b0
    assert b1[0][0] == 0.0, b1
    assert np.isnan(b0[0][1]), "empty frame should be NaN"


if __name__ == "__main__":
    test_betti_numbers_known_shapes()
    test_severed_branch_is_betti0_error()
    test_accepts_torch_tensors()
    test_betti_error_per_frame()
    print("all betti tests passed")
