"""
Pixel alignment of the resizes that bring predictions back to native size.

A model that predicts exactly the truth must score IoU 1 after the round trip
native -> model resolution -> native. The old run_ours.py order (threshold at
1024, then F.interpolate(mode="nearest")) failed this: "nearest" samples the
top-left pixel of each block, so masks came back about a third of a pixel off
and thin neurites moved a whole pixel. Found by an outside reproduction, worth
+0.04 IoU on run 1 with the same logits.
"""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "benchmark"))

from run_ours import logits_to_native  # noqa: E402

SIZES = [(148, 318), (292, 292), (498, 452), (353, 615)]    # the recordings' range


def thin_mask(H, W):
    """1- and 2-pixel-wide lines at assorted offsets, plus a small blob."""
    m = np.zeros((H, W), np.uint8)
    for c in (7, W // 3, W // 2 + 1, W - 9):
        m[:, c] = 1
    for r in (5, H // 2):
        m[r:r + 2, :] = 1
    m[H // 4:H // 4 + 6, W // 5:W // 5 + 9] = 1
    return m


def perfect_logits(mask, S=1024):
    """What a perfect model would output at S x S: the truth as a +-1 field,
    enlarged centre-aligned like the image the model sees."""
    x = torch.from_numpy(mask.astype(np.float32) * 2 - 1)[None, None]
    return F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False)[0]


@pytest.mark.parametrize("H,W", SIZES)
def test_logits_to_native_is_aligned(H, W):
    m = thin_mask(H, W)
    out = logits_to_native(perfect_logits(m), H, W)[0]
    assert out.shape == (H, W)
    assert np.array_equal(out, m), f"{(out != m).sum()} pixels differ at {H}x{W}"


@pytest.mark.parametrize("H,W", SIZES)
def test_old_resize_was_misaligned(H, W):
    """Documents the bug the fix removes: the old order loses thin lines."""
    m = thin_mask(H, W)
    binary = (perfect_logits(m) > 0).float()[:, None]
    old = (F.interpolate(binary, size=(H, W), mode="nearest")[:, 0, ].numpy() > 0.5)[0]
    assert not np.array_equal(old, m)


# --- automatic baselines (benchmark/native_res.py) ---------------------------

def _thin_scene(H, W, rng, n=14):
    """Straight segments at random sub-pixel position, angle and width 0.8-2.5 px."""
    segs = [(rng.uniform(5, H - 5), rng.uniform(5, W - 5), rng.uniform(0, np.pi),
             rng.uniform(10, 40), rng.uniform(0.8, 2.5)) for _ in range(n)]

    def inside(y, x):
        m = np.zeros(y.shape, bool)
        for cy, cx, a, L, w in segs:
            dy, dx = y - cy, x - cx
            along = dx * np.cos(a) + dy * np.sin(a)
            across = -dx * np.sin(a) + dy * np.cos(a)
            m |= (np.abs(along) <= L / 2) & (np.abs(across) <= w / 2)
        return m
    return inside


@pytest.mark.parametrize("s", [2, 3, 4])
def test_baseline_target_comes_back_centred(s):
    """A perfect segmenter on the s-times grid must come back better than the
    old corner-sampling shrink, and unshifted."""
    from scipy import ndimage
    from native_res import target_at_native

    def match(inst, gt):                       # the runners' union matcher
        labs = np.unique(inst[gt])
        return np.isin(inst, labs[labs > 0])

    def iou(a, b):
        return np.logical_and(a, b).sum() / np.logical_or(a, b).sum()

    rng = np.random.default_rng(s)
    H, W = 120, 160
    new_iou, old_iou = [], []
    for _ in range(10):
        inside = _thin_scene(H, W, rng)
        yy, xx = np.mgrid[0:H, 0:W].astype(float)
        gt = inside(yy, xx)                                          # pixel-centre GT
        Y, X = np.mgrid[0:H * s, 0:W * s].astype(float)
        inst_up = ndimage.label(inside((Y + 0.5) / s - 0.5, (X + 0.5) / s - 0.5))[0]
        new = target_at_native(inst_up, gt, match)
        old = match(inst_up[::s, ::s], gt)                           # = cv2.INTER_NEAREST
        new_iou.append(iou(new, gt))
        old_iou.append(iou(old, gt))
        nudges = {(dy, dx): iou(np.roll(np.roll(new, dy, 0), dx, 1), gt)
                  for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
        assert max(nudges, key=nudges.get) == (0, 0)
    assert np.mean(new_iou) > np.mean(old_iou) + 0.05
    assert np.mean(new_iou) > 0.85
