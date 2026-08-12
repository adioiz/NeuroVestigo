"""Regression tests for the supervoxel critical-component detector.

The detector used to scan the whole frame once per connected component: an
``(fn_labels == comp_id)`` compare, a dilation and an ``np.unique`` for every
false-negative component, and -- far worse -- a full-image ``ndimage_label``
for every false-positive component. That is O(n_components x H x W), and both
factors grow with resolution, so raising ``topology_resolution`` from 256 to
512 cost ~16x rather than the expected 4x (measured 1.6s -> 24.9s per training
step of 16 masks). It was the dominant term in a 2.9x training slowdown.

The fix restricts each per-component test to that component's bounding box,
which is sound because a component lies inside its own box and a 1-pixel
dilation cannot escape a box padded by 1. For the false-positive relabel it
restricts to the *parent prediction component's* box, which is sound because
every pixel involved in that test provably lies in that one component.

Both are exact, so the tests below assert bit-identical output against a
straightforward whole-frame reference -- speed must never come at the cost of
a different loss value.

Run: ``KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. pytest tests/test_supervoxel_speed.py``
"""
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import binary_dilation as ndi_binary_dilation
from scipy.ndimage import label as ndimage_label

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loss_fns import _detect_critical_components_2d  # noqa: E402

_S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def _reference(pred_binary, target_binary):
    """Whole-frame implementation, kept deliberately naive.

    This is the pre-optimisation algorithm written out plainly. It is the
    definition of correct for these tests; the shipped version must agree with
    it exactly on every input.
    """
    h, w = pred_binary.shape
    false_neg = (target_binary > 0) & (pred_binary == 0)
    false_pos = (pred_binary > 0) & (target_binary == 0)
    neg = np.zeros((h, w), np.float32)
    pos = np.zeros((h, w), np.float32)

    pred_labels, _ = ndimage_label(pred_binary, structure=_S4)

    fn_labels, n_fn = ndimage_label(false_neg, structure=_S4)
    for cid in range(1, n_fn + 1):
        comp = (fn_labels == cid)
        adj = np.unique(pred_labels[ndi_binary_dilation(comp, structure=_S4) & ~comp])
        if len(adj[adj > 0]) >= 2:
            neg[comp] = 1.0

    fp_labels, n_fp = ndimage_label(false_pos, structure=_S4)
    for cid in range(1, n_fp + 1):
        comp = (fp_labels == cid)
        nb = ndi_binary_dilation(comp, structure=_S4) & ~comp & (pred_binary > 0)
        if nb.sum() < 2:
            continue
        without = pred_binary.copy()
        without[comp] = 0
        labels_without, _ = ndimage_label(without, structure=_S4)
        new = np.unique(labels_without[nb])
        if len(new[new > 0]) > 1:
            pos[comp] = 1.0

    return neg, pos


def _neuron(res, rng):
    """A soma plus six wandering thin branches -- the shape that matters here.

    Thin branches are what fragment into many small error components at high
    resolution, which is exactly the regime that made the old code slow.
    """
    m = np.zeros((res, res), np.float32)
    s = res / 1024.0
    cy, cx = rng.randint(int(0.3 * res), int(0.7 * res), 2)
    r = max(1, int(40 * s))
    m[cy - r:cy + r, cx - r:cx + r] = 1
    for _ in range(6):
        a = rng.rand() * 6.28
        y, x = float(cy), float(cx)
        for _ in range(int(250 * s)):
            y += np.sin(a)
            x += np.cos(a)
            a += rng.randn() * 0.05
            yi, xi = int(y), int(x)
            if 0 <= yi < res - 2 and 0 <= xi < res - 2:
                m[yi:yi + 2, xi:xi + 2] = 1
    return m


def _corrupt(tgt, rng, noise_p):
    """Turn a ground-truth mask into a plausible prediction: speckle + breaks."""
    pred = tgt.copy()
    flip = rng.rand(*tgt.shape) < noise_p
    pred[flip] = 1 - pred[flip]
    ys, xs = np.where(tgt > 0)
    if len(ys):
        for _ in range(5):
            k = rng.randint(len(ys))
            pred[max(0, ys[k] - 2):ys[k] + 2, max(0, xs[k] - 2):xs[k] + 2] = 0
    return pred


@pytest.mark.parametrize("res", [32, 64, 128, 256, 512])
def test_matches_wholeframe_reference_on_neurons(res):
    rng = np.random.RandomState(res)
    for trial in range(6):
        tgt = _neuron(res, rng)
        pred = _corrupt(tgt, rng, rng.choice([0.0, 0.001, 0.005, 0.02]))
        if trial % 2 == 0:
            # Components touching the frame edge exercise the box clipping.
            pred[0, :] = 1
            pred[:, -1] = 1
        p, t = pred.astype(np.uint8), tgt.astype(np.uint8)
        got_neg, got_pos = _detect_critical_components_2d(p, t)
        exp_neg, exp_pos = _reference(p, t)
        assert np.array_equal(got_neg, exp_neg), f"neg differs at res={res} trial={trial}"
        assert np.array_equal(got_pos, exp_pos), f"pos differs at res={res} trial={trial}"


@pytest.mark.parametrize("density", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_matches_reference_on_random_masks(density):
    """Pure noise maximises component count and adjacency weirdness.

    Neuron-shaped inputs are the realistic case but a forgiving one; random
    masks are where a bounding-box shortcut would break if it were wrong.
    """
    rng = np.random.RandomState(int(density * 100))
    for _ in range(5):
        p = (rng.rand(64, 64) < density).astype(np.uint8)
        t = (rng.rand(64, 64) < density).astype(np.uint8)
        got_neg, got_pos = _detect_critical_components_2d(p, t)
        exp_neg, exp_pos = _reference(p, t)
        assert np.array_equal(got_neg, exp_neg)
        assert np.array_equal(got_pos, exp_pos)


@pytest.mark.parametrize("pred_val,tgt_val", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_degenerate_masks(pred_val, tgt_val):
    """Empty and full masks must not crash and must match the reference."""
    p = np.full((48, 48), pred_val, np.uint8)
    t = np.full((48, 48), tgt_val, np.uint8)
    got_neg, got_pos = _detect_critical_components_2d(p, t)
    exp_neg, exp_pos = _reference(p, t)
    assert np.array_equal(got_neg, exp_neg)
    assert np.array_equal(got_pos, exp_pos)


def test_bridge_is_detected():
    """A one-pixel gap splitting a bar is the case the loss exists to catch.

    Guards against a bounding-box bug that silently finds nothing: an
    all-zeros return would pass every equivalence test above if the reference
    were broken the same way, so assert the positive behaviour directly.
    """
    tgt = np.zeros((32, 32), np.uint8)
    tgt[15:17, 5:27] = 1                      # a horizontal bar
    pred = tgt.copy()
    pred[15:17, 15:17] = 0                    # snip it in the middle
    neg, _ = _detect_critical_components_2d(pred, tgt)
    assert neg.sum() > 0, "gap that splits the prediction was not flagged critical"


def test_scales_subquadratically_with_resolution():
    """Cost must not blow up when topology_resolution rises 256 -> 512.

    The old code was ~16x here for a 4x pixel increase, because both the
    component count and the per-component cost scaled with area. The bound is
    loose (6x) so this asserts the algorithmic class, not a machine speed.
    """
    def _time(res):
        rng = np.random.RandomState(0)
        cases = []
        for _ in range(4):
            tgt = _neuron(res, rng)
            cases.append((_corrupt(tgt, rng, 0.004).astype(np.uint8), tgt.astype(np.uint8)))
        _detect_critical_components_2d(*cases[0])          # warm import/JIT paths
        t0 = time.perf_counter()
        for p, t in cases:
            _detect_critical_components_2d(p, t)
        return time.perf_counter() - t0

    t256, t512 = _time(256), _time(512)
    assert t512 < max(6.0 * t256, 0.05), (
        f"512 cost {t512:.3f}s vs 256 cost {t256:.3f}s "
        f"({t512 / max(t256, 1e-9):.1f}x) — per-component work is scaling with frame area"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
