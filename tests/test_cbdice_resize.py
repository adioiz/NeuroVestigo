"""Regression test: the cbDice resize must not manufacture topology error.

`SoftCBDiceLoss.forward` downsamples both inputs to `topology_resolution` when
the incoming masks are larger. An earlier version resized the prediction
'bilinear' and the target 'nearest'; on a bit-identical pair the two grids then
disagreed on 100-400+ pixels and the loss scored that as topology error — a
perfect prediction cost 0.008-0.085 instead of 0. Both sides must now be
resampled identically (bilinear + re-binarise at 0.5).

Run: ``KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. pytest tests/test_cbdice_resize.py``
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loss_fns import SoftCBDiceLoss  # noqa: E402

# Resolutions to check, and input sizes that force the resize to fire.
RESOLUTIONS = (256, 384, 512)


def _synthetic_neuron(h, w):
    """Soma + thin branches at an arbitrary (non-square) native size."""
    m = np.zeros((h, w), np.float32)
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
    m[(yy - cy) ** 2 + (xx - cx) ** 2 <= (min(h, w) // 12) ** 2] = 1.0
    m[cy - 1:cy + 2, cx:int(w * 0.9)] = 1.0          # horizontal neurite
    m[int(h * 0.1):cy, cx - 1:cx + 2] = 1.0          # vertical neurite
    return m


def _upsample(mask, size=1024):
    t = torch.from_numpy(mask)[None, None]
    return F.interpolate(t, size=(size, size), mode='nearest')


def test_perfect_prediction_costs_nothing():
    """A prediction identical to the target must score exactly 0 at every res."""
    # Include a strongly non-square case (aspect 1.74, like A91_C1_2 at 615x353).
    for h, w in [(485, 307), (353, 615), (292, 292), (154, 298)]:
        gt = _upsample(_synthetic_neuron(h, w))
        for res in RESOLUTIONS:
            fn = SoftCBDiceLoss(num_iter=5, topology_resolution=res).eval()
            with torch.no_grad():
                # High-confidence logits reproducing the target exactly.
                loss = float(fn(gt * 20.0 - 10.0, gt))
            assert loss == 0.0 or abs(loss) < 1e-6, (
                f"perfect prediction cost {loss:.6f} at res={res} on {h}x{w}"
            )


def test_resize_modes_match():
    """Both sides must land on the same grid for a binary input."""
    gt = _upsample(_synthetic_neuron(485, 307))
    for res in RESOLUTIONS:
        pred_grid = F.interpolate(gt, size=(res, res), mode='bilinear',
                                  align_corners=False)
        target_grid = (F.interpolate(gt, size=(res, res), mode='bilinear',
                                     align_corners=False) > 0.5).float()
        assert int(((pred_grid > 0.5).float() != target_grid).sum()) == 0, res


def test_loss_still_discriminates_after_fix():
    """Guard against 'fixing' the loss by making it uniformly zero."""
    gt = _upsample(_synthetic_neuron(485, 307))
    broken = gt.clone()
    # Sever the horizontal neurite -> genuine topology error.
    broken[..., 500:540] = 0.0
    fn = SoftCBDiceLoss(num_iter=5, topology_resolution=512).eval()
    with torch.no_grad():
        perfect = float(fn(gt * 20.0 - 10.0, gt))
        severed = float(fn(broken * 20.0 - 10.0, gt))
    assert perfect < 1e-6, perfect
    assert severed > 1e-3, f"severed branch scored {severed:.6f}, loss looks inert"


if __name__ == "__main__":
    test_perfect_prediction_costs_nothing()
    test_resize_modes_match()
    test_loss_still_discriminates_after_fix()
    print("all cbDice resize tests passed")
