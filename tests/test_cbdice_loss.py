"""Regression tests for the cbDice loss port (src/loss_fns.py).

Guards the two bugs fixed when porting to the reference implementation
(github.com/PengchengShi1220/cbDice, github.com/jocpae/clDice):

  Bug 1 - the topology-precision/sensitivity ratios were inert: ``w_tprec``
          was identically 1.0 and missing regions never lowered ``w_tsens``,
          so the loss barely moved while Dice fell by 0.3+.
  Bug 2 - ``soft_skel`` eroded before measuring and skipped clDice's k=0 seed,
          so any structure <=2 px wide produced an empty skeleton.

Run: ``PYTHONPATH=. pytest tests/test_cbdice_loss.py``
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loss_fns import (  # noqa: E402
    SoftCBDiceLoss,
    soft_skel,
    _cbdice_get_weights,
    _combine_tensors,
)


# --------------------------------------------------------------------------- #
# Synthetic neuron: a soma disk with axis-aligned branches. Branches are
# axis-aligned so a "severed" case can cut a clean full-width gap.
# --------------------------------------------------------------------------- #
_SEG = {
    "right": (slice(63, 66), slice(64, 105)),
    "left": (slice(63, 66), slice(23, 64)),
    "up": (slice(23, 64), slice(63, 66)),
    "down": (slice(64, 105), slice(63, 66)),
}


def _make(branches, sever=False):
    H = W = 128
    m = np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[:H, :W]
    m[(yy - 64) ** 2 + (xx - 64) ** 2 <= 8 ** 2] = 1  # soma
    for b in branches:
        rs, cs = _SEG[b]
        m[rs, cs] = 1
    if sever:  # full-width 12 px gap in the right branch
        m[63:66, 84:96] = 0
    return torch.from_numpy(m)


_TARGET = _make(["right", "left", "up"])


def _logits(mask):
    """High-confidence logits from a mask: sigmoid -> ~0.9975 / ~0.0025."""
    return mask * 12.0 - 6.0


def _terms(pred_mask, target_mask):
    """White-box: return (w_tprec, w_tsens) exactly as the loss computes them."""
    y_pred_prob = torch.sigmoid(_logits(pred_mask)).unsqueeze(0)
    y_true = target_mask.unsqueeze(0)
    with torch.no_grad():
        yph = (y_pred_prob > 0.5).float()
        skp = soft_skel(yph.unsqueeze(1), 5).squeeze(1)
        skt = soft_skel(y_true.unsqueeze(1), 5).squeeze(1)
    skpp = skp * y_pred_prob
    q_vl, q_slvl, q_sl = _cbdice_get_weights(y_true, skt, 2, prob_flag=False)
    q_vp, q_spvp, q_sp = _cbdice_get_weights(y_pred_prob, skpp, 2, prob_flag=True)
    wp = (torch.sum(q_sp * q_vl) + 1.0) / (torch.sum(_combine_tensors(q_spvp, q_slvl, q_sp)) + 1.0)
    ws = (torch.sum(q_sl * q_vp) + 1.0) / (torch.sum(_combine_tensors(q_slvl, q_spvp, q_sl)) + 1.0)
    return float(wp), float(ws)


def _loss(pred_mask):
    return float(SoftCBDiceLoss(num_iter=5)(_logits(pred_mask), _TARGET))


# --------------------------------------------------------------------------- #
# Criterion 1: w_tprec is no longer constant — it falls when the prediction
# contains a branch absent from the target (a false/invented structure).
# --------------------------------------------------------------------------- #
def test_tprec_falls_on_invented_branch():
    wp_perfect, _ = _terms(_make(["right", "left", "up"]), _TARGET)
    wp_invented, _ = _terms(_make(["right", "left", "up", "down"]), _TARGET)
    assert wp_perfect > 0.999, wp_perfect          # perfect prediction -> ~1
    assert wp_invented < 0.95, wp_invented         # invented branch -> drops
    assert wp_invented < wp_perfect - 0.02


# --------------------------------------------------------------------------- #
# Criterion 2: w_tsens falls when the target contains a branch the prediction
# misses.
# --------------------------------------------------------------------------- #
def test_tsens_falls_on_missing_branch():
    _, ws_perfect = _terms(_make(["right", "left", "up"]), _TARGET)
    _, ws_missing = _terms(_make(["right", "left"]), _TARGET)
    assert ws_perfect > 0.999, ws_perfect
    assert ws_missing < 0.9, ws_missing
    assert ws_missing < ws_perfect - 0.05


# --------------------------------------------------------------------------- #
# Criterion 3: the loss is strictly ordered by topological damage.
# --------------------------------------------------------------------------- #
def test_loss_strictly_ordered():
    perfect = _loss(_make(["right", "left", "up"]))
    severed = _loss(_make(["right", "left", "up"], sever=True))
    missing_one = _loss(_make(["right", "left"]))
    missing_two = _loss(_make(["right"]))
    assert perfect < 1e-4, perfect
    assert perfect < severed < missing_one < missing_two, (
        perfect, severed, missing_one, missing_two,
    )


# --------------------------------------------------------------------------- #
# Criterion 4: thin (1 px, 2 px) structures yield a non-empty soft skeleton
# (the k=0 seed term). Widths >=3 were already fine.
# --------------------------------------------------------------------------- #
def test_soft_skel_thin_bars_non_empty():
    for width in (1, 2, 3):
        bar = torch.zeros(1, 1, 40, 40)
        bar[0, 0, 18:18 + width, 4:36] = 1.0
        assert soft_skel(bar, 5).sum().item() > 0.0, width


# --------------------------------------------------------------------------- #
# The loss must remain differentiable w.r.t. the prediction logits.
# --------------------------------------------------------------------------- #
def test_gradient_flows():
    logits = _logits(_make(["right", "left"])).clone().requires_grad_(True)
    SoftCBDiceLoss(num_iter=5)(logits, _TARGET).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


if __name__ == "__main__":
    test_tprec_falls_on_invented_branch()
    test_tsens_falls_on_missing_branch()
    test_loss_strictly_ordered()
    test_soft_skel_thin_bars_non_empty()
    test_gradient_flows()
    print("all cbDice regression tests passed")
