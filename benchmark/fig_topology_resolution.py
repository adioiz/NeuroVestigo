#!/usr/bin/env python3
"""fig_topology_resolution.py -- what topology_resolution actually does.

SoftCBDiceLoss resizes its inputs to a SQUARE topology_resolution x
topology_resolution grid whenever either side is larger. This renders the same
ground-truth mask as the metric saw it at 256 and at 512, and reports
cbDice(gt, gt) -- which must be 1.0, since the mask is being compared to itself.

    KMP_DUPLICATE_LIB_OK=TRUE python benchmark/fig_topology_resolution.py
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
import torch, torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import distance_transform_edt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from benchmark.eval_masks import cbdice_score

FIGDIR = os.path.join(REPO, "paper_assets", "figures")
NAVY, BLUE, RED, GREY = "#16324F", "#1F6FB2", "#C1443C", "#8A94A2"
mpl.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 10, "figure.dpi": 120})

VIDEO, FRAME = "A81_C1", 65          # 307x485 -- the worst aspect-ratio case


def through_resize(mask: np.ndarray, res: int) -> np.ndarray:
    """Exactly what the loss does: square resize if either side exceeds res."""
    H, W = mask.shape
    if H <= res and W <= res:
        return mask.copy()
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    small = F.interpolate(t, size=(res, res), mode="nearest")
    back = F.interpolate(small, size=(H, W), mode="nearest")
    return back[0, 0].numpy() > 0.5


def main():
    f = sorted(glob.glob(os.path.join(REPO, "training_data", "masks", VIDEO, "*.png")))[FRAME]
    gt = np.array(Image.open(f)) > 0
    H, W = gt.shape

    # crop on the thinnest structure present (small EDT, far from the soma)
    edt = distance_transform_edt(gt)
    thin = gt & (edt <= 2.0)
    ys, xs = np.where(thin)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    r = 55
    y0, y1 = max(0, cy - r), min(H, cy + r)
    x0, x1 = max(0, cx - r), min(W, cx + r)

    panels = [("native (no resize)", gt, None),
              ("as the loss saw it at\ntopology_resolution=256", through_resize(gt, 256), 256),
              ("as the loss sees it at\ntopology_resolution=512", through_resize(gt, 512), 512)]

    fig, axes = plt.subplots(2, 3, figsize=(10.6, 8.4),
                             gridspec_kw=dict(height_ratios=[1.35, 1]))
    for j, (lbl, m, res) in enumerate(panels):
        lost = gt & ~m
        for i, ax in enumerate(axes[:, j]):
            sl = (slice(None), slice(None)) if i == 0 else (slice(y0, y1), slice(x0, x1))
            ax.imshow(np.zeros_like(m[sl], float), cmap="Greys", vmin=0, vmax=1)
            rgb = np.zeros((*m[sl].shape, 4))
            rgb[m[sl]] = mpl.colors.to_rgba(NAVY)
            rgb[lost[sl]] = mpl.colors.to_rgba(RED)
            ax.imshow(rgb, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GREY); s.set_linewidth(0.8)
            if i == 0:
                ax.set_title(lbl, fontsize=10, color=NAVY, weight="bold", pad=8)
                n = int(lost.sum())
                sub = (f"grid {W}x{H} — unchanged" if res is None
                       else f"grid {W}x{H} -> {res}x{res}"
                            + (f"\n{n} px of neurite erased" if n else "\nnothing erased"))
                ax.set_xlabel(sub, fontsize=8.6,
                              color=RED if (res and lost.sum()) else GREY, labelpad=6)
            else:
                ax.set_xlabel("zoom on a 1–2 px neurite", fontsize=8.2,
                              color=GREY, labelpad=5)
        s = cbdice_score(m, gt) if res is None else _score_at(gt, res)
        axes[0, j].text(0.5, -0.20, f"cbDice(gt, gt) = {s:.4f}",
                        transform=axes[0, j].transAxes, ha="center", fontsize=11,
                        weight="bold", color=NAVY if abs(s - 1) < 1e-4 else RED)

    fig.suptitle("topology_resolution: the grid cbDice is computed on\n"
                 "Comparing a mask to ITSELF must score 1.0 — at 256 it did not",
                 fontsize=12.5, color=NAVY, weight="bold", y=0.99)
    fig.text(0.5, 0.015, "red = foreground present in the ground truth but destroyed by the resize. "
             "A 307x485 mask is squashed to a 256x256 SQUARE,\nso thin processes fall below one "
             "pixel and vanish. At 512 no resize happens and the mask is untouched.",
             ha="center", fontsize=8.6, color=GREY, style="italic")
    fig.tight_layout(rect=[0, 0.045, 1, 0.945])
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "fig_topology_resolution.png")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", out)


def _score_at(gt, res):
    """cbDice(gt,gt) with the loss configured at `res` (what the metric reported)."""
    from src.loss_fns import SoftCBDiceLoss
    fn = SoftCBDiceLoss(num_iter=5, topology_resolution=res).eval()
    t = torch.from_numpy(gt.astype(np.float32))[None, None]
    with torch.no_grad():
        return 1.0 - float(fn(t * 20.0 - 10.0, t))


def fig_floor():
    """Loss on a PERFECT prediction vs topology_resolution. Should be 0 everywhere."""
    from src.loss_fns import SoftCBDiceLoss
    cases = [("A81_C1", 65), ("A31_C1", 10), ("A71_C1", 10), ("A52_C2", 40)]
    res = [256, 384, 512, 1024]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for (v, t) in cases:
        f = sorted(glob.glob(os.path.join(REPO, "training_data", "masks", v, "*.png")))[t]
        g = np.array(Image.open(f)) > 0
        g1024 = F.interpolate(torch.from_numpy(g.astype(np.float32))[None, None],
                              size=(1024, 1024), mode="nearest")
        ys = []
        for r in res:
            fn = SoftCBDiceLoss(num_iter=5, topology_resolution=r).eval()
            with torch.no_grad():
                ys.append(float(fn(g1024 * 20.0 - 10.0, g1024)))
        ax.plot(res, ys, "o-", lw=1.8, ms=5, label=f"{v} ({g.shape[1]}x{g.shape[0]})")
    ax.axhline(0, color=NAVY, lw=1.4, ls="--")
    ax.text(1024, 0.004, "correct: a perfect prediction costs nothing",
            ha="right", fontsize=8.4, color=NAVY, style="italic")
    ax.axvspan(240, 400, color=RED, alpha=0.07)
    ax.text(320, 0.072, "training used 256", ha="center", fontsize=8.6,
            color=RED, weight="bold")
    ax.set_xscale("log", base=2); ax.set_xticks(res)
    ax.set_xticklabels([str(r) for r in res])
    ax.set_xlabel("topology_resolution (grid the cbDice loss is computed on)")
    ax.set_ylabel("cbDice loss on a PERFECT prediction")
    ax.set_ylim(-0.004, 0.085)
    ax.grid(color="#E4E8ED", lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8.4, loc="upper right")
    ax.set_title("The training noise floor — error the optimiser could never remove",
                 loc="left", fontsize=11, color=NAVY, weight="bold", pad=10)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_topology_floor.png")
    fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
    fig_floor()
