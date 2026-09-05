#!/usr/bin/env python
"""Visual comparison: NeuroVestigo trained on 20 recordings vs 22.

The two models differ by ~0.003 IoU on the test set, which is the size of a
difference this benchmark cannot resolve. A bar chart of the means would
therefore be misleading -- it would show a clean win. This plots the
per-frame difference instead, where the oscillation about zero is the point:
the models disagree in both directions on every recording.

    venv/bin/python benchmark/compare_20_vs_22.py
"""
from __future__ import annotations

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from benchmark.eval_masks import cbdice_score, dice_score, iou_score  # noqa: E402

TEST = ["A71_C1", "A81_C1", "A12_C3_100k", "A12_C1_100k_2"]
OLD, NEW = "ours_16f", "ours_22rec"
FIGDIR = os.path.join(REPO, "paper_assets", "figures")
NAVY, OURBLUE, CRIMSON = "#16324F", "#1F6FB2", "#C0392B"
GRID, MUTE = "#E8ECF1", "#8B96A3"

mpl.rcParams.update({"font.size": 9, "axes.spines.top": False,
                    "axes.spines.right": False, "axes.edgecolor": "#5A6472"})


def gt(seq, t):
    m = np.array(Image.open(os.path.join(REPO, "training_data", "masks", seq,
                                         f"{t:05d}.png")))
    if m.ndim == 3:
        m = m[..., 0]
    labels = [v for v in np.unique(m) if v != 0]
    return m == max(labels, key=lambda v: (m == v).sum())


def curves(seq, fn):
    a = tifffile.imread(os.path.join(REPO, "benchmark", "predictions", NEW, f"{seq}.tif"))
    b = tifffile.imread(os.path.join(REPO, "benchmark", "predictions", OLD, f"{seq}.tif"))
    n, o = [], []
    for t in range(1, a.shape[0]):
        g = gt(seq, t)
        n.append(fn(a[t] > 0, g))
        o.append(fn(b[t] > 0, g))
    return np.array(n), np.array(o)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 5.6), sharey=True)
    alld = []
    for ax, seq in zip(axes.ravel(), TEST):
        n, o = curves(seq, iou_score)
        d = n - o
        alld.append(d)
        x = np.arange(1, len(d) + 1)
        ax.axhline(0, color=MUTE, lw=0.9, zorder=2)
        ax.fill_between(x, 0, d, where=d >= 0, color=OURBLUE, alpha=0.55,
                        interpolate=True, lw=0, zorder=3)
        ax.fill_between(x, 0, d, where=d < 0, color=CRIMSON, alpha=0.5,
                        interpolate=True, lw=0, zorder=3)
        ax.set_title(f"{seq}   mean {d.mean():+.4f}   "
                     f"{int((d > 0).sum())} up / {int((d < 0).sum())} down",
                     fontsize=9, color=NAVY, loc="left", pad=6)
        ax.grid(color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        ax.set_xlim(1, len(d))
    for ax in axes[1]:
        ax.set_xlabel("frame (propagated from the frame-0 annotation)")
    for ax in axes[:, 0]:
        ax.set_ylabel("IoU: 22 recordings $-$ 20 recordings")

    d = np.concatenate(alld)
    fig.suptitle("Per-frame difference between the 22-recording and "
                 "20-recording models on the held-out test set",
                 fontsize=11, color=NAVY, x=0.02, ha="left", weight="bold")
    fig.text(0.02, -0.015,
             f"Blue: the 22-recording model scores higher; red: lower. "
             f"Over all {len(d)} frames the mean difference is {d.mean():+.4f} IoU, "
             f"with {int((d > 0).sum())} frames up and {int((d < 0).sum())} down, "
             f"and it changes sign within every recording.",
             fontsize=8.5, color="#5A6472", style="italic")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "compare_20_vs_22.png")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", out)
    print(f"  all frames: mean {d.mean():+.5f}  median {np.median(d):+.5f}  "
          f"up {int((d > 0).sum())} down {int((d < 0).sum())}")


if __name__ == "__main__":
    main()
