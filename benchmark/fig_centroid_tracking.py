#!/usr/bin/env python3
"""fig_centroid_tracking.py -- what centroid displacement looks like.

INTERNAL/explanatory. The centroid is the centre of mass of a mask: average
the coordinates of every foreground pixel. Comparing the predicted centroid
with the ground-truth one asks WHERE the mask sits, not what shape it is --
a question none of IoU/Dice/cbDice/clDice answers.

Top row: ours. Bottom row: base SAM 2, the only other method that propagates
from a single frame-0 prompt. Same frames, same scale.

    KMP_DUPLICATE_LIB_OK=TRUE python benchmark/fig_centroid_tracking.py
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage as ndi

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
FIGDIR = os.path.join(REPO, "paper_assets", "figures")
NAVY, GREEN, RED, GREY = "#16324F", "#2CA6A4", "#C1443C", "#8A94A2"
CASES = [("A71_C1", 20), ("A71_C1", 55), ("A81_C1", 40), ("A12_C3_100k", 60)]


def load(v, t):
    fr = np.array(Image.open(sorted(glob.glob(f"{REPO}/training_data/images/{v}/*.png"))[t]).convert("L"))
    g = np.array(Image.open(sorted(glob.glob(f"{REPO}/training_data/masks/{v}/*.png"))[t])) > 0
    return fr, g


def main():
    rows = [("ours", "Ours"), ("sam2_zeroshot", "SAM 2 (zero-shot)")]
    fig, axes = plt.subplots(2, len(CASES), figsize=(3.1 * len(CASES), 6.6))
    for r, (m, lbl) in enumerate(rows):
        for c, (v, t) in enumerate(CASES):
            ax = axes[r, c]
            fr, g = load(v, t)
            p = tifffile.imread(f"{REPO}/benchmark/predictions/{m}/{v}.tif")[t] > 0
            ax.imshow(fr, cmap="gray")
            # masks as translucent fills
            for msk, col in ((g, GREEN), (p, NAVY)):
                rgba = np.zeros((*msk.shape, 4))
                rgba[msk] = [*matplotlib.colors.to_rgb(col), 0.32]
                ax.imshow(rgba)
            cg = ndi.center_of_mass(g)
            cp = ndi.center_of_mass(p) if p.any() else cg
            d = float(np.linalg.norm(np.array(cp) - np.array(cg)))
            ax.plot([cg[1], cp[1]], [cg[0], cp[0]], "-", color=RED, lw=1.4, zorder=5)
            ax.plot(cg[1], cg[0], "o", ms=7, mfc=GREEN, mec="white", mew=1.2, zorder=6)
            ax.plot(cp[1], cp[0], "o", ms=7, mfc=NAVY, mec="white", mew=1.2, zorder=6)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(GREY); sp.set_linewidth(0.8)
            ax.text(0.03, 0.03, f"{d:.1f} px", transform=ax.transAxes, fontsize=11,
                    color="white", weight="bold", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.28",
                              fc=NAVY if d < 3 else RED, ec="none", alpha=0.9))
            if r == 0:
                ax.set_title(f"{v}  t={t}", fontsize=9.5, color=NAVY, pad=6)
            if c == 0:
                ax.set_ylabel(lbl, fontsize=11, color=NAVY, weight="bold")
    fig.suptitle("Centroid displacement — where the mask sits, not what shape it is\n"
                 "green = ground truth, navy = prediction, dot = centre of mass, "
                 "red line = displacement",
                 fontsize=12.5, color=NAVY, weight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIGDIR, "_internal_centroid_tracking.png")
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
