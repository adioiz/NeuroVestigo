#!/usr/bin/env python3
"""
verify_auto_plot.py -- build the verification figure from the dumped raw/matched
maps (run verify_auto_dump.py first for each model). For each frame shows:
input | GT | ALL raw instances (random colors) | matched target (TP/FN/FP).

    venv/bin/python benchmark/verify_auto_plot.py --model cellpose
    venv/bin/python benchmark/verify_auto_plot.py --model deepsea

Output -> paper_assets/figures/verify_<model>.png
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
from skimage.color import label2rgb

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "training_data")
DUMP = "/tmp/verify"
FIGDIR = os.path.join(REPO, "paper_assets", "figures")
FRAMES = [("A71_C1", 35), ("A81_C1", 44), ("A31_C1", 30), ("A52_C2", 66)]
PRETTY = {"cellpose": "Cellpose-SAM", "deepsea": "DeepSea"}


def gray(v, t):
    return np.array(Image.open(os.path.join(DATA, "images", v, f"{t:05d}.png")).convert("L"))


def gt(v, t):
    return np.array(Image.open(os.path.join(DATA, "masks", v, f"{t:05d}.png"))) > 0


def overlay(g_, gt_, pred):
    im = np.stack([g_.astype(float) / 255.0] * 3, -1) * 0.65
    tp, fn, fp = gt_ & pred, gt_ & ~pred, pred & ~gt_
    im[tp] = [1, .95, .2]; im[fn] = [.15, .9, .3]; im[fp] = [1, .25, .2]
    return np.clip(im, 0, 1)


def iou(p, g):
    u = (p | g).sum(); return 1.0 if u == 0 else (p & g).sum() / u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["cellpose", "deepsea"])
    args = ap.parse_args()
    m = args.model

    ours_dir = os.path.join(REPO, "benchmark", "predictions", "ours")
    fig, axes = plt.subplots(len(FRAMES), 5, figsize=(5 * 2.8, len(FRAMES) * 2.7))
    titles = ["Input frame", "Ground truth", f"{PRETTY[m]}: all instances",
              f"{PRETTY[m]}: matched target", "Ours"]
    for j, tt in enumerate(titles):
        axes[0, j].set_title(tt, fontsize=10, pad=6, color="#16324F",
                             fontweight="bold" if j == 4 else "normal")
    for i, (v, t) in enumerate(FRAMES):
        g = gray(v, t); gmask = gt(v, t)
        raw = tifffile.imread(os.path.join(DUMP, f"{m}_{v}_{t}_raw.tif"))
        matched = tifffile.imread(os.path.join(DUMP, f"{m}_{v}_{t}_matched.tif")) > 0
        ours = tifffile.imread(os.path.join(ours_dir, f"{v}.tif"))[t] > 0
        ys, xs = np.where(gmask)
        y0, y1 = max(0, ys.min() - 25), min(g.shape[0], ys.max() + 25)
        x0, x1 = max(0, xs.min() - 25), min(g.shape[1], xs.max() + 25)
        cr = lambda a: a[y0:y1, x0:x1]
        axes[i, 0].imshow(cr(g), cmap="gray"); axes[i, 0].set_ylabel(f"{v}\nf{t}", fontsize=9, color="#1F3A5F")
        axes[i, 1].imshow(cr(overlay(g, gmask, np.zeros_like(gmask))))
        lab_rgb = label2rgb(cr(raw), image=cr(g).astype(float) / 255.0, bg_label=0, alpha=0.55)
        axes[i, 2].imshow(lab_rgb)
        axes[i, 2].text(0.03, 0.05, f"{len(np.unique(raw))-1} instances",
                        transform=axes[i, 2].transAxes, fontsize=9, color="white", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#1F3A5F", ec="none", alpha=0.85))
        for col, pred, bold in [(3, matched, False), (4, ours, True)]:
            axes[i, col].imshow(cr(overlay(g, gmask, pred)))
            axes[i, col].text(0.03, 0.05, f"IoU {iou(pred, gmask):.2f}",
                              transform=axes[i, col].transAxes, fontsize=9.5, color="white",
                              va="bottom", fontweight="bold" if bold else "normal",
                              bbox=dict(boxstyle="round,pad=0.2", fc="#16324F", ec="none", alpha=0.85))
        for j in range(5):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            for sp in axes[i, j].spines.values():
                if j == 4:
                    sp.set_color("#16324F"); sp.set_linewidth(2.0)
                else:
                    sp.set_visible(False)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(fc=[1, .95, .2], label="correct (TP)"),
                        Patch(fc=[.15, .9, .3], label="missed (FN)"),
                        Patch(fc=[1, .25, .2], label="extra (FP)")],
               loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    note = {"cellpose": "detects the compact soma, misses the thin neurites",
            "deepsea": "over-fragments the field, drops thin neurites"}
    fig.suptitle(f"{PRETTY[m]} (automatic) {note.get(m, '')} — vs Ours (right column)",
                 fontsize=12.5, color="#16324F", y=1.008, weight="bold")
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, f"verify_{m}.png")
    fig.savefig(out, bbox_inches="tight", dpi=190)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
