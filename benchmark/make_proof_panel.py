#!/usr/bin/env python3
"""
make_proof_panel.py -- visual proof: real input frames, real GT, and each model's
REAL predicted mask, overlaid so quality is directly visible.

Overlay colours (per pixel, vs ground truth):
    yellow = correct (predicted AND ground truth)
    green  = MISSED  (ground truth, not predicted)  <- fragmented/lost neurite
    red    = extra   (predicted,  not ground truth)

    python benchmark/make_proof_panel.py --video A31_C1 --frames 8 35 62

Output -> paper_assets/figures/proof_<video>.png
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "training_data")
PRED = os.path.join(REPO, "benchmark", "predictions")
FIGDIR = os.path.join(REPO, "paper_assets", "figures")

PRETTY = {"ours": "NeuroVestigo", "sam2_zeroshot": "SAM 2 (zero-shot)",
          "usam": "µSAM (box-prompt)", "cellpose_sam": "Cellpose-SAM",
          "omnipose": "Omnipose", "neuroquantify": "NeuroQuantify",
          "sam3.1": "SAM 3.1 (point-prompt)",
          "sam3.1_text": "SAM 3.1 (text: \"cells\")"}
# Model key -> predictions subdirectory, where they differ. The SAM 3 sweep
# wrote one directory per configuration, so the winning run does not carry the
# name it is scored under.
PRED_DIR = {"sam3.1": "sam3_final", "sam3.1_text": "sam3_text_final"}


def load_gray(video, t):
    p = os.path.join(DATA, "images", video, f"{t:05d}.png")
    return np.array(Image.open(p).convert("L")).astype(float) / 255.0


def load_gt(video, t):
    p = os.path.join(DATA, "masks", video, f"{t:05d}.png")
    return np.array(Image.open(p)) > 0


def overlay(gray, gt, pred):
    rgb = np.stack([gray] * 3, -1) * 0.65
    tp, fn, fp = gt & pred, gt & ~pred, pred & ~gt
    rgb[tp] = [1.0, 0.95, 0.2]     # yellow = correct
    rgb[fn] = [0.15, 0.9, 0.3]     # green  = missed (lost neurite)
    rgb[fp] = [1.0, 0.25, 0.2]     # red    = extra
    return np.clip(rgb, 0, 1)


def bbox(masks, margin=12):
    ys, xs = np.where(np.any(masks, axis=0))
    if len(ys) == 0:
        return None
    H, W = masks[0].shape
    return (max(0, ys.min() - margin), min(H, ys.max() + margin),
            max(0, xs.min() - margin), min(W, xs.max() + margin))


def iou(pred, gt):
    u = (pred | gt).sum()
    return 1.0 if u == 0 else (pred & gt).sum() / u


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[8, 35, 62])
    ap.add_argument("--models", nargs="+",
                    default=["ours", "sam2_zeroshot", "usam", "sam3.1",
                             "cellpose_sam", "omnipose", "neuroquantify"])
    args = ap.parse_args()

    models = [m for m in args.models
              if os.path.exists(os.path.join(PRED, PRED_DIR.get(m, m), f"{args.video}.tif"))]
    preds = {m: tifffile.imread(os.path.join(PRED, PRED_DIR.get(m, m), f"{args.video}.tif")) > 0
             for m in models}
    gts = {t: load_gt(args.video, t) for t in args.frames}
    y0, y1, x0, x1 = bbox([gts[t] for t in args.frames])

    ncol = 2 + len(models)                    # input | GT | each model
    nrow = len(args.frames)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.6 * nrow))
    if nrow == 1:
        axes = axes[None, :]

    ours_col = 2 + (models.index("ours") if "ours" in models else 0)
    col_titles = ["Input frame", "Ground truth"] + [PRETTY.get(m, m) for m in models]
    for j, title in enumerate(col_titles):
        is_ours = (j == ours_col)
        axes[0, j].set_title(title, fontsize=10.5, pad=6,
                             color="#16324F" if (j < 2 or is_ours) else "#8B96A3",
                             fontweight="bold" if is_ours else "normal")

    for i, t in enumerate(args.frames):
        gray = load_gray(args.video, t)
        gt = gts[t]
        crop = lambda a: a[y0:y1, x0:x1]
        axes[i, 0].imshow(crop(gray), cmap="gray");
        axes[i, 0].set_ylabel(f"frame {t}", fontsize=10, color="#1F3A5F")
        axes[i, 1].imshow(crop(overlay(gray, gt, np.zeros_like(gt))))
        for k, m in enumerate(models):
            pr = preds[m][t] if t < preds[m].shape[0] else np.zeros_like(gt)
            axes[i, 2 + k].imshow(crop(overlay(gray, gt, pr)))
            axes[i, 2 + k].text(0.03, 0.05, f"IoU {iou(pr, gt):.2f}",
                                transform=axes[i, 2 + k].transAxes, fontsize=9,
                                color="white", va="bottom", fontweight="bold" if (2 + k == ours_col) else "normal",
                                bbox=dict(boxstyle="round,pad=0.2", fc="#16324F", ec="none", alpha=0.85))
        for j in range(ncol):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            for sp in axes[i, j].spines.values():
                if j == ours_col:                 # frame the Ours column
                    sp.set_color("#16324F"); sp.set_linewidth(2.0)
                else:
                    sp.set_visible(False)

    # legend
    from matplotlib.patches import Patch
    leg = [Patch(fc=[1, 0.95, 0.2], label="correct (TP)"),
           Patch(fc=[0.15, 0.9, 0.3], label="missed neurite (FN)"),
           Patch(fc=[1, 0.25, 0.2], label="extra (FP)")]
    fig.legend(handles=leg, loc="lower center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Qualitative comparison on a held-out neuron ({args.video})",
                 fontsize=12.5, color="#16324F", y=1.008, weight="bold")
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, f"proof_{args.video}.png")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", out, "| models:", ", ".join(models))


if __name__ == "__main__":
    main()
