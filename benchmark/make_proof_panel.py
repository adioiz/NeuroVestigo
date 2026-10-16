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
import os

import sys

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "training_data")
PRED = os.path.join(REPO, "benchmark", "predictions")
FIGDIR = os.path.join(REPO, "paper_assets", "figures")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import held_out  # noqa: E402

PRETTY = {"ours": "NeuroVestigo", "sam2_zeroshot": "SAM 2 (zero-shot)",
          "usam": "µSAM (box-prompt)", "cellpose_sam": "Cellpose-SAM",
          "omnipose": "Omnipose", "neuroquantify": "NeuroQuantify",
          "sam3_maskfirst": "SAM 3 (mask-prompt)",
          "sam3.1_text": "SAM 3.1 (text: \"cells\")",
          # model-vs-model variants, for internal comparison panels
          "ours_16f": "NeuroVestigo (20 rec)",
          "ours_22rec": "NeuroVestigo (22 rec)",
          "ours_23rec": "NeuroVestigo (23 rec, IoU-best)",
          "ours_23rec_topo": "NeuroVestigo (23 rec, topo)"}
# Model key -> predictions subdirectory, where they differ. The SAM 3 sweep
# wrote one directory per configuration, so the winning run does not carry the
# name it is scored under.
PRED_DIR = {"sam3_maskfirst": "sam3_maskfirst", "sam3.1_text": "sam3_text_final",
            "usam": "usam_single", "ours": "ours_fold2_topo"}


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


def bbox(masks, margin=12, min_aspect=0.0):
    """Crop around everything the panels draw, widened to `min_aspect`.

    A neuron that grows down the frame gives a crop much taller than it is
    wide; nine such columns make an unreadably tall figure. Widening the crop
    adds background on either side -- it never removes a pixel of the target.
    """
    ys, xs = np.where(np.any(masks, axis=0))
    if len(ys) == 0:
        return None
    H, W = masks[0].shape
    y0, y1 = max(0, ys.min() - margin), min(H, ys.max() + margin)
    x0, x1 = max(0, xs.min() - margin), min(W, xs.max() + margin)
    want = int(round(min_aspect * (y1 - y0)))
    if want > x1 - x0:
        pad = want - (x1 - x0)
        x0, x1 = max(0, x0 - pad // 2), min(W, x1 + (pad - pad // 2))
        if x1 - x0 < want:                      # ran into an edge: take it back
            if x0 == 0:
                x1 = min(W, want)
            else:
                x0 = max(0, W - want)
    return y0, y1, x0, x1


def iou(pred, gt):
    u = (pred | gt).sum()
    return 1.0 if u == 0 else (pred & gt).sum() / u


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[8, 35, 62])
    ap.add_argument("--models", nargs="+",
                    default=["ours", "sam2_zeroshot", "usam", "sam3_maskfirst",
                             "cellpose_sam", "omnipose", "neuroquantify"])
    ap.add_argument("--suffix", default="",
                    help="appended to the output filename, so a comparison "
                         "panel cannot overwrite the paper's proof figure")
    ap.add_argument("--min-aspect", type=float, default=0.0,
                    help="widen the crop with background until it is at least "
                         "this wide relative to its height (0 = tight bbox)")
    ap.add_argument("--ours-dir", default=None,
                    help="predictions subdirectory for the 'ours' column "
                         f"(default {PRED_DIR['ours']}); a recording may only "
                         "be drawn with a model that held it out")
    args = ap.parse_args()
    if args.ours_dir:
        PRED_DIR["ours"] = args.ours_dir

    if "ours" in args.models:
        held_out.require(args.video, PRED_DIR["ours"])
    models = [m for m in args.models
              if os.path.exists(os.path.join(PRED, PRED_DIR.get(m, m), f"{args.video}.tif"))]
    preds = {m: tifffile.imread(os.path.join(PRED, PRED_DIR.get(m, m), f"{args.video}.tif")) > 0
             for m in models}
    gts = {t: load_gt(args.video, t) for t in args.frames}
    y0, y1, x0, x1 = bbox([gts[t] for t in args.frames], min_aspect=args.min_aspect)

    ncol = 2 + len(models)                    # input | GT | each model
    nrow = len(args.frames)
    # Panels are as wide as the crop is: a neuron that grows down the frame
    # gives a portrait crop, and a fixed square panel would pad every column
    # with white instead of showing the cell larger.
    aspect = (x1 - x0) / (y1 - y0)
    panel_h = 2.6
    panel_w = min(2.5, max(1.0, panel_h * aspect))
    fig, axes = plt.subplots(nrow, ncol, figsize=(panel_w * ncol, panel_h * nrow))
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
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Qualitative comparison on a held-out neuron ({args.video})",
                 fontsize=12.5, color="#16324F", y=1.008, weight="bold")
    legend_h = 0.42 / (panel_h * nrow)        # keep the legend clear of row 1
    fig.tight_layout(rect=(0, legend_h, 1, 1))
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, f"proof_{args.video}{args.suffix}.png")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", out, "| models:", ", ".join(models))


if __name__ == "__main__":
    main()
