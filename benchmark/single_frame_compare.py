#!/usr/bin/env python3
"""
single_frame_compare.py -- per-frame mask comparison on RANDOM frames.

Everything runs AS DESIGNED: ours uses its normal temporal propagation (its real
predicted mask at that frame); the single-frame auto-segmenters (µSAM, Cellpose-SAM)
produce their normal single-frame masks. We just sample a few random frames and put
the real masks side by side, so you can see per-frame who tracks the neuron.

    venv/bin/python benchmark/single_frame_compare.py

Reads only saved predictions (no model needed). Output ->
paper_assets/figures/single_frame_compare.png
"""
from __future__ import annotations

import os
import random

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "training_data")
PRED = os.path.join(REPO, "benchmark", "predictions")
FIGDIR = os.path.join(REPO, "paper_assets", "figures")

# six models (ours first). SAM 2 = zero-shot propagation; µSAM = box-prompt;
# Cellpose-SAM / Omnipose / NeuroQuantify = automatic (shown as raw all-cell output).
MODELS = [("ours", "Ours"),
          ("sam2_zeroshot", "SAM 2\n(zero-shot)"),
          ("usam", "µSAM\n(box-prompt)"),
          ("cellpose_sam", "Cellpose-SAM\n(auto)"),
          ("omnipose", "Omnipose\n(auto)"),
          ("neuroquantify", "NeuroQuantify\n(phase-contrast)")]
POOL = ["A71_C1", "A81_C1", "A12_C3_100k", "A12_C1_100k_2", "A31_C1", "A52_C2", "A22_C2"]


def load_gray(v, t):
    return np.array(Image.open(os.path.join(DATA, "images", v, f"{t:05d}.png")).convert("L"))


def load_gt(v, t):
    return np.array(Image.open(os.path.join(DATA, "masks", v, f"{t:05d}.png"))) > 0


def overlay(gray, gt, pred):
    g = gray.astype(float) / 255.0
    rgb = np.stack([g] * 3, -1) * 0.65
    tp, fn, fp = gt & pred, gt & ~pred, pred & ~gt
    rgb[tp] = [1.0, 0.95, 0.2]; rgb[fn] = [0.15, 0.9, 0.3]; rgb[fp] = [1.0, 0.25, 0.2]
    return np.clip(rgb, 0, 1)


def iou(p, g):
    u = (p | g).sum()
    return 1.0 if u == 0 else float((p & g).sum() / u)


# alternative frame sets to choose between (ours strong, clear gap, neurites visible)
PRESETS = {
    "v1": [("A71_C1", 42), ("A52_C2", 8), ("A12_C1_100k_2", 46), ("A42_C1", 11)],
    "v2": [("A71_C1", 42), ("A81_C1", 13), ("A31_C1", 40), ("A22_C2", 12)],
    "v3": [("A52_C2", 2), ("A12_C1_100k_2", 41), ("A42_C1", 37), ("A81_C1", 11)],
    "v4": [("A31_C1", 45), ("A22_C2", 28), ("A12_C3_100k", 3), ("A52_C2", 5)],
    # TEST-ONLY: every frame from a genuinely held-out video (never trained or selected on)
    "test": [("A71_C1", 64), ("A81_C1", 13), ("A12_C1_100k_2", 52), ("A71_C1", 41)],
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v1", choices=list(PRESETS))
    ap.add_argument("--style", choices=("target", "raw"), default="raw",
                    help="target = TP/FN/FP overlay + IoU for EVERY method; "
                         "raw = automatic tools shown as native multi-instance output")
    ap.add_argument("--suffix", default="", help="appended to the output filename")
    args = ap.parse_args()
    samples = PRESETS[args.preset]
    AUTO = {"cellpose_sam", "omnipose", "neuroquantify"}   # automatic: shown as RAW all-cell output
    preds, raw = {m: {} for m, _ in MODELS}, {m: {} for m in AUTO}
    for m, _ in MODELS:
        for v, _t in samples:
            if v not in preds[m]:
                preds[m][v] = tifffile.imread(os.path.join(PRED, m, f"{v}.tif")) > 0
    for m in AUTO:
        for v, _t in samples:
            if v not in raw[m]:
                raw[m][v] = tifffile.imread(os.path.join(PRED, f"{m}_raw", f"{v}.tif"))

    ncol = 2 + len(MODELS)
    fig, axes = plt.subplots(len(samples), ncol, figsize=(2.35 * ncol, 2.5 * len(samples)))
    if len(samples) == 1:
        axes = axes[None, :]
    titles = ["Input frame", "Ground truth"] + [lbl for _, lbl in MODELS]
    for j, tt in enumerate(titles):
        is_ours = (j == 2)
        axes[0, j].set_title(tt, fontsize=10.5, pad=6,
                             color="#16324F" if j < 3 else "#8B96A3",
                             fontweight="bold" if is_ours else "normal")
    for i, (v, t) in enumerate(samples):
        gray = load_gray(v, t); gt = load_gt(v, t)
        ys, xs = np.where(gt)
        y0, y1 = max(0, ys.min() - 15), min(gt.shape[0], ys.max() + 15)
        x0, x1 = max(0, xs.min() - 15), min(gt.shape[1], xs.max() + 15)
        crop = lambda a: a[y0:y1, x0:x1]
        axes[i, 0].imshow(crop(gray), cmap="gray")
        axes[i, 0].set_ylabel(f"{v}\nframe {t}", fontsize=9, color="#1F3A5F")
        axes[i, 1].imshow(crop(overlay(gray, gt, np.zeros_like(gt))))
        from skimage.color import label2rgb
        from scipy import ndimage as ndi
        for k, (m, _) in enumerate(MODELS):
            ax = axes[i, 2 + k]
            if m in AUTO and args.style == "raw":   # native multi-instance output
                r = raw[m][v][t] if t < raw[m][v].shape[0] else np.zeros(gt.shape, int)
                # NeuroQuantify raw is a semantic map (soma/neurite); show connected regions.
                disp = ndi.label(r > 0)[0] if m == "neuroquantify" else r
                ax.imshow(label2rgb(crop(disp), image=crop(gray).astype(float) / 255.0,
                                    bg_label=0, alpha=0.6))
                n = int(disp.max()) if m == "neuroquantify" else len(np.unique(r)) - 1
                ax.text(0.03, 0.05, f"{n} regions", transform=ax.transAxes,
                        fontsize=9, color="white", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#1F3A5F", ec="none", alpha=0.85))
            else:                               # target overlay (methods that isolate one object)
                pr = preds[m][v][t] if t < preds[m][v].shape[0] else np.zeros_like(gt)
                ax.imshow(crop(overlay(gray, gt, pr)))
                ax.text(0.03, 0.05, f"IoU {iou(pr, gt):.2f}", transform=ax.transAxes,
                        fontsize=9.5, color="white", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#1F3A5F", ec="none", alpha=0.85))
        for j in range(ncol):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            for sp in axes[i, j].spines.values():
                if j == 2:                      # frame the Ours column
                    sp.set_color("#16324F"); sp.set_linewidth(2.0)
                else:
                    sp.set_visible(False)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(fc=[1, .95, .2], label="correct (TP)"),
                        Patch(fc=[.15, .9, .3], label="missed (FN)"),
                        Patch(fc=[1, .25, .2], label="extra (FP)")],
               loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.03))
    if args.style == "raw":
        fig.text(0.5, -0.055, "Automatic tools (Cellpose-SAM, Omnipose, NeuroQuantify) shown as raw "
             "output — each colour is a separately detected region; they segment the whole field "
             "(or shatter it), never isolating the target neuron the way ours does.",
             ha="center", fontsize=8.5, color="#5A6472", style="italic")
    # The title has to match the style: in 'raw' the automatic tools show their
    # native whole-field output, in 'target' every method is reduced to the
    # target neuron and scored identically.
    title = ("Each method run its own way — ours isolates the target neuron; "
             "automatic tools segment the whole field") if args.style == "raw" else \
            ("Every method scored the same way on the same frames — "
             "green is the branch structure each one loses")
    fig.suptitle(title, fontsize=13, color="#16324F", y=1.006, weight="bold")
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    suffix = "" if args.preset == "v1" else f"_{args.preset}"
    suffix = suffix + args.suffix
    out = os.path.join(FIGDIR, f"single_frame_compare{suffix}.png")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", out, "| samples:", samples)


if __name__ == "__main__":
    main()
