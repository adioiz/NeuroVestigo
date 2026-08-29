#!/usr/bin/env python3
"""overlay_png.py -- prediction-vs-ground-truth overlay, without matplotlib.

make_proof_panel.py builds the publication figure, but it needs matplotlib,
which is deliberately absent from the training/scoring environment. This does
the same visual check with PIL and numpy only, so a suspicious row in the
results table can be inspected wherever the predictions happen to live.

Colours, per pixel, against ground truth:
    yellow = correct   (predicted AND ground truth)
    green  = MISSED    (ground truth, not predicted)
    red    = extra     (predicted, not ground truth)

    python benchmark/overlay_png.py --video A12_C1_100k_2 --frames 10 45 80 \
        --pred ours=ours sam3.1=sam3_final sam3.1_text=sam3_text_final
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import tifffile
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "training_data")
PRED = os.path.join(REPO, "benchmark", "predictions")


def load_gray(video, t):
    p = os.path.join(DATA, "images", video, f"{t:05d}.png")
    return np.array(Image.open(p).convert("L")).astype(np.float32) / 255.0


def load_gt(video, t):
    p = os.path.join(DATA, "masks", video, f"{t:05d}.png")
    return np.array(Image.open(p)) > 0


def bbox(masks, pad=20):
    ys, xs = np.where(np.any(np.stack(masks), axis=0))
    if len(ys) == 0:
        h, w = masks[0].shape
        return 0, h, 0, w
    h, w = masks[0].shape
    return (max(0, ys.min() - pad), min(h, ys.max() + pad + 1),
            max(0, xs.min() - pad), min(w, xs.max() + pad + 1))


def overlay(gray, gt, pred):
    """Grayscale background, then TP/FN/FP painted over it."""
    rgb = np.repeat(gray[..., None], 3, axis=2) * 0.55
    if pred is not None:
        rgb[gt & pred] = [1.0, 0.95, 0.2]     # yellow: correct
        rgb[gt & ~pred] = [0.15, 0.9, 0.3]    # green: missed
        rgb[~gt & pred] = [1.0, 0.2, 0.2]     # red: extra
    else:
        rgb[gt] = [0.15, 0.9, 0.3]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[10, 45, 80])
    ap.add_argument("--pred", nargs="+", required=True,
                    help="label=predictions_subdir, e.g. sam3.1_text=sam3_text_final")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cols = []
    for spec in args.pred:
        label, _, sub = spec.partition("=")
        p = os.path.join(PRED, sub or label, f"{args.video}.tif")
        if not os.path.exists(p):
            print(f"[skip] {label}: no {p}")
            continue
        cols.append((label, tifffile.imread(p) > 0))

    gts = {t: load_gt(args.video, t) for t in args.frames}
    y0, y1, x0, x1 = bbox(list(gts.values()))
    ch, cw = y1 - y0, x1 - x0
    pad, hdr = 6, 22
    ncol, nrow = 1 + len(cols), len(args.frames)

    W = ncol * cw + (ncol + 1) * pad
    H = nrow * ch + (nrow + 1) * pad + hdr
    canvas = Image.new("RGB", (W, H), (18, 18, 18))
    d = ImageDraw.Draw(canvas)

    for ci, name in enumerate(["ground truth"] + [c[0] for c in cols]):
        d.text((pad + ci * (cw + pad) + 4, 5), name, fill=(235, 235, 235))

    for ri, t in enumerate(args.frames):
        g = gts[t]
        gray = load_gray(args.video, t)
        cells = [overlay(gray, g, None)]
        for _, stack in cols:
            pr = stack[t] > 0 if t < len(stack) else np.zeros_like(g)
            cells.append(overlay(gray, g, pr))
        for ci, cell in enumerate(cells):
            im = Image.fromarray(cell[y0:y1, x0:x1])
            canvas.paste(im, (pad + ci * (cw + pad), hdr + pad + ri * (ch + pad)))
        d.text((4, hdr + pad + ri * (ch + pad) + 2), f"t={t}", fill=(200, 200, 90))

    out = args.out or os.path.join(REPO, "paper_assets", "figures",
                                   f"overlay_{args.video}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    canvas.save(out)
    print(f"wrote {out}  ({W}x{H})")
    for label, stack in cols:
        fg = float((stack > 0).mean())
        print(f"  {label:<16} foreground over the whole stack = {fg:.4f}")


if __name__ == "__main__":
    main()
