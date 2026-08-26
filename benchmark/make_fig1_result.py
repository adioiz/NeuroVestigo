#!/usr/bin/env python3
"""Regenerate fig1_result.png -- the small tracked-neuron panel in Figure 1(d).

Single frame, overlay semantics identical to make_proof_panel.py:
    yellow = correct, green = missed ground truth, red = extra prediction.
Output is written at exactly the size the TikZ figure assumes (771x573), so
the Figure 1 layout comments stay valid.

    python benchmark/make_fig1_result.py --video A71_C1 --frame 64
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import tifffile
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from benchmark.make_proof_panel import load_gray, load_gt, overlay, bbox  # noqa: E402

OUT_W, OUT_H = 771, 573


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default="A71_C1")
    ap.add_argument("--frame", type=int, default=64)
    ap.add_argument("--pred-dir", default=os.path.join(REPO, "benchmark", "predictions", "ours"))
    ap.add_argument("--out", default=os.path.join(REPO, "paper_assets", "figures", "fig1_result.png"))
    a = ap.parse_args()

    gray = load_gray(a.video, a.frame)
    gt = load_gt(a.video, a.frame)
    pred = tifffile.imread(os.path.join(a.pred_dir, f"{a.video}.tif"))[a.frame].astype(bool)

    y0, y1, x0, x1 = bbox([gt, pred], margin=14)
    rgb = overlay(gray, gt, pred)[y0:y1, x0:x1]

    im = Image.fromarray((rgb * 255).astype(np.uint8))
    # letterbox to the exact size Figure 1 assumes, preserving aspect ratio
    im.thumbnail((OUT_W, OUT_H), Image.LANCZOS)
    canvas = Image.new("RGB", (OUT_W, OUT_H), (10, 12, 16))
    canvas.paste(im, ((OUT_W - im.width) // 2, (OUT_H - im.height) // 2))
    canvas.save(a.out)

    inter = (gt & pred).sum()
    union = (gt | pred).sum()
    print(f"wrote {a.out}  ({a.video} t={a.frame}, IoU {inter/max(union,1):.3f}, "
          f"size {canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    main()
