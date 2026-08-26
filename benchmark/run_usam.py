#!/usr/bin/env python3
"""
run_usam.py -- micro-sam (µSAM) competitor runner, used AS INTENDED.

Run with the isolated env:
    benchmark/competitors/usam_venv/bin/python benchmark/run_usam.py --videos ...

µSAM is fundamentally an INTERACTIVE tool: you show it where the object is and it
segments it. Its automatic (AIS/AMG) modes are built for compact cells and simply
fail to detect thin neurons -- using them is a strawman. So we use µSAM the way it
is meant to segment a known target: prompt it with the target's bounding box on each
frame (its interactive upper bound; the box is derived from GT, i.e. oracle
localization, exactly as µSAM's own paper evaluates interactive prompts).

NOTE: micro_sam.segment_from_box expects the box as [y0, x0, y1, x1] (row-col),
NOT [x0, y0, x1, y1]. Getting this wrong makes it segment the wrong region (~0 IoU).

Predicted target stacks -> benchmark/predictions/usam/<name>.tif
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import tifffile
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "training_data")
OUT = os.path.join(REPO, "benchmark", "predictions", "usam")


def load_frames(name, limit=None):
    files = sorted(glob.glob(os.path.join(DATA, "images", name, "*.png")))
    if limit:
        files = files[:limit]
    return [np.array(Image.open(f).convert("L")) for f in files]


def load_gt(name, n):
    files = sorted(glob.glob(os.path.join(DATA, "masks", name, "*.png")))[:n]
    return [np.array(Image.open(f)) > 0 for f in files]


def gt_box_yxyx(mask, pad=6):
    ys, xs = np.where(mask)
    H, W = mask.shape
    return np.array([max(0, ys.min() - pad), max(0, xs.min() - pad),
                     min(H, ys.max() + pad), min(W, xs.max() + pad)])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--model", default="vit_l_lm",
                    help="micro-SAM backbone. vit_l_lm selected by a sweep on the "
                         "VALIDATION split only (vit_b_lm .424 vs vit_l_lm .559 IoU) "
                         "and applied unchanged to test; vit_h_lm is not in the "
                         "micro-SAM registry. Running the smallest variant would "
                         "understate the strongest prompted competitor.")
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args()

    import torch
    from micro_sam.util import get_sam_model
    from micro_sam.prompt_based_segmentation import segment_from_box
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"micro-sam model={args.model} device={device} (interactive box-prompt)")
    predictor = get_sam_model(model_type=args.model, device=device)
    os.makedirs(args.out_dir, exist_ok=True)

    for v in args.videos:
        frames = load_frames(v, args.limit_frames)
        gts = load_gt(v, len(frames))
        print(f"  {v}: {len(frames)} frames ...", flush=True)
        target = []
        for fr, g in zip(frames, gts):
            if not g.any():
                target.append(np.zeros(g.shape, np.uint8)); continue
            predictor.set_image(np.stack([fr] * 3, -1))
            res = segment_from_box(predictor, gt_box_yxyx(g),
                                   multimask_output=True, return_all=True)
            masks, scores = np.asarray(res[0]), np.asarray(res[1])
            masks = masks.reshape(-1, *g.shape)
            m = masks[int(np.argmax(scores))].astype(np.uint8)   # µSAM's own best-scoring mask
            target.append(m)
        target = np.stack(target)
        out_path = os.path.join(args.out_dir, f"{v}.tif")
        tifffile.imwrite(out_path, target)
        print(f"    -> {out_path}  (target fg={float((target>0).mean()):.4f})")


if __name__ == "__main__":
    main()
