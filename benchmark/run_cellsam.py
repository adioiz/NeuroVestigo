#!/usr/bin/env python3
"""
run_cellsam.py -- CellSAM competitor runner.

Run with the isolated env:
    benchmark/competitors/cellsam_venv/bin/python benchmark/run_cellsam.py --videos ...

CellSAM (SAM ViT-B + CellFinder detector) is AUTOMATIC and single-frame. Same
fair protocol as Cellpose/uSAM: segment every cell per frame, then instance-match
(union of predicted instances overlapping the target neuron's GT).

Predicted target stacks -> benchmark/predictions/cellsam/<name>.tif
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
OUT = os.path.join(REPO, "benchmark", "predictions", "cellsam")


def load_frames(name, limit=None):
    files = sorted(glob.glob(os.path.join(DATA, "images", name, "*.png")))
    if limit:
        files = files[:limit]
    return [np.array(Image.open(f).convert("L")) for f in files]


def load_gt(name, n):
    files = sorted(glob.glob(os.path.join(DATA, "masks", name, "*.png")))[:n]
    return [np.array(Image.open(f)) > 0 for f in files]


def match_to_target(inst_mask, gt):
    out = np.zeros(gt.shape, bool)
    if not gt.any():
        return out
    for lab in np.unique(inst_mask):
        if lab == 0:
            continue
        m = inst_mask == lab
        if np.logical_and(m, gt).sum() > 0:
            out |= m
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--device", default=None, help="cpu/mps/cuda (default: auto)")
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args()

    import torch
    from cellSAM import segment_cellular_image, get_model
    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"cellSAM device={device}")
    model = get_model()          # downloads cellsam_general on first use
    model = model.to(device).eval()
    os.makedirs(args.out_dir, exist_ok=True)

    for v in args.videos:
        frames = load_frames(v, args.limit_frames)
        gts = load_gt(v, len(frames))
        print(f"  {v}: {len(frames)} frames ...", flush=True)
        target, ninst, raw = [], [], []
        for fr, g in zip(frames, gts):
            try:
                mask, _, _ = segment_cellular_image(fr, model=model, normalize=True, device=device)
                mask = np.asarray(mask).astype(np.int32)
            except Exception as e:
                print(f"    [warn] frame failed ({e}); empty mask")
                mask = np.zeros(g.shape, np.int32)
            ninst.append(len(np.unique(mask)) - 1)
            raw.append(mask)
            target.append(match_to_target(mask, g).astype(np.uint8))
        target = np.stack(target)
        out_path = os.path.join(args.out_dir, f"{v}.tif")
        tifffile.imwrite(out_path, target)
        # also save the RAW instance-label stack (all detected cells) for figures
        raw_dir = args.out_dir + "_raw"; os.makedirs(raw_dir, exist_ok=True)
        tifffile.imwrite(os.path.join(raw_dir, f"{v}.tif"), np.stack(raw))
        print(f"    -> {out_path}  (avg {np.mean(ninst):.1f} instances/frame, "
              f"target fg={float((target>0).mean()):.4f})")


if __name__ == "__main__":
    main()
