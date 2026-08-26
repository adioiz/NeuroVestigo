#!/usr/bin/env python3
"""
run_cellpose.py -- Cellpose-SAM (cpsam) competitor runner.

Run with the isolated env:
    benchmark/competitors/cp_venv/bin/python benchmark/run_cellpose.py

Cellpose-SAM is AUTOMATIC and single-frame: it segments every cell in each
frame with no prompt. To score it on our single-target-neuron task we apply the
documented fair protocol for auto-segmenters -- INSTANCE MATCHING: for each
frame, take the union of predicted instances that overlap the target neuron's
ground truth (the most generous, strawman-proof reduction; genuine gaps between
fragments still show up in clDice). This is the standard way to evaluate a
multi-object segmenter on a single-object benchmark.

Predicted target stacks -> benchmark/predictions/cellpose_sam/<name>.tif
(uint8, 1 = target neuron). Score afterwards with benchmark/score.py (main venv).
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
OUT = os.path.join(REPO, "benchmark", "predictions", "cellpose_sam")
VAL = ["A91_C1_2", "A31_C1", "A52_C2", "A22_C2", "A42_C1"]   # matches Dataset().split("val")


def load_frames(name: str, limit=None):
    files = sorted(glob.glob(os.path.join(DATA, "images", name, "*.png")))
    if limit:
        files = files[:limit]
    return [np.array(Image.open(f).convert("L")) for f in files]     # grayscale [H,W]


def load_gt(name: str, n: int):
    files = sorted(glob.glob(os.path.join(DATA, "masks", name, "*.png")))[:n]
    return [np.array(Image.open(f)) > 0 for f in files]              # bool [H,W]


def match_to_target(inst_mask: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Union of predicted instances that overlap the target-neuron GT."""
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
    ap.add_argument("--videos", nargs="*", default=VAL)
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--scale", type=float, default=2.0,
                    help="upscale factor: our ~1.5 px/um neurons are small enough that cpsam "
                         "under-detects them at native scale (0 cells on many frames). x2 is "
                         "its favorable setting, selected by a sweep on the VALIDATION split "
                         "only (x1/x1.5/x2/x3 -> clDice .461/.368/.473/.425) and then applied "
                         "unchanged to test. Never tuned on test data.")
    ap.add_argument("--cellprob", type=float, default=-1.0,
                    help="cellprob_threshold. Lower = more permissive. -1.0 was "
                         "selected by a sweep on the VALIDATION split only "
                         "(0.0/-1/-2/-3 at scale x2 -> IoU .381/.472/.286/.105) "
                         "and applied unchanged to test. The cellpose default of "
                         "0.0 discards our faint 1-2 px neurites.")
    ap.add_argument("--flow", type=float, default=1.5,
                    help="flow_threshold. cellpose DISCARDS masks whose flow error "
                         "exceeds this; thin neurites produce flow fields unlike the "
                         "blobby cells cpsam was trained on, so the default 0.4 threw "
                         "away valid detections (0 instances on 37/72 frames of "
                         "A71_C1). Swept on the VALIDATION split only "
                         "(0.4/0.8/1.0/1.5/2.0/3.0/0.0 -> IoU .440/.518/.563/.564/"
                         ".564/.564/.564, empty frames 3/40 -> 0/40); the metric "
                         "plateaus from 1.0 up, 1.5 selected, applied unchanged to "
                         "test. See benchmark/sweep_cellpose_flow.py.")
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args()

    import cv2
    from cellpose import models
    import torch
    gpu = torch.backends.mps.is_available() or torch.cuda.is_available()
    print(f"cellpose gpu={gpu} scale=x{args.scale} cellprob={args.cellprob} flow={args.flow}")
    model = models.CellposeModel(gpu=gpu)       # cpsam is the built-in model in cellpose v4
    os.makedirs(args.out_dir, exist_ok=True)

    for v in args.videos:
        frames = load_frames(v, args.limit_frames)
        gts = load_gt(v, len(frames))
        print(f"  {v}: {len(frames)} frames (scale x{args.scale}) ...", flush=True)
        s = args.scale
        up = [cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR) if s != 1.0 else f
              for f in frames]
        masks, _, _ = model.eval(up, batch_size=8, cellprob_threshold=args.cellprob,
                                 flow_threshold=args.flow)     # list of instance-label maps at upscaled res
        # downscale instance maps back to native resolution for matching/scoring
        masks = [cv2.resize(np.asarray(m).astype(np.int32), (f.shape[1], f.shape[0]),
                            interpolation=cv2.INTER_NEAREST) for m, f in zip(masks, frames)]
        target = np.stack([match_to_target(m, g).astype(np.uint8)
                           for m, g in zip(masks, gts)])
        out_path = os.path.join(args.out_dir, f"{v}.tif")
        tifffile.imwrite(out_path, target)
        # also save the RAW instance-label stack (all detected cells) for figures
        raw_dir = args.out_dir + "_raw"; os.makedirs(raw_dir, exist_ok=True)
        tifffile.imwrite(os.path.join(raw_dir, f"{v}.tif"),
                         np.stack([np.asarray(m).astype(np.int32) for m in masks]))
        ninst = np.mean([len(np.unique(m)) - 1 for m in masks])
        print(f"    -> {out_path}  (avg {ninst:.1f} instances/frame, "
              f"target fg={float((target>0).mean()):.4f})")


if __name__ == "__main__":
    main()
