#!/usr/bin/env python3
"""
run_trackastra.py -- Trackastra seg+track pipeline competitor (Gallusser & Weigert,
ECCV 2024; winner of the 7th Cell Tracking Challenge, ISBI 2024).

Trackastra is a TRACKER, not a segmenter: it links a segmenter's per-frame instance
masks into temporally-consistent tracks. It represents the modular SOTA approach --
"segment every frame independently, then link" -- which is the natural alternative to
our JOINT segment-and-track-with-memory. Its weakness for our task is structural: each
frame is segmented independently, so there is no temporal memory to recover a neurite
the segmenter drops on a given frame. We pair it with the automatic segmenter that best
detects our neurons (its most favorable input) and follow the frame-0 target track --
the SAME frame-0-seed protocol as ours.

Pipeline: imgs (T,Y,X) + segmenter raw instance masks (predictions/<seg>_raw/<v>.tif)
-> Trackastra linking -> track-relabeled masks -> follow the track(s) overlapping the
target GT at frame 0 -> binary target-track stack, scored like the other competitors.

Run with the isolated conda env:
    /opt/homebrew/Caskroom/miniconda/base/bin/conda run -n trackastra \
        python benchmark/run_trackastra.py --videos ... --seg omnipose

Predicted target stacks -> benchmark/predictions/trackastra_<seg>/<name>.tif
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
PRED = os.path.join(REPO, "benchmark", "predictions")


def load_frames(name, limit=None):
    files = sorted(glob.glob(os.path.join(DATA, "images", name, "*.png")))
    if limit:
        files = files[:limit]
    return np.stack([np.array(Image.open(f).convert("L")) for f in files])   # (T,Y,X)


def load_gt(name, n):
    files = sorted(glob.glob(os.path.join(DATA, "masks", name, "*.png")))[:n]
    return np.stack([np.array(Image.open(f)) > 0 for f in files])            # (T,Y,X) bool


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--seg", default="omnipose",
                    help="segmenter whose raw instance masks feed Trackastra "
                         "(reads predictions/<seg>_raw/<v>.tif): omnipose|cellpose_sam|deepsea")
    ap.add_argument("--model", default="general_2d",
                    help="Trackastra pretrained model (general_2d|ctc|general_2d_w_SAM2_features)")
    ap.add_argument("--mode", default="greedy", choices=["greedy", "greedy_nodiv", "ilp"])
    ap.add_argument("--limit-frames", type=int, default=None)
    args = ap.parse_args()

    import torch
    from trackastra.model import Trackastra
    device = "cuda" if torch.cuda.is_available() else "cpu"   # trackastra: MPS unsupported -> cpu
    model = Trackastra.from_pretrained(args.model, device=device)
    print(f"trackastra model={args.model} seg={args.seg} device={device}")

    out_dir = os.path.join(PRED, f"trackastra_{args.seg}")
    os.makedirs(out_dir, exist_ok=True)

    for v in args.videos:
        imgs = load_frames(v, args.limit_frames)
        gts = load_gt(v, imgs.shape[0])
        seg_path = os.path.join(PRED, f"{args.seg}_raw", f"{v}.tif")
        masks = tifffile.imread(seg_path).astype(np.int32)[:imgs.shape[0]]     # (T,Y,X) instance
        print(f"  {v}: {imgs.shape[0]} frames, seg has "
              f"{np.mean([len(np.unique(m))-1 for m in masks]):.0f} inst/frame ...", flush=True)

        # Trackastra: link per-frame instances -> track-relabeled masks (consistent IDs over t)
        track_graph, track_masks = model.track(imgs, masks, mode=args.mode)
        track_masks = np.asarray(track_masks)

        # Identify the target track(s): track IDs overlapping the target GT on the first
        # frame that has GT foreground (frame-0 seed, same as ours).
        seed = next((t for t in range(gts.shape[0]) if gts[t].any()), 0)
        seed_ids = set(np.unique(track_masks[seed][gts[seed]])) - {0}
        # follow those track IDs across all frames
        out = np.stack([np.isin(track_masks[t], list(seed_ids)) if seed_ids
                        else np.zeros(gts.shape[1:], bool) for t in range(gts.shape[0])])
        tifffile.imwrite(os.path.join(out_dir, f"{v}.tif"), out.astype(np.uint8))
        # identity continuity: fraction of t>=1 frames where the target track is still present
        cont = float(np.mean([out[t].any() for t in range(1, out.shape[0])]))
        print(f"    -> {out_dir}/{v}.tif  (seed frame {seed}, {len(seed_ids)} track(s), "
              f"target present in {cont*100:.0f}% of t>=1 frames)")


if __name__ == "__main__":
    main()
