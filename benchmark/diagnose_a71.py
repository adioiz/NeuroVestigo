#!/usr/bin/env python3
"""Why does Cellpose-SAM detect almost nothing on A71_C1?

Two hypotheses:
  H1  A71 is out of distribution for cpsam (contrast / intensity statistics
      differ from the other recordings) -- a genuine domain failure, and a
      legitimate result to report.
  H2  cpsam DOES find cells on A71, but they never overlap the target neuron,
      so instance-matching returns empty -- a protocol artefact on our side.

Prints per-recording image statistics, then for the frames where the matched
target is empty, reports what cpsam actually found and where.

    benchmark/competitors/cp_venv/bin/python benchmark/diagnose_a71.py
"""
from __future__ import annotations
import glob, os
import numpy as np
import tifffile
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "training_data")
PRED = os.path.join(REPO, "benchmark", "predictions")
TEST = ["A71_C1", "A81_C1", "A12_C3_100k", "A12_C1_100k_2"]


def main():
    print("=== image statistics per recording (are they comparable?) ===")
    print(f"{'recording':<16}{'mean':>8}{'std':>8}{'p1':>6}{'p99':>6}"
          f"{'GT px':>8}{'GT/frame %':>12}{'HxW':>12}")
    for v in TEST:
        fs = sorted(glob.glob(os.path.join(DATA, "images", v, "*.png")))
        gs = sorted(glob.glob(os.path.join(DATA, "masks", v, "*.png")))
        ims = np.stack([np.array(Image.open(f).convert("L")) for f in fs[::10]])
        gt = np.stack([np.array(Image.open(g)) > 0 for g in gs[::10]])
        print(f"{v:<16}{ims.mean():>8.1f}{ims.std():>8.1f}{np.percentile(ims,1):>6.0f}"
              f"{np.percentile(ims,99):>6.0f}{gt.sum(axis=(1,2)).mean():>8.0f}"
              f"{100*gt.mean():>11.2f}%{str(ims.shape[1:]):>12}")

    print("\n=== on frames where the matched target is EMPTY, what did cpsam find? ===")
    for v in TEST:
        tgt = tifffile.imread(os.path.join(PRED, "cellpose_sam", f"{v}.tif"))
        raw = tifffile.imread(os.path.join(PRED, "cellpose_sam_raw", f"{v}.tif"))
        gs = sorted(glob.glob(os.path.join(DATA, "masks", v, "*.png")))
        empty = [t for t in range(1, len(tgt)) if not tgt[t].any()]
        if not empty:
            print(f"{v:<16} no empty frames")
            continue
        n_found, overlap_any, dist = [], 0, []
        for t in empty:
            r = raw[t]
            labs = [l for l in np.unique(r) if l]
            n_found.append(len(labs))
            g = np.array(Image.open(gs[t])) > 0
            if not labs or not g.any():
                continue
            gy, gx = np.array(np.where(g)).mean(1)
            for l in labs:
                y, x = np.array(np.where(r == l)).mean(1)
                dist.append(np.hypot(y - gy, x - gx))
            if any(np.logical_and(r == l, g).any() for l in labs):
                overlap_any += 1
        print(f"{v:<16} {len(empty):>3} empty frames | cpsam found "
              f"{np.mean(n_found):.1f} regions/frame on them | "
              f"{overlap_any} of those frames had ANY overlap with GT | "
              f"median centroid distance to GT {np.median(dist) if dist else float('nan'):.0f} px")

    print("\nH2 would show as: many regions found, far from the GT centroid.")
    print("H1 would show as: ~0 regions found at all.")


if __name__ == "__main__":
    main()
