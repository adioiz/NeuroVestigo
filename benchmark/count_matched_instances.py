#!/usr/bin/env python3
"""count_matched_instances.py -- how many predicted instances the matching rule
reassembles into one target mask, per frame.

The automatic methods segment every object in the field. To score them on a
single-target task each is credited with the UNION of every predicted instance
overlapping the target's ground truth (Sec. 4.1). This counts how many
instances that union actually contains: a value near 1 means the method found
the neuron as one object, a large value means it fragmented it and the
matching rule put the pieces back together.

Reads the *_raw prediction stacks, which hold the native instance-label maps.

    python benchmark/count_matched_instances.py
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
import tifffile
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST = ["A71_C1", "A81_C1", "A12_C3_100k", "A12_C1_100k_2"]
AUTO = [("omnipose", "Omnipose"), ("cellpose_sam", "Cellpose-SAM"),
        ("neuroquantify", "NeuroQuantify")]


def main():
    print(f"{'method':<16}{'detected/frame':>16}{'merged into target':>21}{'median':>9}{'frames':>8}")
    print("-" * 70)
    for m, lbl in AUTO:
        raw_dir = os.path.join(REPO, "benchmark", "predictions", f"{m}_raw")
        if not os.path.isdir(raw_dir):
            print(f"{lbl:<16}  (no raw stack saved)"); continue
        det, mrg = [], []
        for v in TEST:
            p = os.path.join(raw_dir, f"{v}.tif")
            if not os.path.exists(p):
                continue
            raw = tifffile.imread(p)
            gts = [np.array(Image.open(f)) > 0
                   for f in sorted(glob.glob(os.path.join(REPO, "training_data", "masks", v, "*.png")))]
            for t in range(1, min(len(raw), len(gts))):
                inst, g = raw[t], gts[t]
                if not g.any():
                    continue
                labs = [l for l in np.unique(inst) if l]
                det.append(len(labs))
                mrg.append(sum(1 for l in labs if np.logical_and(inst == l, g).any()))
        mrg = np.array(mrg)
        print(f"{lbl:<16}{np.mean(det):>16.1f}{np.mean(mrg):>21.1f}"
              f"{int(np.median(mrg)):>9}{len(mrg):>8}")


if __name__ == "__main__":
    main()
