#!/usr/bin/env python3
"""tracking_stability.py -- does the propagated mask stay ON the target neuron?

The benchmark metrics score mask quality per frame; they do not say whether the
mask is still on the same cell. This measures that directly, for the only two
methods that actually propagate from a single frame-0 prompt (ours and base
SAM 2). The other methods are given a ground-truth box per frame, or are
reduced by a matching rule that selects regions overlapping the target, so for
them loss of the target is impossible by construction and the question is not
meaningful.

Three criteria, weakest to strongest:
  separated      no overlap at all with the ground truth
  IoU < 0.10     technically overlapping, but effectively off the target
  centroid       distance between predicted and ground-truth centroids

    KMP_DUPLICATE_LIB_OK=TRUE python benchmark/tracking_stability.py
"""
from __future__ import annotations
import glob, os
import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST = ["A71_C1", "A81_C1", "A12_C3_100k", "A12_C1_100k_2"]
PROPAGATING = [("ours", "Ours"), ("sam2_zeroshot", "SAM 2 (zero-shot)")]


def main():
    print(f"{'method':<20}{'separated':>11}{'IoU<0.10':>10}{'centroid px':>14}"
          f"{'>10px':>8}{'>25px':>8}{'frames':>8}")
    print("-" * 79)
    for m, lbl in PROPAGATING:
        sep = low = 0
        d = []
        for v in TEST:
            p = tifffile.imread(os.path.join(REPO, "benchmark", "predictions", m, f"{v}.tif"))
            gts = [np.array(Image.open(f)) > 0
                   for f in sorted(glob.glob(os.path.join(REPO, "training_data", "masks", v, "*.png")))]
            for t in range(1, min(len(p), len(gts))):
                g, pr = gts[t], p[t] > 0
                if not g.any():
                    continue
                if not pr.any() or not (pr & g).any():
                    sep += 1
                    continue
                if (pr & g).sum() / (pr | g).sum() < 0.10:
                    low += 1
                d.append(float(np.linalg.norm(
                    np.array(ndi.center_of_mass(pr)) - np.array(ndi.center_of_mass(g)))))
        d = np.array(d)
        print(f"{lbl:<20}{sep:>11}{low:>10}{np.median(d):>14.1f}"
              f"{100*(d>10).mean():>7.0f}%{100*(d>25).mean():>7.0f}%{len(d):>8}")
    print("\n'separated' and 'IoU<0.10' are counts of frames; centroid is the median\n"
          "over frames, in pixels at the recording's native resolution.")


if __name__ == "__main__":
    main()
