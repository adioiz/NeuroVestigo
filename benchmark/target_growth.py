#!/usr/bin/env python3
"""target_growth.py -- how the target itself changes through a recording.

The drift analysis (Sec. 4.2) measures how much cbDice falls from the first
quarter of a recording to the last. Part of that fall is the model losing the
target; part is the target becoming objectively harder to segment, because the
neuron grows. This quantifies the second part so the two can be told apart.

Reports, per quartile of the propagated frames of each held-out recording:
  area          foreground pixels
  skeleton      centreline length (what cbDice normalises by)
  thickness     area / skeleton length
  components    connected components of the ground truth

    python benchmark/target_growth.py
"""
from __future__ import annotations
import glob, os
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from scipy import ndimage as ndi

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST = ["A71_C1", "A81_C1", "A12_C3_100k", "A12_C1_100k_2"]


def main():
    Q = [[] for _ in range(4)]
    for v in TEST:
        gts = [np.array(Image.open(f)) > 0
               for f in sorted(glob.glob(os.path.join(REPO, "training_data", "masks", v, "*.png")))][1:]
        n = len(gts)
        for i in range(4):
            for m in gts[i * n // 4:(i + 1) * n // 4]:
                sk = skeletonize(m)
                Q[i].append((m.sum(), sk.sum(), m.sum() / max(sk.sum(), 1),
                             ndi.label(m)[0].max()))
    print(f"{'':<5}{'area px':>10}{'skeleton px':>13}{'thickness px':>14}{'components':>12}")
    print("-" * 54)
    for i, q in enumerate(Q):
        a = np.array(q, float)
        print(f"Q{i+1:<4}{a[:,0].mean():>10.0f}{a[:,1].mean():>13.0f}"
              f"{a[:,2].mean():>14.1f}{a[:,3].mean():>12.2f}")
    a1, a4 = np.array(Q[0], float), np.array(Q[3], float)
    print("-" * 54)
    print(f"Q1 -> Q4:  area x{a4[:,0].mean()/a1[:,0].mean():.2f}   "
          f"skeleton x{a4[:,1].mean()/a1[:,1].mean():.2f}   "
          f"thickness x{a4[:,2].mean()/a1[:,2].mean():.2f}")
    print("\nThe target lengthens and thins, so the task itself gets harder. The three\n"
          "automatic methods carry no temporal state, so their decay bounds how much\n"
          "of the fall is difficulty rather than drift.")


if __name__ == "__main__":
    main()
