#!/usr/bin/env python3
"""INTERNAL -- three-way comparison of the training runs.

  A  "inert cbDice"   focal 5 / Dice 2 / cbDice 5 (degenerate) / sv 0.3
  D  "cbDice 0.3"     focal 3 / Dice 2 / cbDice 0.3 (repaired)  / sv 0.3   (epoch 84)
  B  "no cbDice"      focal 5 / Dice 2 /            -           / sv 0.3   (epoch 82)
  C  "cbDice w=1"     focal 5 / Dice 2 / cbDice 1 (repaired)    / sv 0.3   (epoch 72)

Reports, for TEST and VALIDATION separately:
  * per-recording mean clDice (the Table 3 convention) and IoU
  * clDice by quartile of the recording, i.e. drift
  * paired significance of each run against A

Run A is a checkpoint whose cbDice term contributed no gradient, so it is the
closest thing available to an unablated 3-term baseline -- but it is NOT the
same configuration as B, which is why B exists.

    python benchmark/compare_three.py
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
import tifffile
from PIL import Image
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from benchmark.eval_masks import cldice_score, iou_score  # noqa: E402

RUNS = [
    ("A  inert cbDice (w=5)", "_ours_oldmodel", "_ours_oldmodel"),  # test dir, val dir
    ("B  no cbDice   (ep82)", "ours_nocb", "ours_nocb_val"),
    ("C  cbDice w=1  (ep72)", "_ours_C", "_ours_C"),
    ("D  cbDice 0.3 f3 (ep84)", "ours_cb03", "ours_cb03_val"),
]
SPLITS = {
    "TEST": ["A71_C1", "A81_C1", "A12_C3_100k", "A12_C1_100k_2"],
    "VAL": ["A91_C1_2", "A31_C1", "A52_C2", "A22_C2", "A42_C1"],
}


def frames(pred_dir, v):
    pr = tifffile.imread(os.path.join(REPO, "benchmark", "predictions", pred_dir, f"{v}.tif"))
    gts = sorted(glob.glob(os.path.join(REPO, "training_data", "masks", v, "*.png")))
    cl, io = [], []
    for t in range(1, len(pr)):
        g = np.array(Image.open(gts[t])) > 0
        m = pr[t].astype(bool)
        cl.append(cldice_score(m, g)); io.append(iou_score(m, g))
    return np.array(cl), np.array(io)


def main():
    for split, vids in SPLITS.items():
        print(f"\n{'='*78}\n{split}  ({len(vids)} recordings, full-length propagation, t>=1)\n{'='*78}")
        data = {}
        for label, tdir, vdir in RUNS:
            d = tdir if split == "TEST" else vdir
            per = {v: frames(d, v) for v in vids}
            data[label] = per
        # per-recording table
        print(f"{'run':<24}" + "".join(f"{v[:11]:>12}" for v in vids) + f"{'MEAN':>9}{'IoU':>8}")
        print("-" * (24 + 12 * len(vids) + 17))
        for label, _, _ in RUNS:
            per = data[label]
            cls = [per[v][0].mean() for v in vids]
            ios = [per[v][1].mean() for v in vids]
            print(f"{label:<24}" + "".join(f"{c:>12.3f}" for c in cls)
                  + f"{np.mean(cls):>9.4f}{np.mean(ios):>8.3f}")
        # drift
        print(f"\n{'run':<24}{'Q1':>9}{'Q2':>9}{'Q3':>9}{'Q4':>9}{'Q4-Q2':>9}")
        print("-" * 69)
        for label, _, _ in RUNS:
            per = data[label]
            q = [[] for _ in range(4)]
            for v in vids:
                cl = per[v][0]
                for i, idx in enumerate(np.array_split(np.arange(len(cl)), 4)):
                    q[i] += list(cl[idx])
            m = [np.mean(x) for x in q]
            print(f"{label:<24}" + "".join(f"{x:>9.3f}" for x in m) + f"{m[3]-m[1]:>+9.3f}")
        # paired vs A
        base = np.concatenate([data[RUNS[0][0]][v][0] for v in vids])
        print(f"\npaired per-frame clDice vs run A (n={len(base)}):")
        for label, _, _ in RUNS[1:]:
            x = np.concatenate([data[label][v][0] for v in vids])
            d = x - base
            print(f"  {label:<24} mean {d.mean():+.4f}   better on {(d>0).sum():>3}/{len(d)} "
                  f"({100*(d>0).mean():>3.0f}%)   t-test p={stats.ttest_rel(x, base).pvalue:.2e}")


if __name__ == "__main__":
    main()
