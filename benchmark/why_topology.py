#!/usr/bin/env python3
"""why_topology.py -- measures every number in Figure `fig:whytopology` and in
the Section 4.3 metric argument, on exactly the shape the figure draws.

Nothing here is illustrative: the schematic (disc of radius 0.34 cm plus three
1.1 pt segments) is rasterised at 100 px/cm and scored with the same
eval_masks functions used for the benchmark.

    KMP_DUPLICATE_LIB_OK=TRUE python benchmark/why_topology.py
"""
from __future__ import annotations
import os, sys
import numpy as np
from scipy.ndimage import binary_erosion

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark.eval_masks import iou_score, cbdice_score, cldice_score

PX = 100.0                            # px per TikZ cm
SOMA_R, LW = 0.34, 1.1 / 28.45        # circle radius (cm); 1.1 pt in cm
BRANCHES = [((0.25, 0.18), (1.45, 1.15)),
            ((-0.2, 0.28), (-1.1, 1.3)),
            ((0.05, -0.34), (0.35, -1.3))]
H = W = 360
_yy, _xx = np.mgrid[0:H, 0:W]
X, Y = (_xx - W / 2) / PX, (H / 2 - _yy) / PX


def render(keep: int) -> np.ndarray:
    """Soma plus the first `keep` branches, as drawn in the figure."""
    m = (X ** 2 + Y ** 2) <= SOMA_R ** 2
    for (x1, y1), (x2, y2) in BRANCHES[:keep]:
        dx, dy = x2 - x1, y2 - y1
        t = np.clip(((X - x1) * dx + (Y - y1) * dy) / (dx * dx + dy * dy), 0, 1)
        m |= (X - (x1 + t * dx)) ** 2 + (Y - (y1 + t * dy)) ** 2 <= (LW / 2) ** 2
    return m


def main():
    gt = render(3)
    rows = [("ground truth", gt),
            ("one branch lost", render(2)),
            ("two branches lost", render(1)),
            ("uniform erosion 1 px", binary_erosion(gt, iterations=1)),
            ("uniform erosion 2 px", binary_erosion(gt, iterations=2))]
    print(f"{'variant':<24}{'IoU':>8}{'cbDice':>9}{'clDice':>9}")
    print("-" * 50)
    sc = {}
    for lbl, p in rows:
        sc[lbl] = (iou_score(p, gt), cbdice_score(p, gt), cldice_score(p, gt))
        print(f"{lbl:<24}" + "".join(f"{v:>9.3f}" for v in sc[lbl]).rjust(26))
    print("-" * 50)
    g, b = sc["uniform erosion 1 px"], sc["two branches lost"]
    print(f"matched-IoU pair : IoU {g[0]:.3f} vs {b[0]:.3f} "
          f"-> cbDice {g[1]:.3f} vs {b[1]:.3f}  (separation {g[1]-b[1]:.3f})")
    d = [sc["ground truth"][i] - b[i] for i in range(3)]
    print(f"two branches lost: IoU -{d[0]:.3f}, cbDice -{d[1]:.3f} "
          f"({d[1]/d[0]:.1f}x), clDice -{d[2]:.3f}")


if __name__ == "__main__":
    main()
