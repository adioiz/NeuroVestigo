#!/usr/bin/env python3
"""Fairness sweep for Cellpose-SAM: find its most favourable settings.

Run on the VALIDATION split only -- the chosen setting is then applied
unchanged to test, exactly as for every other competitor.

Knobs swept:
  scale             cpsam is advertised as scale-invariant (the diameter /
                    rescaling step was REMOVED in v4), so upscaling may hurt.
  cellprob_threshold  lower = more permissive; thin faint neurites are exactly
                    what a default threshold of 0.0 will drop.
  flow_threshold    higher = accepts less cell-like flow fields.
  min_size          default 15 px discards small fragments; a 2-px-wide
                    neurite fragment can fall below this.

    benchmark/competitors/cp_venv/bin/python benchmark/sweep_cellpose.py
"""
from __future__ import annotations
import glob, os
import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "training_data")
VAL = ["A91_C1_2", "A31_C1", "A52_C2", "A22_C2", "A42_C1"]
FRAMES_PER_VID = 6            # evenly spaced, keeps the sweep tractable


def load(name, t):
    f = sorted(glob.glob(os.path.join(DATA, "images", name, "*.png")))[t]
    g = sorted(glob.glob(os.path.join(DATA, "masks", name, "*.png")))[t]
    return np.array(Image.open(f).convert("L")), np.array(Image.open(g)) > 0


def match_to_target(inst, gt):
    out = np.zeros(gt.shape, bool)
    for lab in np.unique(inst):
        if lab and np.logical_and(inst == lab, gt).any():
            out |= inst == lab
    return out


def iou(p, g):
    u = (p | g).sum()
    return 1.0 if u == 0 else (p & g).sum() / u


def main():
    import cv2, torch
    from cellpose import models
    gpu = torch.backends.mps.is_available() or torch.cuda.is_available()
    model = models.CellposeModel(gpu=gpu)
    print(f"cpsam, gpu={gpu}\n")

    tasks = []
    for v in VAL:
        n = len(glob.glob(os.path.join(DATA, "images", v, "*.png")))
        for t in np.linspace(1, n - 1, FRAMES_PER_VID, dtype=int):
            tasks.append((v, int(t)))
    imgs = [load(v, t) for v, t in tasks]

    grid = []
    for scale in (2.0, 3.0):
        for cellprob in (0.0, -1.0, -2.0):
            for invert in (False, True):
                grid.append((scale, cellprob, invert))

    print(f"{'scale':>6}{'cellprob':>10}{'invert':>8}{'mean IoU':>10}"
          f"{'inst/frame':>12}{'empty frames':>14}")
    print("-" * 62)
    best = None
    for scale, cellprob, invert in grid:
        ious, ninst, empty = [], [], 0
        for (gray, gt) in imgs:
            img = (255 - gray).astype(np.uint8) if invert else gray
            up = cv2.resize(img, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_LINEAR) if scale != 1.0 else img
            m, _, _ = model.eval(up, batch_size=8, cellprob_threshold=cellprob)
            m = np.asarray(m).astype(np.int32)
            if scale != 1.0:
                m = cv2.resize(m, (gray.shape[1], gray.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            ninst.append(len(np.unique(m)) - 1)
            tgt = match_to_target(m, gt)
            if not tgt.any():
                empty += 1
            ious.append(iou(tgt, gt))
        mi = float(np.mean(ious))
        print(f"{scale:>6.1f}{cellprob:>10.1f}{str(invert):>8}{mi:>10.3f}"
              f"{np.mean(ninst):>12.1f}{empty:>10}/{len(imgs)}")
        if best is None or mi > best[0]:
            best = (mi, scale, cellprob, invert)
    print(f"\nBEST on validation: IoU {best[0]:.3f} at scale {best[1]}, "
          f"cellprob_threshold {best[2]}, invert={best[3]}")
    print("Apply this UNCHANGED to the test split.")


if __name__ == "__main__":
    main()
