#!/usr/bin/env python3
"""Fairness sweep for Cellpose-SAM: flow_threshold.

sweep_cellpose.py named flow_threshold in its docstring but never swept it --
the grid was scale x cellprob x invert. flow_threshold is cellpose's flow-QC
step: masks whose flow error exceeds it are DISCARDED. Thin neurites produce
flow fields unlike the blobby cells cpsam was trained on, so the default 0.4
was throwing away valid detections (cpsam returned 0 instances on 37 of 72
frames of A71_C1).

Selected on the VALIDATION split only, then applied unchanged to test.

    benchmark/competitors/cp_venv/bin/python benchmark/sweep_cellpose_flow.py
"""
from __future__ import annotations
import glob, os
import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "training_data")
VAL = ["A91_C1_2", "A31_C1", "A52_C2", "A22_C2", "A42_C1"]
FRAMES_PER_VID = 8


def load(v, t):
    f = sorted(glob.glob(os.path.join(DATA, "images", v, "*.png")))[t]
    g = sorted(glob.glob(os.path.join(DATA, "masks", v, "*.png")))[t]
    return np.array(Image.open(f).convert("L")), np.array(Image.open(g)) > 0


def match(inst, gt):
    out = np.zeros(gt.shape, bool)
    for l in np.unique(inst):
        if l and np.logical_and(inst == l, gt).any():
            out |= inst == l
    return out


def iou(p, g):
    u = (p | g).sum()
    return 1.0 if u == 0 else (p & g).sum() / u


def main():
    import cv2, torch
    from cellpose import models
    gpu = torch.backends.mps.is_available() or torch.cuda.is_available()
    model = models.CellposeModel(gpu=gpu)

    tasks = []
    for v in VAL:
        n = len(glob.glob(os.path.join(DATA, "images", v, "*.png")))
        for t in np.linspace(1, n - 1, FRAMES_PER_VID, dtype=int):
            tasks.append((v, int(t)))
    imgs = [load(v, t) for v, t in tasks]
    print(f"cpsam gpu={gpu}  scale=x2  cellprob=-1.0   {len(imgs)} validation frames\n")
    print(f"{'flow_threshold':>15}{'mean IoU':>10}{'inst/frame':>12}{'empty':>9}")
    print("-" * 46)

    best = None
    for ft in (0.4, 0.8, 1.0, 1.5, 2.0, 3.0, 0.0):
        ii, ni, empty = [], [], 0
        for gray, gt in imgs:
            up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            r = model.eval(up, batch_size=8, cellprob_threshold=-1.0, flow_threshold=ft)
            mk = cv2.resize(np.asarray(r[0]).astype(np.int32),
                            (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
            ni.append(len(np.unique(mk)) - 1)
            tg = match(mk, gt)
            empty += (not tg.any())
            ii.append(iou(tg, gt))
        mi = float(np.mean(ii))
        lbl = "0.0 (QC off)" if ft == 0.0 else f"{ft}"
        print(f"{lbl:>15}{mi:>10.3f}{np.mean(ni):>12.1f}{empty:>6}/{len(imgs)}")
        if best is None or mi > best[0]:
            best = (mi, ft)
    print(f"\nBEST on validation: flow_threshold={best[1]} (IoU {best[0]:.3f}). "
          f"Apply unchanged to test.")
    print("NB: normalize=False was also tried and reaches a higher raw IoU, but only by\n"
          "    shattering the frame into ~4900 instances, which the union-matching\n"
          "    protocol then rewards. cpsam is trained WITH normalisation; that config\n"
          "    is a scoring artefact, not a favourable setting, and is excluded.")


if __name__ == "__main__":
    main()
