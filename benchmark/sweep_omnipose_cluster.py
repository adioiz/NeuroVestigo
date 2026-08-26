#!/usr/bin/env python3
"""Fairness sweep for Omnipose: the knobs it exposes for FILAMENTOUS cells.

run_omnipose.py sets mask_threshold=-1.0 and flow_threshold=0.0 (already
permissive) but leaves cluster=False, niter=None and min_size=15 at library
defaults. Omnipose documents cluster=True for thin / low-resolution cells --
exactly our 1-2 px neurites -- and min_size=15 can discard neurite fragments
outright. Omnipose is currently the weakest method in the benchmark, so this
is where a handicap would hide.

Validation split only; the winner is applied unchanged to test.

    conda run -n omnipose python benchmark/sweep_omnipose_cluster.py
"""
from __future__ import annotations
import glob, os
import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "training_data")
VAL = ["A91_C1_2", "A31_C1", "A52_C2", "A22_C2", "A42_C1"]
FRAMES_PER_VID = 3


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
    from cellpose_omni import models
    gpu = torch.backends.mps.is_available() or torch.cuda.is_available()
    model = models.CellposeModel(gpu=gpu, model_type="bact_phase_omni", omni=True)

    tasks = []
    for v in VAL:
        n = len(glob.glob(os.path.join(DATA, "images", v, "*.png")))
        for t in np.linspace(1, n - 1, FRAMES_PER_VID, dtype=int):
            tasks.append((v, int(t)))
    imgs = [load(v, t) for v, t in tasks]
    print(f"omnipose gpu={gpu}  bact_phase_omni  scale=x3  {len(imgs)} validation frames\n")

    BASE = dict(channels=[0, 0], omni=True, diameter=None, mask_threshold=-1.0,
                flow_threshold=0.0, resample=True, verbose=False)
    CFG = [("shipped (defaults)",        {}),
           ("cluster=True",              dict(cluster=True)),
           ("cluster + min_size=3",      dict(cluster=True, min_size=3)),
           ("min_size=3 only",           dict(min_size=3)),
           ("cluster + niter=200",       dict(cluster=True, niter=200)),
           ("affinity_seg=True",         dict(affinity_seg=True))]

    print(f"{'config':<24}{'mean IoU':>10}{'inst/frame':>12}{'empty':>9}")
    print("-" * 55)
    best = None
    for lbl, extra in CFG:
        ii, ni, empty = [], [], 0
        for gray, gt in imgs:
            up = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LINEAR)
            try:
                r = model.eval([up], **{**BASE, **extra})
                mk = np.asarray(r[0][0]).astype(np.int32)
            except Exception as e:
                print(f"{lbl:<24}  ERROR {str(e)[:44]}"); ii = None; break
            mk = cv2.resize(mk, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
            ni.append(len(np.unique(mk)) - 1)
            tg = match(mk, gt)
            empty += (not tg.any())
            ii.append(iou(tg, gt))
        if ii is None:
            continue
        mi = float(np.mean(ii))
        print(f"{lbl:<24}{mi:>10.3f}{np.mean(ni):>12.1f}{empty:>6}/{len(imgs)}")
        if best is None or mi > best[0]:
            best = (mi, lbl, extra)
    if best:
        print(f"\nBEST on validation: {best[1]} (IoU {best[0]:.3f}) -> {best[2]}")
        print("Apply unchanged to test.")


if __name__ == "__main__":
    main()
