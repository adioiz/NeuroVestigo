#!/usr/bin/env python3
"""Fairness sweep for Omnipose: model choice x scale.

The benchmark runs `bact_phase_omni` -- a BACTERIAL phase-contrast model --
on the strength of a code comment claiming cyto2_omni "detects nothing on
phase contrast". That is a note, not a measurement, and the model choice
moves Omnipose's score a lot. This measures it on the VALIDATION split; the
winner is applied unchanged to test.

    benchmark/competitors/omnipose_venv/bin/python benchmark/sweep_omnipose.py
"""
from __future__ import annotations
import glob, os
import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "training_data")
VAL = ["A91_C1_2", "A31_C1", "A52_C2", "A22_C2", "A42_C1"]
FRAMES_PER_VID = 5
CANDIDATES = ["bact_phase_omni", "cyto2_omni", "bact_fluor_omni", "plant_omni"]
SCALES = [1.0, 2.0, 3.0]


def load(name, t):
    f = sorted(glob.glob(os.path.join(DATA, "images", name, "*.png")))[t]
    g = sorted(glob.glob(os.path.join(DATA, "masks", name, "*.png")))[t]
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

    tasks = []
    for v in VAL:
        n = len(glob.glob(os.path.join(DATA, "images", v, "*.png")))
        for t in np.linspace(1, n - 1, FRAMES_PER_VID, dtype=int):
            tasks.append((v, int(t)))
    imgs = [load(v, t) for v, t in tasks]
    print(f"gpu={gpu}   {len(imgs)} validation frames\n")
    print(f"{'model':<20}{'scale':>7}{'mean IoU':>10}{'inst/frame':>12}{'empty':>8}")
    print("-" * 57)

    best = None
    for mt in CANDIDATES:
        try:
            model = models.CellposeModel(gpu=gpu, model_type=mt, omni=True)
        except Exception as e:
            print(f"{mt:<20}{'unavailable':>17}   ({str(e)[:45]})")
            continue
        for s in SCALES:
            ious, ninst, empty = [], [], 0
            for gray, gt in imgs:
                up = cv2.resize(gray, None, fx=s, fy=s,
                                interpolation=cv2.INTER_LINEAR) if s != 1.0 else gray
                try:
                    m, _, _ = model.eval([up], channels=[0, 0], omni=True, diameter=None,
                                         mask_threshold=-1.0, flow_threshold=0.0,
                                         resample=True, verbose=False)
                    m = np.asarray(m[0]).astype(np.int32)
                except Exception:
                    m = np.zeros_like(gray, np.int32)
                if s != 1.0:
                    m = cv2.resize(m, (gray.shape[1], gray.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
                ninst.append(len(np.unique(m)) - 1)
                tgt = match(m, gt)
                empty += (not tgt.any())
                ious.append(iou(tgt, gt))
            mi = float(np.mean(ious))
            print(f"{mt:<20}{s:>7.1f}{mi:>10.3f}{np.mean(ninst):>12.1f}{empty:>6}/{len(imgs)}")
            if best is None or mi > best[0]:
                best = (mi, mt, s)
    if best:
        print(f"\nBEST on validation: {best[1]} at scale {best[2]} (IoU {best[0]:.3f})."
              f"  Apply unchanged to test.")


if __name__ == "__main__":
    main()
