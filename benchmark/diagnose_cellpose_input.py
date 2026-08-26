#!/usr/bin/env python3
"""Is Cellpose-SAM failing on A71/A81, or are we feeding it badly?

Tests input handling, which is the most likely place for a silent bug:
  1  raw uint8 grayscale 2-D          (what run_cellpose.py currently passes)
  2  grayscale replicated to 3 channels
  3  per-frame contrast stretch to full 0-255 range, then grayscale
  4  float32 in [0,1]
  5  invert (phase contrast can be dark-on-light or light-on-dark)

For each variant, reports how many instances cpsam returns and the IoU after
target matching. If variant 1 is far worse than the others, the benchmark has
been handicapping cpsam through preprocessing, not through the model.

    benchmark/competitors/cp_venv/bin/python benchmark/diagnose_cellpose_input.py
"""
from __future__ import annotations
import glob, os
import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "training_data")
CASES = [("A71_C1", 20), ("A71_C1", 45), ("A71_C1", 64),
         ("A81_C1", 13), ("A81_C1", 40), ("A81_C1", 60)]


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


def variants(gray):
    lo, hi = np.percentile(gray, (1, 99))
    stretched = np.clip((gray.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255)
    return {
        "1 raw uint8 gray": gray,
        "2 gray->3ch": np.stack([gray] * 3, -1),
        "3 stretched": stretched.astype(np.uint8),
        "4 float [0,1]": (gray.astype(np.float32) / 255.0),
        "5 inverted": (255 - gray).astype(np.uint8),
    }


def main():
    import cv2, torch
    from cellpose import models
    gpu = torch.backends.mps.is_available() or torch.cuda.is_available()
    model = models.CellposeModel(gpu=gpu)
    SCALE, CELLPROB = 2.0, -1.0
    print(f"cpsam  scale={SCALE}  cellprob_threshold={CELLPROB}  gpu={gpu}\n")

    names = list(variants(np.zeros((4, 4), np.uint8)).keys())
    print(f"{'frame':<16}" + "".join(f"{n:>18}" for n in names))
    print("-" * (16 + 18 * len(names)))
    totals = {n: [] for n in names}
    for v, t in CASES:
        gray, gt = load(v, t)
        row = []
        for n, img in variants(gray).items():
            up = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)
            try:
                m, _, _ = model.eval(up, batch_size=8, cellprob_threshold=CELLPROB)
                m = np.asarray(m).astype(np.int32)
                m = cv2.resize(m, (gray.shape[1], gray.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            except Exception as e:
                row.append(f"ERR {str(e)[:8]}"); totals[n].append(0.0); continue
            ni = len(np.unique(m)) - 1
            s = iou(match(m, gt), gt)
            totals[n].append(s)
            row.append(f"{ni:>3}inst {s:>5.3f}")
        print(f"{v+' t'+str(t):<16}" + "".join(f"{c:>18}" for c in row))
    print("-" * (16 + 18 * len(names)))
    print(f"{'MEAN IoU':<16}" + "".join(f"{np.mean(totals[n]):>18.3f}" for n in names))


if __name__ == "__main__":
    main()
