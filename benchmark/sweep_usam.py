#!/usr/bin/env python3
"""Fairness sweep for micro-SAM: which backbone gives it the best shot?

The benchmark currently runs `vit_b_lm`, the SMALLEST light-microscopy
variant. micro-SAM also ships vit_l_lm and vit_h_lm, and its own docs
recommend the larger models for quality. Running a competitor's smallest
model while claiming "most favourable mode" is not defensible, so this
sweeps them on the VALIDATION split; the winner is applied unchanged to test.

    benchmark/competitors/usam_venv/bin/python benchmark/sweep_usam.py
"""
from __future__ import annotations
import glob, os
import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "training_data")
VAL = ["A91_C1_2", "A31_C1", "A52_C2", "A22_C2", "A42_C1"]
FRAMES_PER_VID = 5
MODELS = ["vit_b_lm", "vit_l_lm", "vit_h_lm"]


def load(name, t):
    f = sorted(glob.glob(os.path.join(DATA, "images", name, "*.png")))[t]
    g = sorted(glob.glob(os.path.join(DATA, "masks", name, "*.png")))[t]
    return np.array(Image.open(f).convert("L")), np.array(Image.open(g)) > 0


def gt_box(g):
    ys, xs = np.where(g)
    if len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def iou(p, g):
    u = (p | g).sum()
    return 1.0 if u == 0 else (p & g).sum() / u


def main():
    import torch
    from micro_sam.util import get_sam_model
    from micro_sam.prompt_based_segmentation import segment_from_box
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    tasks = []
    for v in VAL:
        n = len(glob.glob(os.path.join(DATA, "images", v, "*.png")))
        for t in np.linspace(1, n - 1, FRAMES_PER_VID, dtype=int):
            tasks.append((v, int(t)))
    imgs = [load(v, t) for v, t in tasks]

    print(f"device={dev}   {len(imgs)} validation frames\n")
    print(f"{'model':<12}{'mean IoU':>10}")
    print("-" * 22)
    best = None
    for mt in MODELS:
        try:
            predictor = get_sam_model(model_type=mt, device=dev)
        except Exception as e:
            print(f"{mt:<12}{'unavailable':>10}   ({str(e)[:60]})")
            continue
        ious = []
        for gray, gt in imgs:
            b = gt_box(gt)
            if b is None:
                continue
            predictor.set_image(np.stack([gray] * 3, -1))
            m = segment_from_box(predictor, np.array([b[1], b[0], b[3], b[2]]))
            ious.append(iou(np.asarray(m).astype(bool), gt))
        mi = float(np.mean(ious))
        print(f"{mt:<12}{mi:>10.3f}")
        if best is None or mi > best[0]:
            best = (mi, mt)
    if best:
        print(f"\nBEST on validation: {best[1]} (IoU {best[0]:.3f}). Apply unchanged to test.")


if __name__ == "__main__":
    main()
