#!/usr/bin/env python3
"""Fairness sweep for micro-SAM: how the output mask is chosen.

run_usam.py asks for three candidate masks and keeps uSAM's own highest-scoring
one -- the documented usage. On ~9% of test frames, all late in a recording when
the target's bounding box is large, uSAM's predicted-IoU score is miscalibrated
and ranks a mask filling most of the box above a much better one. This asks
whether the single-mask mode (multimask_output=False) avoids that.

Selection on the VALIDATION split only, then applied unchanged to test.

    benchmark/competitors/usam_venv/bin/python benchmark/sweep_usam_multimask.py
"""
from __future__ import annotations
import glob, os
import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "training_data")
VAL = ["A91_C1_2", "A31_C1", "A52_C2", "A22_C2", "A42_C1"]
FRAMES_PER_VID = 8


def gt_box_yxyx(g):
    ys, xs = np.where(g)
    return np.array([ys.min(), xs.min(), ys.max() + 1, xs.max() + 1])


def iou(p, g):
    u = (p | g).sum()
    return 1.0 if u == 0 else (p & g).sum() / u


def main():
    import torch
    from micro_sam.util import get_sam_model
    from micro_sam.prompt_based_segmentation import segment_from_box
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    pred = get_sam_model(model_type="vit_l_lm", device=dev)

    tasks = []
    for v in VAL:
        n = len(glob.glob(os.path.join(DATA, "images", v, "*.png")))
        for t in np.linspace(1, n - 1, FRAMES_PER_VID, dtype=int):
            tasks.append((v, int(t)))
    print(f"micro-sam vit_l_lm  {len(tasks)} validation frames\n")
    print(f"{'mask selection':<34}{'mean IoU':>10}{'blow-ups >3x GT':>18}")
    print("-" * 62)

    res = {"uSAM's top-scoring of 3 (shipped)": [], "single mask (multimask_output=False)": []}
    blow = {k: 0 for k in res}
    for v, t in tasks:
        fr = np.array(Image.open(sorted(glob.glob(os.path.join(DATA, "images", v, "*.png")))[t]).convert("L"))
        g = np.array(Image.open(sorted(glob.glob(os.path.join(DATA, "masks", v, "*.png")))[t])) > 0
        if not g.any():
            continue
        pred.set_image(np.stack([fr] * 3, -1))
        box = gt_box_yxyx(g)
        for lbl, mm in [("uSAM's top-scoring of 3 (shipped)", True),
                        ("single mask (multimask_output=False)", False)]:
            r = segment_from_box(pred, box, multimask_output=mm, return_all=True)
            masks = np.asarray(r[0]).reshape(-1, *g.shape)
            scores = np.asarray(r[1]).ravel()
            m = masks[int(np.argmax(scores))].astype(bool)
            res[lbl].append(iou(m, g))
            blow[lbl] += (m.sum() / max(g.sum(), 1) > 3)
    for lbl, v in res.items():
        print(f"{lbl:<34}{np.mean(v):>10.3f}{blow[lbl]:>12}/{len(v)}")
    best = max(res, key=lambda k: np.mean(res[k]))
    print(f"\nBEST on validation: {best}.  Apply unchanged to test.")


if __name__ == "__main__":
    main()
