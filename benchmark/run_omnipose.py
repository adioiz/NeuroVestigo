#!/usr/bin/env python3
"""
run_omnipose.py -- Omnipose competitor runner (Cutler et al., Nature Methods 2022).

Omnipose is THE method built for elongated / branching / filamentous morphology --
it replaces Cellpose's centroid flow field with a boundary-distance (eikonal) field
so filaments don't fragment. It is the strongest morphology competitor for our
thin-neurite thesis, so a reviewer will demand it. AUTOMATIC + single-frame (no
prompt, no temporal memory), so same fair protocol as the other auto-segmenters:
segment every object per frame, then union instance-match to the target-neuron GT.

Run with the isolated conda env:
    /opt/homebrew/Caskroom/miniconda/base/bin/conda run -n omnipose \
        python benchmark/run_omnipose.py --videos ... --model cyto2_omni

Predicted target stacks -> benchmark/predictions/omnipose/<name>.tif
Raw instance-label stacks -> benchmark/predictions/omnipose_raw/<name>.tif
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import tifffile
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "training_data")
OUT = os.path.join(REPO, "benchmark", "predictions", "omnipose")


def load_frames(name, limit=None):
    files = sorted(glob.glob(os.path.join(DATA, "images", name, "*.png")))
    if limit:
        files = files[:limit]
    return [np.array(Image.open(f).convert("L")) for f in files]


def load_gt(name, n):
    files = sorted(glob.glob(os.path.join(DATA, "masks", name, "*.png")))[:n]
    return [np.array(Image.open(f)) > 0 for f in files]


def match_to_target(inst_mask, gt):
    out = np.zeros(gt.shape, bool)
    if not gt.any():
        return out
    for lab in np.unique(inst_mask):
        if lab == 0:
            continue
        m = inst_mask == lab
        if np.logical_and(m, gt).sum() > 0:
            out |= m
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--model", default="bact_phase_omni",
                    help="Omnipose pretrained model. bact_phase_omni is its phase-contrast "
                         "model and the only one that fires on our neurons (cyto2_omni "
                         "detects nothing on phase contrast).")
    ap.add_argument("--diameter", type=float, default=0.0,
                    help="0 = auto-estimate per image (Omnipose size model)")
    # NOTE (2026-08-11): the x3 choice came from a sweep over 5 frames/video on
    # validation. Re-scored later over the FULL validation split by full-video
    # propagation, x2 and x3 are indistinguishable: IoU 0.4173 (x2) vs 0.4154
    # (x3), a 0.002 gap against a 0.045 between-recording s.d. Kept at x3 as
    # originally selected; the choice does not matter at this resolution.
    ap.add_argument("--scale", type=float, default=3.0,
                    help="upscale factor. x3 is the VALIDATION argmax of a sweep over "
                         "model x scale (bact_phase_omni .316/.412/.417 at x1/x2/x3; "
                         "cyto2_omni 0.000 at every scale, 25/25 frames empty; "
                         "bact_fluor_omni .248/.381/.152). x2 and x3 differ by .005, "
                         "i.e. within noise. Selected on validation only, applied "
                         "unchanged to test.")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or OUT

    import cv2
    import torch
    from cellpose_omni import models
    gpu = torch.backends.mps.is_available() or torch.cuda.is_available()
    print(f"omnipose model={args.model} gpu={gpu}")
    model = models.CellposeModel(gpu=gpu, model_type=args.model, omni=True)
    os.makedirs(out_dir, exist_ok=True)

    for v in args.videos:
        frames = load_frames(v, args.limit_frames)
        gts = load_gt(v, len(frames))
        print(f"  {v}: {len(frames)} frames (scale x{args.scale}) ...", flush=True)
        s = args.scale
        up = [cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR) if s != 1.0 else f
              for f in frames]
        masks, _, _ = model.eval(
            up, channels=[0, 0], omni=True,
            diameter=(args.diameter or None),
            mask_threshold=-1.0, flow_threshold=0.0, resample=True, verbose=False)
        # downscale instance maps back to native resolution for matching/scoring
        raw = [cv2.resize(np.asarray(m).astype(np.int32), (f.shape[1], f.shape[0]),
                          interpolation=cv2.INTER_NEAREST) for m, f in zip(masks, frames)]
        target = np.stack([match_to_target(m, g).astype(np.uint8)
                           for m, g in zip(raw, gts)])
        out_path = os.path.join(out_dir, f"{v}.tif")
        tifffile.imwrite(out_path, target)
        raw_dir = out_dir + "_raw"; os.makedirs(raw_dir, exist_ok=True)
        tifffile.imwrite(os.path.join(raw_dir, f"{v}.tif"), np.stack(raw))
        ninst = np.mean([len(np.unique(m)) - 1 for m in raw])
        print(f"    -> {out_path}  (avg {ninst:.1f} instances/frame, "
              f"target fg={float((target>0).mean()):.4f})")


if __name__ == "__main__":
    main()
