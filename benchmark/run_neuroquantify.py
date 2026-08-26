#!/usr/bin/env python3
"""
run_neuroquantify.py -- NeuroQuantify competitor runner (Chen et al., J. Neurosci.
Methods 2024). The most on-point DOMAIN competitor: a U-Net built specifically to
segment cell bodies AND neurites in PHASE-CONTRAST microscopy -- same modality and
same thin-neurite structure as ours. It is SEMANTIC (classes: soma / neurite), not
instance/temporal, so it competes on the segmentation + clDice/connectivity axis,
NOT on temporal propagation. That is the honest framing.

We run it in the authors' intended favorable pipeline: gamma correction -> large
TorchScript model (imageBulkProcess.py does exactly `unet_predict(gamma_img, 1)`).
Foreground = soma (class 1) + neurite (class 2). Reduced to the target neuron by the
same union instance-matching as the other automatic competitors (connected components
of the predicted foreground overlapping the target GT).

Run with the isolated env:
    benchmark/competitors/neuroquantify_venv/bin/python benchmark/run_neuroquantify.py --videos ...

Predicted target stacks -> benchmark/predictions/neuroquantify/<name>.tif
Raw semantic-label stacks -> benchmark/predictions/neuroquantify_raw/<name>.tif
"""
from __future__ import annotations

import argparse
import glob
import math
import os

import numpy as np
import tifffile
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "training_data")
OUT = os.path.join(REPO, "benchmark", "predictions", "neuroquantify")
NQ = os.path.join(REPO, "benchmark", "competitors", "neuroquantify_repo",
                  "neural-image-segmentation")
CKPT = os.path.join(NQ, "unet", "optimized_model.pt")     # large TorchScript model


def load_frames(name, limit=None):
    files = sorted(glob.glob(os.path.join(DATA, "images", name, "*.png")))
    if limit:
        files = files[:limit]
    return [np.array(Image.open(f).convert("L")) for f in files]


def load_gt(name, n):
    files = sorted(glob.glob(os.path.join(DATA, "masks", name, "*.png")))[:n]
    return [np.array(Image.open(f)) > 0 for f in files]


def gamma_correction(bgr):
    """Author's gamma step (unet_utils.gamma_correction), on a 3-channel image."""
    import cv2
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray)) or 1.0
    gamma = math.log(0.5 * 255) / math.log(mean)
    return np.power(bgr, gamma).clip(0, 255).astype(np.uint8)


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
    ap.add_argument("--scale", type=float, default=4.0,
                    help="upscale factor to bring our ~1.5 px/um frames to NeuroQuantify's "
                         "training magnification (native scale is out-of-distribution: it "
                         "labels ~65%% of the frame as cell). x4 selected by a sweep on the "
                         "VALIDATION split only (x2/x3/x4/x5/x6 -> clDice .63/.78/.83/.81/.82; "
                         "peaks at x4) and applied unchanged to test. Never tuned on test data.")
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args()

    import cv2
    import torch
    from scipy import ndimage as ndi
    torch.backends.quantized.engine = "qnnpack"      # large model is int8-quantized (ARM engine)
    # TorchScript model was traced on CPU (quantized); keep on CPU.
    model = torch.jit.load(CKPT, map_location="cpu").eval()
    print(f"NeuroQuantify loaded (large model); device=cpu (traced)")
    os.makedirs(args.out_dir, exist_ok=True)

    def pad32(a):
        h, w = a.shape[:2]
        ph, pw = (-h) % 32, (-w) % 32
        return np.pad(a, ((0, ph), (0, pw), (0, 0)), mode="reflect"), (h, w)

    for v in args.videos:
        frames = load_frames(v, args.limit_frames)
        gts = load_gt(v, len(frames))
        print(f"  {v}: {len(frames)} frames ...", flush=True)
        target, ninst, raw = [], [], []
        with torch.no_grad():
            for fr, g in zip(frames, gts):
                s = args.scale
                up = cv2.resize(fr, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR) \
                    if s != 1.0 else fr
                bgr = np.repeat(up[:, :, None], 3, axis=2).astype(np.uint8)   # gray->BGR
                gimg = gamma_correction(bgr)
                padded, (h, w) = pad32(gimg)
                x = padded[:, :, 0].astype(np.float32)
                if x.max() > 1:
                    x = x / 255.0
                x = torch.from_numpy(x)[None, None].float()
                seg = model(x).argmax(dim=1)[0].cpu().numpy()[:h, :w]         # 0/1/2/3 at upscaled res
                seg = cv2.resize(seg.astype(np.int32), (fr.shape[1], fr.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)             # back to native
                fg = np.isin(seg, (1, 2))                                     # soma + neurite
                lab = ndi.label(fg)[0].astype(np.int32)
                ninst.append(int(lab.max()))
                raw.append(seg.astype(np.int32))
                target.append(match_to_target(lab, g).astype(np.uint8))
        target = np.stack(target)
        out_path = os.path.join(args.out_dir, f"{v}.tif")
        tifffile.imwrite(out_path, target)
        raw_dir = args.out_dir + "_raw"; os.makedirs(raw_dir, exist_ok=True)
        tifffile.imwrite(os.path.join(raw_dir, f"{v}.tif"), np.stack(raw))
        print(f"    -> {out_path}  (avg {np.mean(ninst):.1f} components/frame, "
              f"target fg={float((target>0).mean()):.4f})")


if __name__ == "__main__":
    main()
