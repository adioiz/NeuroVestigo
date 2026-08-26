#!/usr/bin/env python3
"""
run_deepsea.py -- DeepSea competitor runner (Zargari et al., Cell Reports Methods 2023).

Run with the isolated env:
    benchmark/competitors/deepsea_venv/bin/python benchmark/run_deepsea.py --videos ...

DeepSea is your CLOSEST task relative: a small U-Net (+edge branch) for single-cell
segmentation + tracking in phase-contrast time-lapse. Bundled pretrained weights.
It is AUTOMATIC (segments all cells per frame), so same instance-matching protocol:
union of predicted instances overlapping the target neuron's GT. DeepSea forces a
383x512 input; we resize its instance map back to native GT resolution before matching.

Predicted target stacks -> benchmark/predictions/deepsea/<name>.tif
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import tifffile
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "training_data")
OUT = os.path.join(REPO, "benchmark", "predictions", "deepsea")
DS_DIR = os.path.join(REPO, "benchmark", "competitors", "deepsea_repo", "deepsea")
CKPT = os.path.join(DS_DIR, "trained_models", "segmentation.pth")


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
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args()

    sys.path.insert(0, DS_DIR)
    import torch
    import torchvision.transforms as T
    from scipy import ndimage as ndi
    from skimage.morphology import remove_small_objects
    from skimage.transform import resize as sk_resize
    from model import DeepSeaSegmentation

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"DeepSea device={device}")
    model = DeepSeaSegmentation(n_channels=1, n_classes=2, bilinear=True)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model = model.to(device).eval()
    tf = T.Compose([T.ToPILImage(), T.Resize([383, 512]), T.ToTensor(),
                    T.Normalize(mean=[0.5], std=[0.5])])
    os.makedirs(args.out_dir, exist_ok=True)

    for v in args.videos:
        frames = load_frames(v, args.limit_frames)
        gts = load_gt(v, len(frames))
        print(f"  {v}: {len(frames)} frames ...", flush=True)
        target, ninst, raw = [], [], []
        with torch.no_grad():
            for fr, g in zip(frames, gts):
                rng = float(fr.max() - fr.min()) or 1.0
                img8 = (255 * ((fr - fr.min()) / rng)).astype(np.uint8)
                x = tf(img8).to(device, torch.float32).unsqueeze(0)
                mask_pred, _ = model(x)
                m = mask_pred.argmax(dim=1).cpu().numpy()[0]           # 383x512
                lab = ndi.label(remove_small_objects(m > 0, min_size=20, connectivity=1))[0]
                lab_native = sk_resize(lab, g.shape, order=0, preserve_range=True,
                                       anti_aliasing=False).astype(np.int32)
                ninst.append(len(np.unique(lab_native)) - 1)
                raw.append(lab_native)
                target.append(match_to_target(lab_native, g).astype(np.uint8))
        target = np.stack(target)
        out_path = os.path.join(args.out_dir, f"{v}.tif")
        tifffile.imwrite(out_path, target)
        # also save the RAW instance-label stack (all detected cells) for figures
        raw_dir = args.out_dir + "_raw"; os.makedirs(raw_dir, exist_ok=True)
        tifffile.imwrite(os.path.join(raw_dir, f"{v}.tif"), np.stack(raw))
        print(f"    -> {out_path}  (avg {np.mean(ninst):.1f} instances/frame, "
              f"target fg={float((target>0).mean()):.4f})")


if __name__ == "__main__":
    main()
