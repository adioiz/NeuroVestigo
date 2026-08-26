#!/usr/bin/env python3
"""
verify_auto_dump.py -- dump RAW instance maps + matched target for an automatic
competitor, so we can visually confirm it segments the right region (no coord/
resolution bug like the µSAM box order). Run once per model with its env:

    benchmark/competitors/cp_venv/bin/python benchmark/verify_auto_dump.py --model cellpose
    benchmark/competitors/deepsea_venv/bin/python benchmark/verify_auto_dump.py --model deepsea

Writes /tmp/verify/<model>_<v>_<t>_{raw,matched}.tif
"""
import argparse
import glob
import os
import sys

import numpy as np
import tifffile
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "training_data")
OUT = "/tmp/verify"
FRAMES = [("A71_C1", 35), ("A81_C1", 44), ("A31_C1", 30), ("A52_C2", 66)]


def gray(v, t):
    return np.array(Image.open(os.path.join(DATA, "images", v, f"{t:05d}.png")).convert("L"))


def gt(v, t):
    return np.array(Image.open(os.path.join(DATA, "masks", v, f"{t:05d}.png"))) > 0


def match_to_target(inst, g):
    out = np.zeros(g.shape, bool)
    if not g.any():
        return out
    for lab in np.unique(inst):
        if lab != 0 and np.logical_and(inst == lab, g).sum() > 0:
            out |= inst == lab
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["cellpose", "deepsea"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if args.model == "cellpose":
        import torch
        from cellpose import models
        m = models.CellposeModel(gpu=torch.backends.mps.is_available())
        def seg(fr):
            masks, _, _ = m.eval([fr], batch_size=8)
            return np.asarray(masks[0]).astype(np.int32)
    else:  # deepsea
        DS = os.path.join(REPO, "benchmark", "competitors", "deepsea_repo", "deepsea")
        sys.path.insert(0, DS)
        import torch
        import torchvision.transforms as T
        from scipy import ndimage as ndi
        from skimage.morphology import remove_small_objects
        from skimage.transform import resize as sk_resize
        from model import DeepSeaSegmentation
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        net = DeepSeaSegmentation(n_channels=1, n_classes=2, bilinear=True)
        net.load_state_dict(torch.load(os.path.join(DS, "trained_models", "segmentation.pth"),
                                       map_location="cpu"))
        net = net.to(dev).eval()
        tf = T.Compose([T.ToPILImage(), T.Resize([383, 512]), T.ToTensor(),
                        T.Normalize(mean=[0.5], std=[0.5])])
        def seg(fr):
            rng = float(fr.max() - fr.min()) or 1.0
            img8 = (255 * ((fr - fr.min()) / rng)).astype(np.uint8)
            with torch.no_grad():
                x = tf(img8).to(dev, torch.float32).unsqueeze(0)
                mp, _ = net(x)
            lab = ndi.label(remove_small_objects(mp.argmax(1).cpu().numpy()[0] > 0,
                                                 min_size=20, connectivity=1))[0]
            return sk_resize(lab, fr.shape, order=0, preserve_range=True,
                             anti_aliasing=False).astype(np.int32)

    for v, t in FRAMES:
        raw = seg(gray(v, t))
        matched = match_to_target(raw, gt(v, t)).astype(np.uint8)
        tifffile.imwrite(os.path.join(OUT, f"{args.model}_{v}_{t}_raw.tif"), raw)
        tifffile.imwrite(os.path.join(OUT, f"{args.model}_{v}_{t}_matched.tif"), matched)
        n = len(np.unique(raw)) - 1
        ov = matched.sum()
        print(f"  {args.model} {v} f{t}: {n} raw instances, matched fg={int(ov)}px, GT={int(gt(v,t).sum())}px")


if __name__ == "__main__":
    main()
