#!/usr/bin/env python3
"""
probe_teacher.py -- does the distillation teacher know anything, or is it
copying the prompt?

WHY THIS EXISTS. The prompt-pathway distillation was motivated by a measured
gap: frame 0 (given the GT mask as a dense prompt) scores 0.9466 IoU while
frames 1-15 (memory only) score ~0.83. That was read as "the decoder can
produce a 0.947 mask of this image, it just cannot reach that unaided" -- so
train the unaided pathway to reproduce it.

That reading conflates two different abilities:

    (a) refining a mask it is HANDED
    (b) knowing where the neuron IS

Only (b) is transferable. If the teacher's advantage is entirely (a), there is
nothing to distil: the teacher's whole edge is the answer itself, which the
student can never have. Distillation would then be pulling the student toward
a target with no learnable content -- which is exactly what two runs at
alpha = 36% and alpha = 10% showed, with the t0-t1 gap unmoved at both.

This probe separates them in about two minutes, by decoding the SAME frame
three ways:

    correct GT prompt  -> the teacher as distillation actually uses it
    WRONG mask prompt  -> a different frame's GT. If the score follows the
                          wrong mask, the decoder is copying, not segmenting.
    NO prompt          -> the decoder's standalone ability. This is the number
                          that matters: near zero means the teacher holds no
                          knowledge to transfer.

Measured on checkpoints-16-frames (the shipped baseline, trained WITHOUT
prompt dropout, so it never had to segment unprompted):

    correct GT   0.877 / 0.961 / 0.969
    WRONG mask   0.420 / 0.620 / 0.415
    NO prompt    0.014 / 0.013 / 0.019

A model trained WITH prompt dropout should score higher unprompted -- that is
precisely what dropout is for. Running this on such a checkpoint measures
whether dropout worked, independently of whether distillation did.

Usage:
    python benchmark/probe_teacher.py --ckpt checkpoints-distill-v2/lora_weights.pt
    python benchmark/probe_teacher.py --ckpt X --videos A31_C1 --frames 0 20 40
"""
import argparse
import glob
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.config import TrainingConfig
from src.lora import load_lora_weights
from src.trainer import _load_normalization_stats, build_training_model, detect_device

DATA = os.path.join(REPO, "training_data")


def load_frame(video, idx, mean, std, device):
    imgs = sorted(glob.glob(os.path.join(DATA, "images", video, "*.png")))
    msks = sorted(glob.glob(os.path.join(DATA, "masks", video, "*.png")))
    if idx >= len(imgs):
        return None, None
    im = np.array(Image.open(imgs[idx]).convert("L"), dtype=np.float32) / 255.0
    gt = np.array(Image.open(msks[idx])) > 0

    x = torch.from_numpy(im)[None, None].repeat(1, 3, 1, 1)
    x = F.interpolate(x, (1024, 1024), mode="bilinear", align_corners=False)
    x = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)

    g = torch.from_numpy(gt.astype(np.float32))[None, None]
    g = (F.interpolate(g, (1024, 1024), mode="bilinear", align_corners=False) > 0.5).float()
    return x.to(device), g.to(device)


def iou(logits, gt):
    p = (torch.sigmoid(logits) > 0.5).float()
    inter = (p * gt).sum()
    return float(inter / (p.sum() + gt.sum() - inter + 1e-6))


@torch.no_grad()
def decode(base, x, prompt):
    """One decoder pass in the frame-0 conditioning regime, as _teacher_mask does."""
    bo = base.forward_image(x)
    _, vf, _, fs = base._prepare_backbone_features(bo)
    high_res = [
        t.permute(1, 2, 0).view(t.size(1), t.size(2), *s)
        for t, s in zip(vf[:-1], fs[:-1])
    ]
    # `+ no_mem_embed` mirrors SAM 2's initial conditioning frame exactly
    # (directly_add_no_mem_embed=True); it is a trained parameter.
    feat = vf[-1] + base.no_mem_embed
    pix = feat.permute(1, 2, 0).view(1, base.hidden_dim, *fs[-1])
    out = base._forward_sam_heads(
        backbone_features=pix, point_inputs=None, mask_inputs=prompt,
        high_res_features=high_res, multimask_output=False,
    )
    return out[4]  # high_res_masks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="LoRA weights to probe")
    ap.add_argument("--videos", nargs="*", default=["A91_C1_2", "A31_C1", "A52_C2"])
    ap.add_argument("--frames", nargs="*", type=int, default=[0, 20, 40])
    ap.add_argument("--wrong-offset", type=int, default=40,
                    help="frames ahead to borrow the WRONG mask from")
    args = ap.parse_args()

    device = detect_device()
    cfg = TrainingConfig()
    cfg.data.precompute_features = False
    model, _ = build_training_model(cfg, device)
    load_lora_weights(model.model, args.ckpt, device=device)
    model.eval()
    base = model.model
    mean, std = _load_normalization_stats(cfg)

    print(f"\nProbing {args.ckpt} on {device}\n")
    print(f"{'video':<14}{'frame':>6}{'correct GT':>12}{'WRONG mask':>12}{'NO prompt':>12}")
    print("-" * 56)

    rows = []
    for v in args.videos:
        for fi in args.frames:
            x, g = load_frame(v, fi, mean, std, device)
            if x is None:
                continue
            _, g_wrong = load_frame(v, fi + args.wrong_offset, mean, std, device)
            if g_wrong is None:
                _, g_wrong = load_frame(v, max(0, fi - args.wrong_offset), mean, std, device)
            correct = iou(decode(base, x, g), g)
            wrong = iou(decode(base, x, g_wrong), g) if g_wrong is not None else float("nan")
            none_ = iou(decode(base, x, None), g)
            rows.append((correct, wrong, none_))
            print(f"{v:<14}{fi:>6}{correct:>12.4f}{wrong:>12.4f}{none_:>12.4f}")

    if not rows:
        print("no frames found -- are the videos prepared into training_data/?")
        return
    c, w, n = (float(np.mean([r[i] for r in rows])) for i in range(3))
    print("-" * 56)
    print(f"{'MEAN':<14}{'':>6}{c:>12.4f}{w:>12.4f}{n:>12.4f}\n")

    print("READING THIS:")
    if n < 0.10:
        print(f"  NO prompt = {n:.4f}. The decoder cannot find the neuron unaided, so the")
        print("  teacher's advantage IS the ground-truth mask, not a skill. There is")
        print("  nothing for distillation to transfer, at any weight.")
    elif n < 0.5 * c:
        print(f"  NO prompt = {n:.4f} against {c:.4f} prompted. Some standalone ability,")
        print("  but the prompt still does most of the work.")
    else:
        print(f"  NO prompt = {n:.4f} against {c:.4f} prompted. The decoder segments")
        print("  largely on its own, so the teacher holds knowledge worth distilling.")
    if w < 0.6 * c:
        print(f"  WRONG mask = {w:.4f} vs {c:.4f} correct: the output follows the prompt,")
        print("  i.e. the decoder is refining what it is handed rather than segmenting.")


if __name__ == "__main__":
    main()
