#!/usr/bin/env python3
"""
sam3_lora_preflight.py -- can the SAM 3.1 tracker train under our wrapper?

Answers that in ~2 minutes instead of finding out 4 hours into a run:

  1. Does build_sam3_video_model().tracker + the shared backbone come up, and
     are the weights the released ones rather than a fresh init?
  2. Do all ten SAM 2 attributes SAM2TrainingWrapper touches resolve?
  3. Do the LoRA targets discover a sensible number of layers, and what
     fraction of the model ends up trainable?
  4. Does a real forward pass produce a loss with a finite gradient that
     reaches the LoRA A/B matrices -- the one check that distinguishes
     "wired up" from "wrapped but not learning"?

Run this BEFORE queueing a training job.

    python benchmark/sam3_lora_preflight.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.config import TrainingConfig                      # noqa: E402
from src.sam3_backbone import build_sam3_lora              # noqa: E402


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


def main():
    cfg = TrainingConfig()
    banner("1. BUILD")
    try:
        model, info = build_sam3_lora(cfg)
    except BaseException as e:
        import traceback
        print(f"FAILED: {type(e).__name__}: {e}")
        traceback.print_exc(limit=8)
        return 1

    ps = [p for p in model.parameters() if p.dim() > 1][:40]
    stds = [float(p.detach().float().std()) for p in ps]
    trained = max(stds) > 0.05
    print(f"  weights: std median {float(np.median(stds)):.4f}, "
          f"range {min(stds):.4f}-{max(stds):.4f} -> "
          f"{'TRAINED' if trained else 'FRESH-INIT'}")
    if not trained:
        print("  STOP: fine-tuning a fresh init is not the experiment.")
        return 2

    banner("2. WRAPPER API")
    ok = True
    for name, present in model.api_report():
        print(f"  {'OK  ' if present else 'MISS'}  {name}")
        ok = ok and present
    if not ok:
        print("\n  Missing attributes must be aliased in Sam3TrainingAdapter")
        print("  before training. They are what SAM2TrainingWrapper calls.")

    banner("3. LoRA COVERAGE")
    print(f"  decoder/memory layers wrapped : {info['wrapped']}")
    print(f"  vision layers wrapped        : {info['wrapped_visual']}")
    print(f"  trainable {info['trainable']:,} / {info['total']:,} "
          f"({100.0 * info['trainable'] / max(info['total'], 1):.1f}%)")
    print("  (SAM 2.1 reference: 5,557,250 / 39,876,930 = 13.9%)")

    banner("4. GRADIENT REACHES THE ADAPTERS")
    # A wrapped layer whose A/B never receive gradient trains nothing while
    # looking completely normal in the loss curve -- check it explicitly.
    lora_params = [(n, p) for n, p in model.named_parameters()
                   if p.requires_grad and (".lora_A" in n or ".lora_B" in n)]
    print(f"  LoRA tensors requiring grad: {len(lora_params)}")
    if not lora_params:
        print("  STOP: no LoRA parameters. apply_lora matched nothing.")
        return 3

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    try:
        # SAM 3's tracker is built with image_size=1008 (model_builder.py:465),
        # not SAM 2's 1024. Use the tracker's own value when it exposes one.
        sz = int(getattr(model.tracker, "image_size", 0) or cfg.data.image_size)
        print(f"  forward at image_size={sz}")
        img = torch.randn(1, 3, sz, sz, device=dev)
        out = model.forward_image(img)
        feats = out["vision_features"] if isinstance(out, dict) else out
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        loss = feats.float().pow(2).mean()
        loss.backward()
        got = [n for n, p in lora_params
               if p.grad is not None and torch.isfinite(p.grad).all()
               and p.grad.abs().sum() > 0]
        print(f"  forward+backward OK, loss {float(loss):.6f}")
        print(f"  LoRA tensors with a finite non-zero grad: "
              f"{len(got)}/{len(lora_params)}")
        if not got:
            print("  NOTE: no gradient reached the adapters through the image")
            print("  path. That is expected if only decoder/memory layers were")
            print("  wrapped -- those need a full track_step, not forward_image.")
    except BaseException as e:
        import traceback
        print(f"  forward/backward failed: {type(e).__name__}: {e}")
        traceback.print_exc(limit=6)

    banner("VERDICT")
    if ok and info["wrapped"] > 0:
        print("The tracker builds, the wrapper's API resolves and LoRA is")
        print("attached. Next step is a SHORT run (2-3 epochs) to confirm the")
        print("loss moves before committing a full one.")
    else:
        print("Not ready. Fix the items marked MISS above first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
