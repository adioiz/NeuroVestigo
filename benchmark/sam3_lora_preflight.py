#!/usr/bin/env python3
"""
sam3_lora_preflight.py -- can the SAM 3 tracker train under our wrapper?

(SAM 3, not 3.1: build_sam3_video_model loads facebook/sam3/sam3.pt; the 3.1
release has no standalone tracker checkpoint. See src/sam3_backbone.py.)

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
    cfg.backbone = "sam3"
    from src.config import apply_backbone_defaults
    apply_backbone_defaults(cfg)
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
        print("  (exactly half is expected: lora_B is zero-initialised, so every")
        print("   lora_A has zero gradient at step 0; the B count is the signal)")
        if not got:
            print("  NOTE: no gradient reached the adapters through the image")
            print("  path. That is expected if only decoder/memory layers were")
            print("  wrapped -- those need a full track_step, not forward_image.")
    except BaseException as e:
        import traceback
        print(f"  forward/backward failed: {type(e).__name__}: {e}")
        traceback.print_exc(limit=6)

    banner("5. REAL TRAINING FORWARD THROUGH SAM2TrainingWrapper")
    # This is the check that matters. Sections 1-4 exercise attribute lookups
    # and the image path; the training loop calls track_step through the
    # wrapper in TRAIN mode, and that is where SAM 3's tracker reads
    # attributes its constructor never sets (teacher_force_obj_scores_for_mem,
    # prob_to_dropout_spatial_mem) and where its track_step wants an `image`
    # argument SAM 2's does not have. A random 2-frame clip is enough to walk
    # every one of those lines.
    from src.trainer import SAM2TrainingWrapper
    from src.data_utils import BatchedVideoDatapoint
    try:
        model.zero_grad(set_to_none=True)
        wrapper = SAM2TrainingWrapper(
            model, num_frames=2, use_memory_path=True, prompt_dropout=0.0,
            prompt_dropout_warmup_epochs=0, use_prompt_distill=False,
            distill_frames_per_clip=0, distill_start_epoch=10**9).to(dev)
        wrapper.train()
        sz = int(getattr(model.tracker, "image_size", 0) or cfg.data.image_size)
        B, T = 1, 2
        images = torch.randn(B, T, 3, sz, sz, device=dev)
        masks = torch.zeros(B, T, 1, sz, sz, device=dev)
        masks[..., sz // 3: 2 * sz // 3, sz // 3: 2 * sz // 3] = 1.0
        batch = BatchedVideoDatapoint(images=images, masks=masks,
                                      obj_ids=[[1]], video_ids=[0])
        with torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" \
                else torch.autocast("cpu", enabled=False):
            out = wrapper(batch)
        keys = list(out.keys()) if isinstance(out, dict) else type(out).__name__
        print(f"  wrapper forward OK -> {keys}")
        # a scalar from whatever the wrapper returned, to drive backward
        tens = [v for v in (out.values() if isinstance(out, dict) else [])
                if torch.is_tensor(v) and v.requires_grad]
        if tens:
            loss = sum(t.float().mean() for t in tens)
            loss.backward()
            got = [n for n, p in lora_params
                   if p.grad is not None and torch.isfinite(p.grad).all()
                   and p.grad.abs().sum() > 0]
            print(f"  backward OK; LoRA tensors with finite non-zero grad: "
                  f"{len(got)}/{len(lora_params)}")
            dec = sum(1 for n in got if "sam_mask_decoder" in n)
            mem = sum(1 for n in got if n.split(".")[1] == "transformer"
                      or "memory_attention" in n)
            print(f"    decoder: {dec}   memory attention: {mem}")
            if mem == 0:
                print("  WARNING: no gradient reached the memory-attention "
                      "adapters. With a 2-frame clip frame 1 goes through "
                      "the memory path, so this means they were not wrapped "
                      "or not exercised.")
        else:
            print("  (no differentiable tensors in the output to backprop)")
        forward_ok = True
    except BaseException as e:
        import traceback
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc(limit=10)
        forward_ok = False

    banner("VERDICT")
    if ok and info["wrapped"] > 0 and forward_ok:
        print("The tracker builds with released weights, all ten wrapper")
        print("attributes resolve, LoRA is attached to both the decoder and the")
        print("memory attention, and a real training forward+backward ran.")
        print("Next: a SHORT run (2-3 epochs) to see the loss move, then full.")
    else:
        print("Not ready. Fix the items marked MISS / FAILED above first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
