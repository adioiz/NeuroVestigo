#!/usr/bin/env python3
"""
run_ours.py -- inference runner for the fine-tuned SAM 2 + dual-rank LoRA model.

Reuses the project's own build_training_model + LoRA loader + validation
transform, but propagates FULL sequences (frame-0 GT mask as the prompt ->
track_step through every frame) instead of the 8-frame training clips. Exports
one predicted mask stack per sequence, aligned to the ground-truth resolution,
that benchmark/eval_masks.py can score directly.

    python benchmark/run_ours.py                      # all val videos, full length
    python benchmark/run_ours.py --videos A31_C1 --limit-frames 12   # quick smoke test
    python benchmark/run_ours.py --eval               # also print IoU/Dice/clDice/cbDice

Predicted stacks -> benchmark/predictions/ours/<name>.tif  (uint8, 1 = neuron)
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # SAM2 has a few MPS-unsupported ops
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.config import TrainingConfig
from src.trainer import build_training_model, detect_device, _load_normalization_stats
from src.lora import load_lora_weights
from src.neuron_dataset import NeuronVideoDataset, neuron_collate_fn
from src.augmentations import VideoValidationTransform

DATA_DIR = os.path.join(REPO, "training_data")
DEFAULT_CKPT = os.path.join(REPO, "checkpoints", "lora_weights.pt")
OUT_DIR = os.path.join(REPO, "benchmark", "predictions", "ours")


def _seq_len(name: str) -> int:
    import glob
    return len(glob.glob(os.path.join(DATA_DIR, "images", name, "*.png")))


def build_model(ckpt: str, device: str, load_lora: bool = True):
    """Build SAM 2 + dual-rank LoRA. When `load_lora` is False the adapters stay
    at their zero-init (LoRA-B=0 -> exact identity), giving stock SAM 2 through
    the same harness -- a controlled zero-shot baseline that isolates exactly
    what the fine-tuned adapters contribute."""
    config = TrainingConfig()
    config.use_memory_attention_path = True
    config.data.precompute_features = False
    model, _ = build_training_model(config, device)
    # FAIR PROTOCOL (identical across models): frame 0 is the GIVEN prompt, so
    # seed it by passing the GT mask straight through to output + memory. This
    # gives every model the same clean seed; we then score only propagated
    # frames (t>=1). (Score with --skip-first-frame / score.py --skip-first-frame.)
    model.model.use_mask_input_as_output_without_sam = True
    if load_lora:
        load_lora_weights(model.model, ckpt, device=device)  # inner SAM2 -> normalized keys match
        model.model.pred_obj_scores = False   # our obj-presence head is untrained -> keep off
        print("  [ours] fine-tuned adapters loaded; frame-0 GT passthrough seed")
    else:
        model.model.pred_obj_scores = True    # stock SAM2's occlusion head IS pretrained -> fair to use
        print("  [no-lora] stock SAM 2 + native occlusion head -> fair zero-shot baseline")
    model.eval()
    norm_mean, norm_std = _load_normalization_stats(config)
    return model, config, norm_mean, norm_std


@torch.no_grad()
def _propagate_stream(model, batch, device: str) -> torch.Tensor:
    """Frame-by-frame track_step propagation (memory-frugal for MPS).

    Mirrors SAM2TrainingWrapper._forward_memory but encodes ONE frame at a time
    so the encoder's per-frame activations don't accumulate across the whole
    sequence (the wrapper's single forward_image over all T frames OOMs on MPS).
    Returns predicted logits [T, H, W] for the single target object.
    """
    sam = model.model
    images = batch.images                       # [B, T, C, H, W]
    gt = batch.masks                            # [B, T, N, H, W]
    B, T, C, H, W = images.shape
    N = gt.shape[2]
    BN = B * N
    frame0 = gt[:, 0].reshape(BN, 1, H, W).float()
    output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    preds = []
    for t in range(T):
        backbone_t = sam.forward_image(images[:, t])          # encode just frame t
        _, vf, vpe, fs = sam._prepare_backbone_features(backbone_t)
        vf_bn = [x.repeat_interleave(N, dim=1) for x in vf]
        vpe_bn = [x.repeat_interleave(N, dim=1) for x in vpe]
        out_t = sam.track_step(
            frame_idx=t, is_init_cond_frame=(t == 0),
            current_vision_feats=vf_bn, current_vision_pos_embeds=vpe_bn,
            feat_sizes=fs, point_inputs=None,
            mask_inputs=(frame0 if t == 0 else None),
            output_dict=output_dict, num_frames=T, run_mem_encoder=True,
        )
        output_dict["cond_frame_outputs" if t == 0 else "non_cond_frame_outputs"][t] = out_t
        preds.append(out_t["pred_masks_high_res"].detach().to("cpu"))
        del backbone_t, vf, vpe, vf_bn, vpe_bn
        if device == "mps" and (t % 8 == 0):
            torch.mps.empty_cache()
    return torch.stack(preds, dim=1).squeeze(2)[0]            # [T, H, W] logits on CPU


@torch.no_grad()
def infer_sequence(model, config, norm_mean, norm_std, name: str,
                   device: str, limit_frames: int | None = None) -> np.ndarray:
    """Full-sequence propagation. Returns a (T, H_gt, W_gt) uint8 mask stack."""
    n = _seq_len(name)
    if limit_frames:
        n = min(n, limit_frames)
    transform = VideoValidationTransform(image_size=config.data.image_size,
                                         normalize_mean=norm_mean, normalize_std=norm_std)
    ds = NeuronVideoDataset(DATA_DIR, [name], num_frames=n,
                            max_objects=config.data.max_objects, transform=transform,
                            clip_stride=n, cache_in_memory=False)
    batch, _ = neuron_collate_fn([ds[0]])
    batch = batch.to(device)

    pred = _propagate_stream(model, batch, device)   # [T, H, W] logits @ 1024 (CPU)
    binary = (pred > 0).float().unsqueeze(1)          # [T,1,H,W]

    # native GT resolution from the mask PNGs (same as neuron-labeling GT)
    from PIL import Image
    gt0 = np.array(Image.open(os.path.join(DATA_DIR, "masks", name, "00000.png")))
    H_gt, W_gt = gt0.shape[-2:]
    resized = F.interpolate(binary, size=(H_gt, W_gt), mode="nearest")[:, 0]
    return (resized.cpu().numpy() > 0.5).astype(np.uint8)


def main():
    import tifffile
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="*", default=None,
                    help="video names (default: the config val split)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--no-lora", action="store_true",
                    help="stock SAM 2 zero-shot baseline (do not load fine-tuned adapters)")
    ap.add_argument("--limit-frames", type=int, default=None,
                    help="cap sequence length (quick smoke test)")
    ap.add_argument("--out-dir", default=None,
                    help="default: predictions/ours, or predictions/sam2_zeroshot with --no-lora")
    ap.add_argument("--eval", action="store_true",
                    help="score predictions vs GT after inference")
    args = ap.parse_args()

    device = detect_device()
    print(f"device: {device}")
    if args.videos:
        videos = args.videos
    else:
        from benchmark.data import Dataset
        videos = Dataset().split("val")   # authoritative val split (from training_config.json)
    print(f"videos: {videos}")
    out_dir = args.out_dir or (os.path.join(REPO, "benchmark", "predictions",
                               "sam2_zeroshot" if args.no_lora else "ours"))
    os.makedirs(out_dir, exist_ok=True)
    args.out_dir = out_dir

    tag = "stock SAM 2 (zero-shot)" if args.no_lora else "fine-tuned SAM 2 + LoRA"
    print(f"building model: {tag} ...")
    model, config, nm, ns = build_model(args.ckpt, device, load_lora=not args.no_lora)

    for v in videos:
        print(f"  propagating {v} ...", flush=True)
        masks = infer_sequence(model, config, nm, ns, v, device, args.limit_frames)
        out_path = os.path.join(args.out_dir, f"{v}.tif")
        tifffile.imwrite(out_path, masks)
        fg = float((masks > 0).mean())
        print(f"    {v}: {masks.shape[0]} frames, fg={fg:.4f} -> {out_path}")

    if args.eval:
        from benchmark.eval_masks import evaluate_stack, aggregate
        from benchmark.data import Dataset
        ds = Dataset()
        results = []
        print(f"\n{'sequence':<14}{'IoU':>9}{'Dice':>9}{'clDice':>9}{'cbDice':>9}")
        print("-" * 50)
        for v in videos:
            gt = ds.load_gt(v)
            pred = tifffile.imread(os.path.join(args.out_dir, f"{v}.tif"))
            r = evaluate_stack(pred, gt, name=v, with_cbdice=True)
            results.append(r)
            print(f"{v:<14}" + "".join(f"{r.mean(m):>9.4f}"
                  for m in ["iou", "dice", "cldice", "cbdice"]))
        agg = aggregate(results)
        print("-" * 50)
        print(f"{'MEAN':<14}" + "".join(f"{agg[m]['mean']:>9.4f}"
              for m in ["iou", "dice", "cldice", "cbdice"]))


if __name__ == "__main__":
    main()
