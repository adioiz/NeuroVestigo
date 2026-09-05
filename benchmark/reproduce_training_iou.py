#!/usr/bin/env python3
"""
reproduce_training_iou.py -- prove the loaded weights are correct.

Runs the EXACT training-validation protocol (8-frame clips, clip_stride=4, the
wrapper's training-time flags) and computes per-frame IoU the same way
validate_epoch does -- but memory-frugally (batch=1, per-frame encoding, no
loss) so it fits on MPS. If this reports ~0.84 IoU it proves
checkpoints/lora_weights.pt is the right checkpoint, and the lower
full-sequence number is purely the harder 73-frame propagation protocol.
"""
import os, sys
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import torch
from src.config import TrainingConfig
from src.trainer import build_training_model, detect_device, _load_normalization_stats
from src.lora import load_lora_weights
from src.neuron_dataset import NeuronVideoDataset, neuron_collate_fn
from src.augmentations import VideoValidationTransform
from benchmark.data import Dataset
from benchmark.run_ours import _propagate_stream

CKPT = os.path.join(REPO, "checkpoints", "lora_weights.pt")


def iou(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    u = np.logical_or(p, g).sum()
    return 1.0 if u == 0 else float(np.logical_and(p, g).sum() / u)


def main():
    config = TrainingConfig()
    config.data.val_videos = Dataset().split("val")
    device = detect_device()
    print(f"device={device}  protocol: num_frames={config.data.num_frames}, "
          f"clip_stride={config.data.clip_stride} (TRAINING val protocol)")

    nm, ns = _load_normalization_stats(config)
    tf = VideoValidationTransform(config.data.image_size, nm, ns)
    model, _ = build_training_model(config, device)     # wrapper training-time flags (unchanged)
    load_lora_weights(model.model, CKPT, device=device)
    model.eval()

    per_video = {}
    all_ious = []
    for v in config.data.val_videos:
        ds = NeuronVideoDataset(os.path.join(REPO, "training_data"), [v],
                                num_frames=config.data.num_frames,
                                max_objects=config.data.max_objects, transform=tf,
                                clip_stride=config.data.clip_stride, cache_in_memory=False)
        vids = []
        for i in range(len(ds)):
            batch, _ = neuron_collate_fn([ds[i]])
            batch = batch.to(device)
            logits = _propagate_stream(model, batch, device)          # [T,H,W] cpu
            pred = (logits > 0).numpy()
            gt = (batch.masks[0, :, 0] > 0).cpu().numpy()             # [T,H,W] @1024
            for t in range(pred.shape[0]):
                if not pred[t].any() and not gt[t].any():
                    continue
                vids.append(iou(pred[t], gt[t]))
            if device == "mps":
                torch.mps.empty_cache()
        per_video[v] = float(np.mean(vids))
        all_ious.extend(vids)
        print(f"  {v:<12} clips={len(ds):>3}  mean_IoU={per_video[v]:.4f}")

    print("\n=== TRAINING-PROTOCOL VALIDATION (8-frame clips, all frames) ===")
    print(f"  mean val_iou = {np.mean(all_ious):.4f}   (reported Run-7: ~0.84)")
    print(f"  per_video = { {k: round(x,3) for k,x in per_video.items()} }")


if __name__ == "__main__":
    main()
