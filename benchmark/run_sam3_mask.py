#!/usr/bin/env python3
"""
run_sam3_mask.py -- SAM 3.1 seeded with a frame-0 MASK prompt.

Follows Meta's own recipe for SAM 2-style video tasks, from
`sam3/examples/sam3_for_sam2_video_task_example.ipynb` (cell 11), which states:

    "SAM 3 also supports directly using a mask prompt as input via the
     `add_new_mask` method in the `Sam3TrackerPredictor` class."

    sam3_model = build_sam3_video_model()
    predictor  = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    inference_state = predictor.init_state(video_path=video_path)

Three points that cost earlier attempts, all settled by that notebook:

  * `build_sam3_video_model`, not `build_sam3_predictor`. The multiplex
    predictor's tracker is a Sam3VideoTrackingMultiplexDemo whose init_state
    takes cached_features and no video_path.
  * The tracker is built with `backbone=None` (model_builder.py:459) because
    the vision encoder is SHARED with the detector; the caller attaches it.
    That is the AttributeError: 'NoneType' has no attribute 'forward_image'.
  * `propagate_in_video` yields FIVE values --
    (frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores) --
    and takes propagate_preflight=True rather than a separate preflight call.

WHY IT MATTERS: every other SAM 3.1 row we report is seeded through the session
API, whose add_prompt takes text, points and boxes but no mask
(sam3_base_predictor.py:149-163). Our model and zero-shot SAM 2 both get the
frame-0 mask, so SAM 3.1 has been scored on a weaker seed and 0.4712 is an
underestimate.

    python benchmark/run_sam3_mask.py --videos A81_C1
    python benchmark/run_sam3_mask.py --videos <test set> \
        --out-dir benchmark/predictions/sam3_maskfirst
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_sam3 import DATA, load_frames, load_gt          # noqa: E402
from probe_sam3_mask import disable_cudnn, jpeg_mirror   # noqa: E402


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


def build_predictor():
    """Cell 11 of the SAM 2-task notebook, verbatim."""
    import torch
    from sam3.model_builder import build_sam3_video_model

    sam3_model = build_sam3_video_model()
    predictor = sam3_model.tracker
    # build_tracker(with_backbone=False) leaves backbone=None on purpose: the
    # vision encoder is shared with the detector and attached by the caller.
    predictor.backbone = sam3_model.detector.backbone
    print(f"  predictor: {type(predictor).__name__}")
    print(f"  backbone : {type(predictor.backbone).__name__}")

    # Trained or fresh? A fresh init measured std 0.0179-0.0238; released
    # weights 0.0135-0.1458. Any IoU from an untrained tracker is meaningless.
    ps = [p for p in predictor.parameters() if p.dim() > 1][:40]
    stds = [float(p.detach().float().std()) for p in ps]
    trained = max(stds) > 0.05
    print(f"  weights  : std median {float(np.median(stds)):.4f}, "
          f"range {min(stds):.4f}-{max(stds):.4f} -> "
          f"{'TRAINED' if trained else 'FRESH-INIT'}")
    if torch.cuda.is_available():
        predictor = predictor.to("cuda")
    return predictor, trained


def run_video(predictor, video, limit, frames_dir, gts):
    """Seed frame 0 with the GT mask, then track. Returns [T,H,W] uint8."""
    import torch
    from contextlib import nullcontext

    def amp():
        return (torch.autocast("cuda", dtype=torch.bfloat16)
                if torch.cuda.is_available() else nullcontext())

    T, H, W = len(gts), gts[0].shape[0], gts[0].shape[1]
    # SAM 2-style loaders glob for .jpg; our frames are .png.
    jpg = jpeg_mirror(frames_dir, T)

    if hasattr(predictor, "reset_state"):
        try:
            predictor.reset_state(getattr(predictor, "_last_state", None))
        except BaseException:
            pass
    state = predictor.init_state(video_path=jpg)
    print(f"  init_state OK ({T} frames)")

    m0 = torch.from_numpy(gts[0].astype(bool))
    with amp():
        predictor.add_new_mask(inference_state=state, frame_idx=0,
                               obj_id=1, mask=m0)
    print("  add_new_mask OK -- frame-0 GT mask accepted")

    out = np.zeros((T, H, W), np.uint8)
    seen = 0
    with amp():
        # FIVE values, and preflight is a kwarg here, not a separate call.
        for (frame_idx, obj_ids, _low_res, video_res_masks,
             _scores) in predictor.propagate_in_video(
                state, start_frame_idx=0, max_frame_num_to_track=T,
                reverse=False, propagate_preflight=True):
            if frame_idx >= T:
                continue
            m = (video_res_masks > 0.0).squeeze(1).cpu().numpy()
            m = np.any(m.reshape(-1, *m.shape[-2:]), axis=0)
            if m.shape != (H, W):
                from run_sam3 import _resize_stack
                m = _resize_stack(m[None].astype(np.uint8), H, W)[0] > 0
            out[frame_idx] = m.astype(np.uint8)
            seen += 1
    print(f"  propagated {seen} frames")
    return out, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=["A81_C1"])
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--cudnn", action="store_true")
    args = ap.parse_args()

    if not args.cudnn:
        disable_cudnn()

    banner("BUILD -- build_sam3_video_model().tracker + shared backbone")
    predictor, trained = build_predictor()
    if not trained:
        print("\nRefusing to continue: the tracker's weights look fresh-init.")
        return 4

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    all_iou = []
    for v in args.videos:
        banner(f"{v} -- frame-0 mask prompt")
        frames_dir = os.path.join(DATA, "images", v)
        n = len(load_frames(v, args.limit_frames))
        gts = load_gt(v, n)
        try:
            stack, seen = run_video(predictor, v, args.limit_frames,
                                    frames_dir, gts)
        except BaseException as e:
            import traceback
            print(f"  FAILED: {type(e).__name__}: {e}")
            traceback.print_exc(limit=6)
            continue
        if seen == 0:
            print("  no frames returned"); continue
        ious = [iou(stack[t].astype(bool), gts[t]) for t in range(len(gts))]
        mean = float(np.nanmean(ious))
        all_iou.append(mean)
        print(f"  frame 0 IoU {ious[0]:.4f}"
              + (f"   frame 1 IoU {ious[1]:.4f}" if len(ious) > 1 else ""))
        print(f"  mean IoU over {len(ious)} frames: {mean:.4f}")
        if args.out_dir:
            import tifffile
            p = os.path.join(args.out_dir, f"{v}.tif")
            tifffile.imwrite(p, stack)
            print(f"  -> {p}  fg={float((stack > 0).mean()):.4f}")

    if all_iou:
        banner("SUMMARY")
        print(f"mask-prompted SAM 3.1: mean IoU {float(np.mean(all_iou)):.4f} "
              f"over {len(all_iou)} recording(s)")
        print("Compare with the point-prompted row (0.4712) -- if this is "
              "higher, that row understates SAM 3.1 and the paper's protocol "
              "note needs revising.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
