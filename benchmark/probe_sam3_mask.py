#!/usr/bin/env python3
"""
probe_sam3_mask.py -- can SAM 3.1 be seeded with a frame-0 MASK?

WHY THIS EXISTS
---------------
The paper currently states that SAM 3.1 cannot be initialised the way our model
and SAM 2 are. That is verified only of the DOCUMENTED video session API
(handle_request/add_prompt reads text, points, bounding_boxes -- no mask key).
But `--probe` also found `Sam3TrackerPredictor.add_new_mask` and
`Sam3VideoTrackingMultiplexDemo.add_new_masks`, and the winning point path
already logs "Running SAM2 propagation for objects [1]" -- so a SAM 2-style
mask-promptable tracker is in there. If it can be driven standalone, SAM 3.1
can be benchmarked under exactly our protocol and the disclosed protocol
asymmetry disappears, which is a stronger result than the disclaimer.

This probe answers that question and nothing else. It never writes into
benchmark/predictions/.

STAGES -- each prints what it found and keeps going where it can:
  A  find the tracker class and factory, and print the real signatures
  B  build a tracker (several construction paths tried in turn)
  C  initialise state on a frames directory
  D  seed frame 0 with the ground-truth mask
  E  propagate, and report per-frame foreground

    conda activate sam3-env      # NOT neurotrack-env
    python benchmark/probe_sam3_mask.py --video A81_C1 --limit-frames 12

A clean run ending "STAGE E OK" means the paper's claim must be softened and
the row can be re-run mask-prompted. Any earlier stop is itself the answer:
paste the output back.
"""
from __future__ import annotations

import argparse
import glob
import inspect
import os
import sys
import traceback

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "training_data")


def load_gt(name, n=None):
    files = sorted(glob.glob(os.path.join(DATA, "masks", name, "*.png")))
    if n:
        files = files[:n]
    return np.stack([np.array(Image.open(f)) > 0 for f in files])


def show(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def sig(obj):
    try:
        return str(inspect.signature(obj))
    except (ValueError, TypeError):
        return "(signature unavailable)"


# ---------------------------------------------------------------- stage A
def find_tracker():
    """Locate every mask-accepting tracker entry point and print its signature."""
    import importlib, pkgutil
    import sam3
    root = os.path.dirname(sam3.__file__)

    classes, factories = {}, {}
    for m in pkgutil.walk_packages([root], prefix="sam3."):
        try:
            mod = importlib.import_module(m.name)
        except BaseException as e:
            continue
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if obj.__module__ != m.name:
                continue
            meths = [n for n, _ in inspect.getmembers(obj, inspect.isfunction)]
            if any("add_new_mask" in n or "add_new_masks" in n for n in meths):
                classes[f"{m.name}.{name}"] = obj
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            if "tracker" in name.lower() and name.startswith("build"):
                factories[f"{m.name}.{name}"] = obj

    print("mask-accepting classes:")
    for k, c in classes.items():
        print(f"  {k}")
        for n, f in inspect.getmembers(c, inspect.isfunction):
            if any(s in n for s in ("add_new_mask", "init_state", "propagate",
                                    "reset", "add_new_points")):
                print(f"      .{n}{sig(f)}")
    print("\ntracker factories:")
    for k, f in factories.items():
        print(f"  {k}{sig(f)}")
    if not classes:
        print("  NONE -- the build has no mask-accepting tracker; the paper's")
        print("  claim stands as written.")
    return classes, factories


# ---------------------------------------------------------------- stage B
def build(factories, classes, ckpt):
    """Try each plausible construction path; report every failure verbatim."""
    attempts = []
    for k, f in factories.items():
        for kw in ({"with_backbone": True}, {}, {"with_backbone": False}):
            attempts.append((f"{k}({', '.join(f'{a}={b}' for a, b in kw.items())})",
                             lambda f=f, kw=kw: f(**kw)))
    # the documented video predictor may expose the tracker as an attribute
    def via_video():
        from sam3.model_builder import build_sam3_video_predictor
        vp = build_sam3_video_predictor()
        for attr in ("tracker", "_tracker", "video_tracker", "sam2_tracker"):
            if hasattr(vp, attr):
                print(f"    (found .{attr} on the video predictor)")
                return getattr(vp, attr)
        raise RuntimeError(f"no tracker attribute on {type(vp).__name__}; "
                           f"has {[a for a in dir(vp) if 'track' in a.lower()]}")
    attempts.append(("build_sam3_video_predictor().<tracker attr>", via_video))

    for label, fn in attempts:
        print(f"  trying {label} ...")
        try:
            t = fn()
            print(f"    OK -> {type(t).__name__}")
            return t
        except BaseException as e:
            print(f"    {type(e).__name__}: {str(e)[:160]}")
    return None


# ---------------------------------------------------------------- stages C-E
def drive(tracker, frames_dir, gt):
    print("  init_state ...")
    state = None
    for kw in ({"video_path": frames_dir}, {"video_path": frames_dir, "async_loading_frames": False}):
        try:
            state = tracker.init_state(**kw); print("    OK"); break
        except BaseException as e:
            print(f"    {type(e).__name__}: {str(e)[:160]}")
    if state is None:
        print("  STOP: cannot initialise state -- report this output.")
        return False

    print("  add_new_mask(frame 0) ...")
    ok = False
    for kw in ({"inference_state": state, "frame_idx": 0, "obj_id": 1, "mask": gt[0]},
               {"inference_state": state, "frame_idx": 0, "obj_id": 1, "mask": gt[0].astype(np.uint8)}):
        try:
            tracker.add_new_mask(**kw); print("    OK"); ok = True; break
        except BaseException as e:
            print(f"    {type(e).__name__}: {str(e)[:200]}")
    if not ok:
        print("  STOP: the tracker exists but will not take a mask -- that is the")
        print("  answer, and the paper's claim stands. Report this output.")
        return False

    print("  propagate_in_video ...")
    try:
        n = 0
        for frame_idx, obj_ids, logits in tracker.propagate_in_video(state):
            m = np.asarray(logits.detach().cpu() if hasattr(logits, "detach") else logits)
            m = (m > 0).reshape(-1, *m.shape[-2:])
            fg = float(np.any(m, 0).mean())
            iou = None
            if frame_idx < len(gt):
                a = np.any(m, 0)
                if a.shape == gt[frame_idx].shape:
                    u = (a | gt[frame_idx]).sum()
                    iou = (a & gt[frame_idx]).sum() / u if u else float("nan")
            print(f"    frame {frame_idx:>3}  objs={len(obj_ids)}  fg={fg:.4f}"
                  + (f"  IoU={iou:.3f}" if iou is not None else ""))
            n += 1
        print(f"  OK -- {n} frames propagated from a MASK prompt")
        return True
    except BaseException as e:
        print(f"    {type(e).__name__}: {str(e)[:200]}")
        traceback.print_exc(limit=3)
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default="A81_C1")
    ap.add_argument("--limit-frames", type=int, default=12)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    show("STAGE A -- what mask-accepting tracker exists?")
    classes, factories = find_tracker()
    if not classes and not factories:
        print("\nRESULT: no mask-promptable tracker in this build.")
        return

    show("STAGE B -- can it be built standalone?")
    tracker = build(factories, classes, args.checkpoint)
    if tracker is None:
        print("\nRESULT: the class exists but no construction path worked. That is")
        print("a defensible basis for the paper's claim -- quote the failures above.")
        return

    show("STAGE C/D/E -- init, seed with a mask, propagate")
    frames_dir = os.path.join(DATA, "images", args.video)
    gt = load_gt(args.video, args.limit_frames)
    print(f"  video={args.video}  frames={len(gt)}  shape={gt.shape[1:]}")
    ok = drive(tracker, frames_dir, gt)

    show("RESULT")
    if ok:
        print("STAGE E OK -- SAM 3.1 CAN be seeded with a frame-0 mask.")
        print("The paper must stop saying it cannot, and the row should be re-run")
        print("mask-prompted, which removes the protocol asymmetry entirely.")
    else:
        print("Stopped before propagating from a mask. The paper's claim stands,")
        print("but phrase it as 'no supported path we could drive', not as a")
        print("statement about the architecture. Paste this output back.")


if __name__ == "__main__":
    main()
