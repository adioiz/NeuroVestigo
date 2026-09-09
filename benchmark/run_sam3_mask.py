#!/usr/bin/env python3
"""
run_sam3_mask.py -- drive SAM 3.1 with a frame-0 MASK prompt.

WHY THIS EXISTS
---------------
Every other SAM 3.1 row we report goes through the session interface, and
`add_prompt` there takes text, points and boxes but no mask
(sam3/model/sam3_base_predictor.py:149-163). So SAM 3.1 was benchmarked with a
weaker seed than our model and zero-shot SAM 2, which both get the frame-0
mask.

Masks ARE an accepted SAM 3 visual prompt -- `Sam3TrackerPredictor.add_new_mask`
and `Sam3VideoTrackingMultiplexDemo.add_new_masks` are real entry points. They
are simply not reachable from the session API, because `start_session` builds a
multiplex state and those methods want a demo-path state; feeding one to the
other fails with KeyError: 'obj_id_to_idx'.

This script closes that gap: it builds the demo-path state, which is what the
mask entry points expect, on a tracker carrying the RELEASED weights.

THREE ROUTES, tried in order, first success wins. Each reports why it failed,
so one job settles which is viable rather than one job per guess.

  A  demo init_state -- the inner tracker from build_sam3_predictor() is a
     Sam3VideoTrackingMultiplexDemo, and its base VideoTrackingMultiplexDemo
     has init_state(video_path, ...) which loads frames itself. Call that base
     init_state on the weights-loaded instance.

  B  build_tracker + transplanted weights -- build_tracker(..., with_backbone=
     True) returns a Sam3TrackerPredictor, which has both add_new_mask AND an
     init_state taking video_path, but no checkpoint argument (its weights are
     a fresh init, std 0.0179-0.0238). Copy the released weights into it from
     the predictor and verify the copy actually landed.

  C  session state -- kept last and expected to fail; recorded so the log shows
     it was tried on the same run.

WEIGHTS ARE CHECKED, NOT ASSUMED. A fresh init would still answer "does the API
accept a mask" but any IoU from it is meaningless, so the weight signature is
printed for whichever route wins and the run refuses to save predictions from
an untrained tracker.

    python benchmark/run_sam3_mask.py --videos A81_C1
    python benchmark/run_sam3_mask.py --videos <test videos> --out-dir \
        benchmark/predictions/sam3_maskfirst
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
import textwrap

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_sam3 import (DATA, _binarize, _resize_stack, load_frames,  # noqa: E402
                      load_gt)
from probe_sam3_mask import disable_cudnn, jpeg_mirror  # noqa: E402


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


# --------------------------------------------------------------------------
def find_inner(vp):
    """The object that actually owns add_new_mask(s), and its owning wrapper.

    build_sam3_predictor() returns Sam3MultiplexVideoPredictor; .model.tracker
    is a Sam3MultiplexPredictorWrapper that proxies attributes to an inner
    Sam3VideoTrackingMultiplexDemo. We want the inner one: the wrapper's
    __getattr__ hides which class we are really calling, and the base-class
    init_state trick in route A needs the real instance.
    """
    import torch

    def accepts(v):
        return any(callable(getattr(v, n, None))
                   for n in ("add_new_mask", "add_new_masks"))

    seen, out = set(), []

    def walk(o, path, d=0):
        if d > 6 or id(o) in seen:
            return
        seen.add(id(o))
        kids = []
        if isinstance(o, torch.nn.Module):
            kids += list(o.named_children())
        dd = getattr(o, "__dict__", None)
        if isinstance(dd, dict):
            kids += [(n, v) for n, v in dd.items() if not n.startswith("__")]
        for n, v in kids:
            if v is None or isinstance(v, (str, bytes, int, float, bool,
                                           torch.Tensor)):
                continue
            if inspect.isroutine(v) or inspect.isclass(v):
                continue
            if accepts(v):
                out.append((f"{path}{n}", v))
                # keep descending: the wrapper proxies to the real instance,
                # and the real instance is the one we want.
            walk(v, f"{path}{n}.", d + 1)

    walk(vp, "")
    for p, v in out:
        print(f"  mask-capable: .{p} -> {type(v).__name__}")
    if not out:
        return None
    # Prefer a concrete demo/tracker class over a proxy wrapper.
    for p, v in out:
        if "Wrapper" not in type(v).__name__:
            print(f"  -> using .{p} ({type(v).__name__})")
            return v
    p, v = out[0]
    inner = getattr(v, "tracker", None) or getattr(v, "model", None)
    if inner is not None and accepts(inner):
        print(f"  -> unwrapping the proxy to {type(inner).__name__}")
        return inner
    print(f"  -> using .{p} ({type(v).__name__})")
    return v


def weight_signature(m, label):
    """Trained or fresh? A fresh init is tightly clustered around its gain."""
    import torch
    try:
        ps = [p for p in m.parameters() if p.dim() > 1][:40]
        if not ps:
            return None
        stds = [float(p.detach().float().std()) for p in ps]
        lo, hi, med = min(stds), max(stds), float(np.median(stds))
        trained = hi > 0.05          # fresh init measured 0.0179-0.0238
        print(f"  {label}: std median {med:.4f}, range {lo:.4f}-{hi:.4f}"
              f"  -> {'TRAINED' if trained else 'looks FRESH-INIT'}")
        return trained
    except BaseException as e:
        print(f"  {label}: could not inspect ({type(e).__name__})")
        return None


def dump_source(fn, name, lines=40):
    """Print the entry point's own code. If a route fails, the reason is here
    rather than one GPU job away."""
    try:
        src = inspect.getsource(fn)
    except BaseException:
        return
    head = "".join(src.splitlines(True)[:lines])
    print(f"\n--- {name} (first {lines} lines) ---")
    print(textwrap.indent(head.rstrip(), "  "))
    print("--- end ---\n")


# ------------------------------------------------------------------ routes
def route_demo(vp, inner, frames_dir, n):
    """A: the demo base class loads video itself."""
    from sam3.model.video_tracking_multiplex_demo import VideoTrackingMultiplexDemo
    jpg = jpeg_mirror(frames_dir, n)
    base = VideoTrackingMultiplexDemo
    print(f"  isinstance of {base.__name__}: {isinstance(inner, base)}")
    attempts = [
        ("kwargs", lambda: base.init_state(inner, video_path=jpg,
                                           offload_video_to_cpu=False,
                                           offload_state_to_cpu=False)),
        ("positional", lambda: base.init_state(inner, jpg, False, False)),
        ("png dir", lambda: base.init_state(inner, frames_dir, False, False)),
    ]
    for lbl, fn in attempts:
        try:
            st = fn()
            print(f"  init_state via base class OK ({lbl})")
            return inner, st
        except BaseException as e:
            print(f"  [{lbl}] {type(e).__name__}: {str(e)[:200]}")
    return None, None


def route_transplant(vp, inner, frames_dir, n):
    """B: Sam3TrackerPredictor has video_path init_state; give it real weights."""
    from sam3.model_builder import build_tracker
    t = build_tracker(False, with_backbone=True)
    src = inner.state_dict()
    landed = False
    for label, target in (("tracker", t), ("tracker.model", getattr(t, "model", None))):
        if target is None:
            continue
        try:
            missing, unexpected = target.load_state_dict(src, strict=False)
            n_tot = len(target.state_dict())
            n_missing = len(missing)
            print(f"  load_state_dict into {label}: {n_tot - n_missing}/{n_tot} "
                  f"keys matched ({n_missing} missing, {len(unexpected)} unexpected)")
            if n_tot and (n_tot - n_missing) > 0.5 * n_tot:
                landed = True
                break
        except BaseException as e:
            print(f"  load into {label}: {type(e).__name__}: {str(e)[:160]}")
    if not landed:
        print("  weights did NOT transplant -- refusing this route: any IoU "
              "from a fresh init would be meaningless")
        return None, None
    import torch
    if torch.cuda.is_available():
        t = t.to("cuda")
    weight_signature(t, "after transplant")
    jpg = jpeg_mirror(frames_dir, n)
    for lbl, kw in (("jpg", {"video_path": jpg}), ("png", {"video_path": frames_dir})):
        try:
            st = t.init_state(**kw)
            print(f"  init_state OK ({lbl})")
            return t, st
        except BaseException as e:
            print(f"  [{lbl}] {type(e).__name__}: {str(e)[:200]}")
    return None, None


def route_session(vp, inner, frames_dir, n):
    """C: the session state. Known to mismatch; tried so the log is complete."""
    from probe_sam3_mask import session_state
    st, _sid = session_state(vp, frames_dir)
    return (inner, st) if st is not None else (None, None)


# ------------------------------------------------------------------- drive
def seed_mask(tracker, state, m0):
    """Offer the frame-0 mask in every shape the two entry points document."""
    import torch
    from contextlib import nullcontext

    def amp():
        return (torch.autocast("cuda", dtype=torch.bfloat16)
                if torch.cuda.is_available() else nullcontext())

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    two_d = [("cpu bool", torch.from_numpy(m0.astype("bool"))),
             ("dev bool", torch.from_numpy(m0.astype("bool")).to(dev)),
             ("dev float", torch.from_numpy(m0.astype("float32")).to(dev))]
    if hasattr(tracker, "add_new_mask"):
        for lbl, mt in two_d:
            try:
                with amp():
                    tracker.add_new_mask(inference_state=state, frame_idx=0,
                                         obj_id=1, mask=mt,
                                         add_mask_to_memory=True)
                print(f"  add_new_mask OK ({lbl})")
                return True
            except BaseException as e:
                print(f"  add_new_mask [{lbl}] {type(e).__name__}: {str(e)[:200]}")
    if hasattr(tracker, "add_new_masks"):
        for lbl, mt in two_d:
            try:
                with amp():
                    tracker.add_new_masks(inference_state=state, frame_idx=0,
                                          obj_ids=[1], masks=mt[None],
                                          add_mask_to_memory=True)
                print(f"  add_new_masks OK ({lbl}, 1xHxW)")
                return True
            except BaseException as e:
                print(f"  add_new_masks [{lbl}] {type(e).__name__}: {str(e)[:200]}")
    return False


def propagate(tracker, state, gts):
    import torch
    from contextlib import nullcontext

    def amp():
        return (torch.autocast("cuda", dtype=torch.bfloat16)
                if torch.cuda.is_available() else nullcontext())

    T, H, W = len(gts), gts[0].shape[0], gts[0].shape[1]
    out = np.zeros((T, H, W), np.uint8)
    try:
        with amp():
            tracker.propagate_in_video_preflight(state)
        print("  preflight OK")
    except BaseException as e:
        print(f"  preflight skipped ({type(e).__name__}: {str(e)[:90]})")

    seen = 0
    with amp():
        for frame_idx, _obj_ids, raw in tracker.propagate_in_video(
                state, start_frame_idx=0, max_frame_num_to_track=None,
                reverse=False):
            if frame_idx >= T:
                continue
            m = _binarize(raw)
            m = _resize_stack(m.reshape(-1, *m.shape[-2:]), H, W)
            out[frame_idx] = np.any(m, axis=0).astype(np.uint8)
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

    banner("BUILD -- predictor with released weights")
    from sam3.model_builder import build_sam3_predictor
    vp = build_sam3_predictor()
    inner = find_inner(vp)
    if inner is None:
        print("no mask-capable object found"); return 1
    trained = weight_signature(inner, "inner tracker")
    dump_source(getattr(inner, "add_new_masks", None) or inner.add_new_mask,
                f"{type(inner).__name__}.add_new_mask(s)")

    v0 = args.videos[0]
    frames_dir = os.path.join(DATA, "images", v0)
    n0 = len(load_frames(v0, args.limit_frames))
    gts0 = load_gt(v0, n0)

    banner("ROUTES -- first one that seeds a mask wins")
    chosen = None
    for name, fn in (("A demo init_state", route_demo),
                     ("B build_tracker + transplanted weights", route_transplant),
                     ("C session state", route_session)):
        print(f"\n--- route {name} ---")
        try:
            tk, st = fn(vp, inner, frames_dir, n0)
        except BaseException as e:
            print(f"  route failed: {type(e).__name__}: {str(e)[:220]}")
            continue
        if tk is None or st is None:
            continue
        if seed_mask(tk, st, gts0[0]):
            chosen = (name, tk, st)
            break
        print("  state built but the mask was not accepted on this route")

    if chosen is None:
        banner("RESULT: no route seeded a mask")
        print("Read the per-route errors above. Only a refusal naming masks or")
        print("prompts is evidence about SAM 3; plumbing errors are ours.")
        return 2

    name, tk, st = chosen
    banner(f"PROPAGATE -- route {name}")
    out, seen = propagate(tk, st, gts0)
    if seen == 0:
        print("mask accepted but propagation returned nothing"); return 3
    ious = [iou(out[t].astype(bool), gts0[t]) for t in range(len(gts0))]
    print(f"\n  frame 0 IoU {ious[0]:.4f}   frame 1 IoU {ious[1]:.4f}"
          if len(ious) > 1 else "")
    print(f"  mean IoU over {len(ious)} frames: {np.nanmean(ious):.4f}")

    if args.out_dir:
        if trained is False:
            print("\nREFUSING to save: the tracker's weights look fresh-init, so "
                  "these masks are not a benchmarkable row.")
            return 4
        import tifffile
        os.makedirs(args.out_dir, exist_ok=True)
        for v in args.videos:
            fd = os.path.join(DATA, "images", v)
            n = len(load_frames(v, args.limit_frames))
            g = load_gt(v, n)
            if v != v0:
                tk2, st2 = dict(A=route_demo, B=route_transplant,
                                C=route_session)[name[0]](vp, inner, fd, n)
                if tk2 is None or not seed_mask(tk2, st2, g[0]):
                    print(f"  {v}: could not seed; skipped"); continue
                stack, _ = propagate(tk2, st2, g)
            else:
                stack = out
            p = os.path.join(args.out_dir, f"{v}.tif")
            tifffile.imwrite(p, stack)
            print(f"  -> {p}  fg={float((stack>0).mean()):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
