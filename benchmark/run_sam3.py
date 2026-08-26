#!/usr/bin/env python3
"""
run_sam3.py -- SAM 3 competitor runner, used in its most favourable mode.

WHY THE PROTOCOL DIFFERS FROM SAM 2
-----------------------------------
SAM 3's video predictor accepts text, points and bounding boxes -- but NOT a
mask (see sam3/model/sam3_base_predictor.py: add_prompt takes `text`,
`points`/`point_labels`, `bounding_boxes`/`bounding_box_labels`). It therefore
cannot be given the frame-0 mask that our model and zero-shot SAM 2 receive.
That is a property of the model, not a handicap we impose.

Its closest available analogue is the box prompt, which is exactly what micro-SAM
is given. We therefore sweep the prompt regimes below on the VALIDATION split
only, adopt the best, and apply it unchanged to test -- the same rule every other
competitor in this benchmark was tuned under.

    box-every   GT box on every frame            (matches uSAM's supervision)
    box-first   GT box on frame 0, then track    (matches OUR supervision budget)
    points-N    N points sampled inside the GT   (frame 0 or every frame)
    text        concept prompt, e.g. "neuron"    (automatic; union-matched to target)

RUNNING IT
----------
SAM 3 needs CUDA 12.6+, Python 3.12+, PyTorch 2.7+ -- it does not run on MPS.
Run this on the cluster, not on a laptop.

    python benchmark/run_sam3.py --probe                     # dump the real API first
    python benchmark/run_sam3.py --mode box-every --videos ...

Predicted target stacks -> benchmark/predictions/sam3_<mode>/<name>.tif
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


# --------------------------------------------------------------------------- io
def load_frames(name, limit=None):
    files = sorted(glob.glob(os.path.join(DATA, "images", name, "*.png")))
    if limit:
        files = files[:limit]
    return [np.array(Image.open(f).convert("L")) for f in files]


def load_gt(name, n):
    files = sorted(glob.glob(os.path.join(DATA, "masks", name, "*.png")))[:n]
    return [np.array(Image.open(f)) > 0 for f in files]


def gt_box_xywh(mask, pad=6):
    """SAM 3 takes boxes as xywh (see boxes_xywh in the base predictor)."""
    ys, xs = np.where(mask)
    H, W = mask.shape
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1, y1 = min(W - 1, xs.max() + pad), min(H - 1, ys.max() + pad)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def gt_points(mask, n=3, rng=None):
    """n positive points, spread along the target's skeleton-ish extent."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return [], []
    idx = np.linspace(0, len(ys) - 1, num=min(n, len(ys))).astype(int)
    order = np.argsort(xs)                      # spread across the arbour
    idx = order[idx]
    return [[float(xs[i]), float(ys[i])] for i in idx], [1] * len(idx)


def match_to_target(inst_mask, gt):
    """Union of predicted instances overlapping the target -- as for the
    other non-target-aware methods (see run_cellpose.match_to_target)."""
    out = np.zeros(gt.shape, bool)
    if not gt.any():
        return out
    for lab in np.unique(inst_mask):
        if lab == 0:
            continue
        m = inst_mask == lab
        if np.logical_and(m, gt).any():
            out |= m
    return out


# ------------------------------------------------------------------- api probe
def probe():
    """Print the ACTUAL SAM 3 API on this machine.

    Meta documents only the text-prompt path for video, and the module layout
    differs from the public README (the real entry points live in
    `sam3.model_builder`, not `sam3.model.build_sam`). So discover rather than
    guess: this walks the installed package, finds every build_* factory and
    every predictor class, and prints the request types and prompt keys the
    build really accepts.
    """
    import importlib
    import inspect
    import pkgutil
    import re

    import sam3
    root = os.path.dirname(sam3.__file__)
    print("sam3 from :", root)
    print("version   :", getattr(sam3, "__version__", "?"))

    # -- every build_* factory, wherever it lives
    print("\n--- factories ---")
    mods = [sam3]
    for m in pkgutil.iter_modules([root]):
        if m.name.startswith("_"):
            continue
        try:                       # a sibling module may assert/raise on import
            mods.append(importlib.import_module(f"sam3.{m.name}"))
        except BaseException as e:
            print(f"  [skip] sam3.{m.name}: {type(e).__name__}: {e}")
    for mod in mods:
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            if name.startswith("build_"):
                try:
                    sig = inspect.signature(obj)
                except (ValueError, TypeError):
                    sig = "(?)"
                print(f"  {mod.__name__}.{name}{sig}")

    # -- predictor classes and their prompt/propagate methods
    print("\n--- predictor classes ---")
    for m in pkgutil.walk_packages([root], prefix="sam3."):
        if not any(k in m.name for k in ("predictor", "build", "video")):
            continue
        try:
            mod = importlib.import_module(m.name)
        except BaseException as e:
            print(f"  [skip] {m.name}: {type(e).__name__}: {e}")
            continue
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if obj.__module__ != m.name:
                continue
            meths = [n for n, _ in inspect.getmembers(obj, inspect.isfunction)
                     if any(k in n for k in ("prompt", "propagate", "session",
                                             "request", "track", "add_"))]
            if meths:
                print(f"  {m.name}.{name}: {meths}")

    # -- what request dicts actually carry
    print("\n--- request types / prompt keys (scanned from source) ---")
    types, keys = set(), set()
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.endswith(".py"):
                continue
            try:
                src = open(os.path.join(dirpath, f), encoding="utf-8").read()
            except Exception:
                continue
            types |= set(re.findall(r"type\s*==\s*[\"\']([a-z_]+)[\"\']", src))
            keys |= set(re.findall(r"request\.get\(\s*[\"\']([a-z_]+)[\"\']", src))
            keys |= set(re.findall(r"request\[[\"\']([a-z_]+)[\"\']\]", src))
    print("  request types:", sorted(types))
    print("  prompt keys  :", sorted(keys))


# ----------------------------------------------------------------------- driver
def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x)


def _binarize(raw):
    """SAM 3 returns binary masks, but be defensive about what a given build
    hands back: `> 0` on a [0,1] probability map marks EVERY pixel foreground,
    which produces a full-frame prediction and a plausible-looking score rather
    than an error."""
    m = _to_numpy(raw)
    if m.dtype == bool:
        return m
    if np.issubdtype(m.dtype, np.floating):
        finite = m[np.isfinite(m)]
        if (finite.size and finite.min() >= 0.0 and finite.max() <= 1.0
                and not np.isin(finite, (0.0, 1.0)).all()):
            return m >= 0.5                      # probabilities
        return m > 0.0                           # logits
    return m > 0


def _resize_stack(masks, H, W):
    """Model-resolution masks -> label resolution, nearest neighbour."""
    if masks.shape[-2:] == (H, W):
        return masks
    out = np.zeros((masks.shape[0], H, W), bool)
    for i, m in enumerate(masks):
        img = Image.fromarray(m.astype(np.uint8) * 255)
        out[i] = np.array(img.resize((W, H), Image.NEAREST)) > 127
    return out


def run_video(predictor, frames_dir, gts, mode, text, npoints):
    """One recording. Returns a [T,H,W] uint8 stack.

    Request names and keys are the ones `--probe` and the build_sam3_predictor
    docstring confirm on this build:

        handle_request({"type": "start_session", "resource_path": <dir>})
        handle_request({"type": "add_prompt", "session_id":.., "frame_index":.., ..})
        handle_stream_request({"type": "propagate_in_video", ..})   -> generator

    Two defaults in Sam3BasePredictor bite if left alone:
    `add_prompt(rel_coordinates=True)` would read our pixel boxes as normalised
    coordinates, and `propagate_in_video(propagation_direction="both")` walks
    the clip forwards and then backwards, writing every frame twice.
    """
    H, W = gts[0].shape
    T = len(gts)

    resp = predictor.handle_request(
        {"type": "start_session", "resource_path": frames_dir})
    sid = resp["session_id"]

    try:
        prompt_frames = [0] if mode.endswith("first") or mode == "text" else range(T)
        n_prompted = 0
        for t in prompt_frames:
            g = gts[t]
            if not g.any():
                continue
            req = {"type": "add_prompt", "session_id": sid, "frame_index": int(t),
                   "rel_coordinates": False}     # ours are pixels, not normalised
            if mode == "text":
                req["text"] = text
            elif mode.startswith("box"):
                req["bounding_boxes"] = [gt_box_xywh(g)]
                req["bounding_box_labels"] = [1]
                req["obj_id"] = 1                # every box refines ONE track
            elif mode.startswith("points"):
                pts, labs = gt_points(g, npoints)
                if not pts:
                    continue
                req["points"] = pts
                req["point_labels"] = labs
                req["obj_id"] = 1
            predictor.handle_request(req)
            n_prompted += 1
        if n_prompted == 0:
            raise RuntimeError(f"no prompt accepted for {frames_dir}")

        out = np.zeros((T, H, W), np.uint8)
        stream = predictor.handle_stream_request({
            "type": "propagate_in_video",
            "session_id": sid,
            "propagation_direction": "forward",
            "start_frame_index": 0,
        })

        seen = 0
        for frame_idx, _obj_ids, raw in _iter_results(stream):
            if frame_idx >= T:
                continue
            masks = _binarize(raw)
            masks = masks.reshape(-1, *masks.shape[-2:])
            masks = _resize_stack(masks, H, W)
            seen += 1
            if mode == "text":                   # not target-aware -> union match
                inst = np.zeros((H, W), np.int32)
                for i, m in enumerate(masks, 1):
                    inst[m] = i
                out[frame_idx] = match_to_target(inst, gts[frame_idx]).astype(np.uint8)
            else:
                out[frame_idx] = np.any(masks, axis=0).astype(np.uint8)

        if seen == 0:
            raise RuntimeError(
                f"propagation returned no frames for {frames_dir}: the session "
                "most likely loaded zero frames. Check that SAM 3's video "
                "loader accepts a directory of .png on this build.")
    finally:
        try:
            predictor.handle_request({"type": "close_session", "session_id": sid})
        except Exception:
            pass
    return out


_MASK_KEYS = ("out_binary_masks", "binary_masks", "pred_masks", "masks")


def _iter_results(stream):
    """handle_stream_request yields {"frame_index":.., "outputs": {..}}; the
    docstring example reads the mask straight off the item, so some builds
    flatten it. Accept either, and fail loudly rather than silently empty."""
    for item in stream:
        if not isinstance(item, dict):
            raise RuntimeError(f"unexpected propagate item: {type(item)}")
        fi = item.get("frame_index", item.get("frame_idx"))
        payload = item.get("outputs")
        masks = obj_ids = None
        for c in (payload, item):
            if not isinstance(c, dict):
                continue
            for k in _MASK_KEYS:
                if k in c:
                    masks, obj_ids = c[k], c.get("obj_ids")
                    break
            if masks is not None:
                break
        if fi is None or masks is None:
            raise RuntimeError(
                f"unexpected propagate payload; item keys={sorted(item)} "
                f"outputs keys="
                f"{sorted(payload) if isinstance(payload, dict) else type(payload)}\n"
                "Run `python benchmark/run_sam3.py --probe` and adjust _MASK_KEYS.")
        yield int(fi), obj_ids, masks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true",
                    help="print the installed SAM 3 API and exit (run this first)")
    ap.add_argument("--videos", nargs="+")
    ap.add_argument("--mode", default="box-every",
                    choices=["box-every", "box-first", "points-every",
                             "points-first", "text"])
    ap.add_argument("--text", default="neuron",
                    help="concept prompt for --mode text")
    ap.add_argument("--npoints", type=int, default=3)
    ap.add_argument("--checkpoint", default=None,
                    help="path or HF id; defaults to the repo's own default")
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--version", default="sam3.1", choices=["sam3.1", "sam3"],
                    help="SAM 3 weights to benchmark (3.1 is the stronger default)")
    ap.add_argument("--use-fa3", action="store_true",
                    help="enable Flash Attention 3; off by default because "
                         "flash_attn_interface is not installed in sam3-env")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.probe:
        probe()
        return
    if not args.videos:
        raise SystemExit("--videos is required unless --probe")

    import torch
    if not torch.cuda.is_available():
        print("[warn] SAM 3 requires CUDA 12.6+; no CUDA device visible. "
              "This will very likely fail -- run it on the cluster.")
    # build_sam3_video_predictor returns Sam3VideoPredictorMultiGPU (it wants an
    # NCCL group); build_sam3_predictor is the documented single-process entry.
    from sam3.model_builder import build_sam3_predictor
    kw = {"version": args.version, "use_fa3": args.use_fa3}
    if args.checkpoint:
        kw["checkpoint_path"] = args.checkpoint
    predictor = build_sam3_predictor(**kw)

    out_dir = args.out_dir or os.path.join(
        REPO, "benchmark", "predictions", f"sam3_{args.mode.replace('-', '')}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"SAM 3  version={args.version}  mode={args.mode}  out={out_dir}")

    for v in args.videos:
        frames_dir = os.path.join(DATA, "images", v)
        n = len(load_frames(v, args.limit_frames))
        gts = load_gt(v, n)
        print(f"  {v}: {n} frames ...", flush=True)
        stack = run_video(predictor, frames_dir, gts, args.mode,
                          args.text, args.npoints)
        p = os.path.join(out_dir, f"{v}.tif")
        tifffile.imwrite(p, stack)
        print(f"    -> {p}  (fg={float((stack>0).mean()):.4f})")


if __name__ == "__main__":
    main()
