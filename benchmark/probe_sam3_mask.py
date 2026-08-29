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


def jpeg_mirror(png_dir, n):
    """SAM 2-style video loaders glob for .jpg and report "no images found" on a
    directory of .png -- which is what the first corrected run hit. Mirror the
    frames as maximum-quality JPEG in a temp directory so the loader can read
    them. Lossy in principle; irrelevant for answering "does add_new_mask work".
    """
    import tempfile
    files = sorted(glob.glob(os.path.join(png_dir, "*.png")))[:n]
    if not files:
        raise RuntimeError(f"no .png frames in {png_dir}")
    out = tempfile.mkdtemp(prefix="sam3probe_")
    for i, f in enumerate(files):
        Image.open(f).convert("RGB").save(os.path.join(out, f"{i:05d}.jpg"),
                                          quality=100, subsampling=0)
    print(f"    (mirrored {len(files)} frames as JPEG in {out})")
    return out


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
    """Try each plausible construction path; report every failure verbatim.

    Real signature (from stage A of the first run on the cluster):
        build_tracker(apply_temporal_disambiguation: bool,
                      with_backbone: bool = False, compile_mode=None)
    apply_temporal_disambiguation is REQUIRED and positional -- omitting it was
    what made the first probe report "no construction path worked". with_backbone
    defaults to False, which leaves the tracker without an image encoder and
    expecting cached_features, so True is what we want for a standalone run.
    """
    attempts = []
    for k, f in factories.items():
        for atd in (False, True):
            for wb in (True, False):
                attempts.append((f"{k}(apply_temporal_disambiguation={atd}, with_backbone={wb})",
                                 lambda f=f, a=atd, w=wb: f(a, with_backbone=w)))

    def via_video():
        from sam3.model_builder import build_sam3_video_predictor
        vp = build_sam3_video_predictor()
        seen = set()
        def walk(o, path, depth=0):
            if depth > 3 or id(o) in seen:
                return None
            seen.add(id(o))
            for a in dir(o):
                if a.startswith("__"):
                    continue
                try:
                    v = getattr(o, a)
                except BaseException:
                    continue
                if hasattr(v, "add_new_mask") or hasattr(v, "add_new_masks"):
                    print(f"    (found mask-accepting tracker at .{path}{a})")
                    return v
                if hasattr(v, "__dict__") and not callable(v):
                    r = walk(v, f"{path}{a}.", depth + 1)
                    if r is not None:
                        return r
            return None
        t = walk(vp, "")
        if t is None:
            raise RuntimeError(f"no mask-accepting tracker reachable from "
                               f"{type(vp).__name__} within 3 levels")
        return t
    attempts.append(("build_sam3_video_predictor() -> nested tracker", via_video))

    for label, fn in attempts:
        print(f"  trying {label} ...")
        try:
            t = fn()
            print(f"    OK -> {type(t).__name__}")
            return t
        except BaseException as e:
            print(f"    {type(e).__name__}: {str(e)[:160]}")
    return None


def prepare(tracker):
    """Move the tracker to the GPU and report whether its weights look trained.

    build_tracker() returns a model on CPU in float32. SAM 3 feeds CUDA bfloat16,
    so the first mask call dies with "Input type (CUDABFloat16Type) and weight
    type (torch.FloatTensor) should be the same" -- a device/dtype mismatch that
    says nothing about mask support. This was the third false negative.
    """
    import torch
    if not torch.cuda.is_available():
        print("    no CUDA visible -- run this on a GPU node"); return tracker, None
    for label, fn in (("tracker.to(cuda)", lambda: tracker.to("cuda")),
                      ("tracker.model.to(cuda)", lambda: tracker.model.to("cuda")),
                      ("tracker.cuda()", lambda: tracker.cuda())):
        try:
            fn(); print(f"    {label} OK"); break
        except BaseException as e:
            print(f"    {label}: {type(e).__name__}: {str(e)[:110]}")

    # Are these trained weights or a fresh init? build_tracker takes no
    # checkpoint argument, so the answer decides whether any IoU below is
    # meaningful -- the mask question is answerable either way.
    try:
        ps = [p for p in tracker.parameters() if p.dim() > 1]
        if ps:
            import numpy as _np
            stds = [float(p.detach().float().std()) for p in ps[:40]]
            print(f"    weight std over {len(stds)} tensors: "
                  f"median {_np.median(stds):.4f}, range {min(stds):.4f}-{max(stds):.4f}")
            print("    (a fresh init is tightly clustered; trained weights vary widely.")
            print("     If these look untrained, add_new_mask still answers the API")
            print("     question, but any IoU printed below is meaningless.)")
    except BaseException as e:
        print(f"    could not inspect weights: {type(e).__name__}")
    return tracker, "cuda"


# ---------------------------------------------------------------- stages C-E
def drive(tracker, frames_dir, gt):
    """Stages C-E, using the signatures stage A actually reported:

        init_state(video_height=None, video_width=None, num_frames=None,
                   video_path=None, cached_features=None, ...)
        add_new_mask(inference_state, frame_idx, obj_id, mask,
                     add_mask_to_memory=False)
        propagate_in_video(inference_state, start_frame_idx,
                           max_frame_num_to_track, reverse, ...)

    Two details matter. add_mask_to_memory defaults to False, which would seed
    the prompt WITHOUT putting it in the memory bank -- the opposite of our
    protocol, so it is passed True. And propagate_in_video's first four
    arguments are all required, with no defaults.
    """
    H, W = gt.shape[1:]
    T = len(gt)

    print("  init_state ...")
    try:
        jpg_dir = jpeg_mirror(frames_dir, T)
    except BaseException as e:
        print(f"    could not mirror frames: {e}"); jpg_dir = None

    state, loaded_video = None, False
    cands = ([{"video_path": jpg_dir}] if jpg_dir else []) + [
        {"video_path": frames_dir},
        {"video_path": jpg_dir, "video_height": H, "video_width": W, "num_frames": T}
        if jpg_dir else {"video_path": frames_dir},
    ]
    for kw in cands:
        try:
            state = tracker.init_state(**kw)
            loaded_video = True
            print(f"    OK with {sorted(kw)} -- video frames loaded")
            break
        except BaseException as e:
            print(f"    {type(e).__name__}: {str(e)[:170]}")
    if state is None:
        try:
            state = tracker.init_state(video_height=H, video_width=W, num_frames=T)
            print("    OK with dimensions only -- but this is the cached_features")
            print("    path: the state holds NO images, so add_new_mask below will")
            print("    fail on 'images' regardless of whether masks are supported.")
        except BaseException as e:
            print(f"    {type(e).__name__}: {str(e)[:170]}")
    if state is None:
        print("  STOP at C: cannot initialise state at all. Paste this output back.")
        return False

    print("  add_new_mask(frame 0, add_mask_to_memory=True) ...")
    ok = False
    import torch
    from contextlib import nullcontext
    def amp():                      # fresh context each use; reusing one is fragile
        return (torch.autocast("cuda", dtype=torch.bfloat16)
                if torch.cuda.is_available() else nullcontext())
    m0 = gt[0]
    # SAM 2's add_new_mask asserts mask.dim()==2 and does .float().to(device)
    # itself, so a CPU 2-D tensor is the documented shape. SAM 3's refused a bare
    # ndarray ("no attribute 'dim'"), confirming it wants a tensor.
    for cast, lbl in ((lambda m: torch.from_numpy(m.astype("bool")), "cpu bool 2-D"),
                      (lambda m: torch.from_numpy(m.astype("float32")), "cpu float32 2-D"),
                      (lambda m: torch.from_numpy(m.astype("bool")).cuda()
                       if torch.cuda.is_available()
                       else torch.from_numpy(m.astype("bool")), "cuda bool 2-D")):
        try:
            mt = cast(m0)
            with amp():
                tracker.add_new_mask(inference_state=state, frame_idx=0, obj_id=1,
                                     mask=mt, add_mask_to_memory=True)
            print(f"    OK ({lbl})"); ok = True; break
        except BaseException as e:
            print(f"    [{lbl}] {type(e).__name__}: {str(e)[:170]}")
    if not ok and hasattr(tracker, "add_new_masks"):
        print("  add_new_mask failed; trying add_new_masks (plural) ...")
        # Sam3VideoTrackingMultiplexDemo.add_new_masks(inference_state, frame_idx,
        #     obj_ids, masks, add_mask_to_memory=False, reconditioning=False)
        for cast, lbl in ((lambda m: torch.from_numpy(m.astype("bool"))[None], "cpu bool 1xHxW"),
                          (lambda m: torch.from_numpy(m.astype("float32"))[None], "cpu float32 1xHxW")):
            try:
                with amp():
                    tracker.add_new_masks(inference_state=state, frame_idx=0,
                                          obj_ids=[1], masks=cast(m0),
                                          add_mask_to_memory=True)
                print(f"    OK ({lbl}, via add_new_masks)"); ok = True; break
            except BaseException as e:
                print(f"    [{lbl}] {type(e).__name__}: {str(e)[:170]}")

    if not ok:
        if not loaded_video:
            print("  STOP at D, but INCONCLUSIVE: init_state never loaded the video,")
            print("  so the mask was offered to a state with no frames. This says")
            print("  nothing about whether masks are supported -- do NOT quote it.")
        else:
            print("  STOP at D with the video loaded. Check the error above before")
            print("  quoting this: a device/dtype or tensor-shape error is still a")
            print("  probe limitation. Only a refusal that names masks, prompts or")
            print("  unsupported input is evidence about mask support.")
        return False

    print("  propagate_in_video ...")
    try:                      # SAM 2 runs this inside propagate; SAM 3 exposes it
        with amp():
            tracker.propagate_in_video_preflight(state)
        print("    preflight OK")
    except BaseException as e:
        print(f"    preflight skipped ({type(e).__name__}: {str(e)[:90]})")

    for mx, lbl in ((T, f"max_frame_num_to_track={T}"), (None, "max_frame_num_to_track=None")):
        try:
            n = 0
            with amp():
              for frame_idx, obj_ids, logits in tracker.propagate_in_video(
                    state, start_frame_idx=0, max_frame_num_to_track=mx, reverse=False):
                  m = (logits.detach().float().cpu().numpy() if hasattr(logits, "detach")
                       else np.asarray(logits, dtype="float32"))
                  m = (m > 0).reshape(-1, *m.shape[-2:])
                  a = np.any(m, 0)
                  iou = None
                  if frame_idx < T and a.shape == gt[frame_idx].shape:
                      u = (a | gt[frame_idx]).sum()
                      iou = (a & gt[frame_idx]).sum() / u if u else float("nan")
                  print(f"    frame {frame_idx:>3}  objs={len(obj_ids)}  fg={float(a.mean()):.4f}"
                        + (f"  IoU={iou:.3f}" if iou is not None else ""))
                  n += 1
            print(f"  OK -- {n} frames propagated from a MASK prompt ({lbl})")
            return True
        except BaseException as e:
            print(f"    [{lbl}] {type(e).__name__}: {str(e)[:200]}")
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

    show("STAGE B2 -- move to device, and are the weights trained?")
    tracker, _dev = prepare(tracker)

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
        print("Stopped before propagating from a mask. Read the stage that stopped:")
        print("only a stop AFTER the video loaded is evidence about mask support.")
        print("A loader or signature failure is a probe limitation, not a finding,")
        print("and must not be quoted in the paper. Paste this output back.")


if __name__ == "__main__":
    main()
