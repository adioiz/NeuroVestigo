#!/usr/bin/env python3
"""
drift_quartiles.py -- Table 3 (temporal drift) for every method, computed once,
the same way, from the prediction stacks in benchmark/predictions/.

Each test recording is scored per frame (frame 0 excluded -- it is the prompt),
the per-frame cbDice is split into four contiguous quartiles, each quartile is
averaged within a recording, and recordings are then weighted equally. Decay is
Q4 - Q1: the full span. An earlier draft used Q4 - Q2, which is arbitrary and
hides SAM 2 dropping hardest immediately after its prompt.

    KMP_DUPLICATE_LIB_OK=TRUE python benchmark/drift_quartiles.py
    ... --extra ours_fixed=8-frame        # add a run for the ablation numbers
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, tifffile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from benchmark.data import Dataset
from benchmark.eval_masks import evaluate_stack

# label -> predictions subdirectory. The SAM 3 sweep wrote one directory per
# configuration, so the winning run does not carry the name it is scored under.
OURS_LABEL = "NeuroVestigo"
METHODS = [(OURS_LABEL, "ours_22rec"), ("Cellpose-SAM", "cellpose_sam"),
           ("Omnipose", "omnipose"), ("SAM 2 (zero-shot)", "sam2_zeroshot"),
           ("$\\mu$SAM (box-prompt)", "usam_single"), ("NeuroQuantify", "neuroquantify"),
           ("SAM 3.1 (point-prompt)", "sam3_final")]


def quartiles(pred_dir, videos, ds):
    per_vid = []
    for v in videos:
        p = os.path.join(REPO, "benchmark", "predictions", pred_dir, f"{v}.tif")
        if not os.path.exists(p):
            return None
        r = evaluate_stack(tifffile.imread(p), ds.load_gt(v), name=v,
                           with_cbdice=True, skip_first_frame=True)
        q = np.array_split(np.asarray(r.per_frame["cbdice"]), 4)
        per_vid.append([x.mean() for x in q])
    return np.mean(per_vid, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[],
                    help="label=dir pairs to score alongside the table rows")
    args = ap.parse_args()
    ds = Dataset(); videos = ds.split("test")
    rows = list(METHODS) + [tuple(reversed(e.split("=", 1)))[::-1] for e in args.extra]

    out = []
    for lbl, d in rows:
        m = quartiles(d, videos, ds)
        if m is None:
            print(f"  [skip] {lbl}: no predictions in {d}"); continue
        out.append((lbl, m, m[3] - m[0]))
    out.sort(key=lambda r: -r[2])                    # least decay first

    print(f"\n{'Method':<24}{'Q1':>8}{'Q2':>8}{'Q3':>8}{'Q4':>8}{'Decay':>9}")
    for lbl, m, dec in out:
        print(f"{lbl:<24}" + "".join(f"{x:>8.3f}" for x in m) + f"{dec:>9.3f}")

    print("\n--- LaTeX rows for tab:drift ---")
    for lbl, m, dec in out:
        b = (lambda x: f"\\textbf{{{x:.3f}}}") if lbl == OURS_LABEL else (lambda x: f"{x:.3f}")
        dd = f"$\\mathbf{{{dec:.3f}}}$" if lbl == OURS_LABEL else f"${dec:.3f}$"
        name = f"\\textbf{{{lbl}}}" if lbl == OURS_LABEL else lbl
        print(f"{name:<24} & " + " & ".join(b(x) for x in m) + f" & {dd} \\\\")

    ours = next(r for r in out if r[0] == OURS_LABEL)
    others = [r for r in out if r[0] != OURS_LABEL]
    best = max(others, key=lambda r: r[2])
    print(f"\nours decay        {ours[2]:.3f}")
    print(f"baseline range    {min(abs(r[2]) for r in others):.3f} .. {max(abs(r[2]) for r in others):.3f}")
    print(f"vs best baseline  {abs(best[2]/ours[2]):.2f}x more stable ({best[0]})")
    for r in others:
        print(f"  {r[0]:<24} {abs(r[2]/ours[2]):.2f}x faster")


if __name__ == "__main__":
    main()
