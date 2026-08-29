#!/usr/bin/env python3
"""
audit_predictions.py -- adversarial check on every prediction stack behind the
paper's tables. Run before submitting, and after any re-run.

Aggregate scores hide the failures that matter most, so this checks the ones a
mean cannot show:

  1. shape / frame count / dtype / binarity against the ground truth
  2. temporal alignment -- score at frame offsets -1, 0, +1. If an offset beats
     0, predictions are shifted in time and every number is wrong.
  3. orientation -- score as-is against fliplr / flipud / transpose. A flipped
     mask scores plausibly-low rather than zero, so it survives an eyeball.
  4. frame-0 seeding -- mask-prompted methods must receive an identical seed,
     or the controlled comparison is not controlled.
  5. end-to-end -- recompute every table value from the stacks and compare to
     summary.csv AND to main.tex, with half-up rounding at 3 dp.

    KMP_DUPLICATE_LIB_OK=TRUE python benchmark/audit_predictions.py
"""
from __future__ import annotations
import csv, os, re, sys
from decimal import Decimal, ROUND_HALF_UP
import numpy as np, tifffile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
from benchmark.data import Dataset
from benchmark.eval_masks import evaluate_stack

MODELS = {"ours": "ours", "sam2_zeroshot": "sam2_zeroshot", "usam": "usam",
          "sam3.1": "sam3_final", "cellpose_sam": "cellpose_sam",
          "neuroquantify": "neuroquantify", "omnipose": "omnipose",
          "sam3.1_text": "sam3_text_test"}
TEX = {"Ours": "ours", "SAM\\,2 (zero-shot)": "sam2_zeroshot",
       "$\\mu$SAM (box-prompt)": "usam", "SAM\\,3.1 (point-prompt)": "sam3.1",
       "Cellpose-SAM": "cellpose_sam", "NeuroQuantify": "neuroquantify",
       "Omnipose": "omnipose", "SAM\\,3.1 (text-prompt)": "sam3.1_text"}
COLS = ["iou_mean", "dice_mean", "cbdice_mean", "cldice_mean"]


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else np.nan


def main():
    ds = Dataset(); vids = ds.split("test")
    _gt = {v: ds.load_gt(v) for v in vids}
    _pred = {}
    def load(d, v):                      # stacks are re-read many times below
        if (d, v) not in _pred:
            _pred[(d, v)] = tifffile.imread(f"benchmark/predictions/{d}/{v}.tif")
        return _pred[(d, v)]
    def gt(v):
        return _gt[v]
    fail = []

    print("=== 1. shape / dtype / binarity ===")
    for m, d in MODELS.items():
        for v in vids:
            a, g = load(d, v), gt(v)
            if a.shape != g.shape: fail.append(f"shape {m}/{v}: {a.shape} vs {g.shape}")
            if not set(np.unique(a).tolist()) <= {0, 1}: fail.append(f"non-binary {m}/{v}")
    print("   ok" if not fail else "\n".join("   " + f for f in fail))

    print("\n=== 2. temporal alignment (offset -1 / 0 / +1) ===")
    for m, d in MODELS.items():
        sc = {}
        for off in (-1, 0, 1):
            vals = [iou(load(d, v)[t + off] > 0, gt(v)[t] > 0)
                    for v in vids for t in range(1, len(gt(v)))
                    if 0 <= t + off < len(gt(v))]
            sc[off] = np.nanmean(vals)
        best = max(sc, key=lambda k: sc[k])
        if best != 0: fail.append(f"TIME-SHIFTED {m}: offset {best:+d} scores higher")
        print(f"   {m:<14}" + "".join(f"{sc[o]:>9.4f}" for o in (-1, 0, 1))
              + ("   ok" if best == 0 else f"   SHIFTED {best:+d}"))

    print("\n=== 3. orientation ===")
    for m, d in MODELS.items():
        r = {}
        for nm, fn in [("as-is", lambda x: x), ("fliplr", lambda x: x[:, ::-1]),
                       ("flipud", lambda x: x[::-1, :]), ("transp", lambda x: x.T)]:
            vals = [iou(fn(load(d, v)[t] > 0), gt(v)[t] > 0)
                    for v in vids for t in range(1, len(gt(v)))
                    if not (nm == "transp" and load(d, v).shape[1] != load(d, v).shape[2])]
            r[nm] = np.nanmean(vals) if vals else np.nan
        best = max((k for k in r if not np.isnan(r[k])), key=lambda k: r[k])
        if best != "as-is": fail.append(f"ORIENTATION {m}: {best} scores higher")
        print(f"   {m:<14}" + "".join(f"{r[k]:>9.4f}" for k in r) +
              ("   ok" if best == "as-is" else f"   WRONG: {best}"))

    print("\n=== 4. frame-0 seed identical across mask-prompted methods ===")
    for v in vids:
        a = load(MODELS["ours"], v)[0] > 0
        b = load(MODELS["sam2_zeroshot"], v)[0] > 0
        same = bool((a == b).all())
        if not same: fail.append(f"SEED DIFFERS at frame 0 for {v}")
        print(f"   {v:<16} ours == sam2 seed: {same}   (vs GT: "
              f"{iou(a, gt(v)[0] > 0):.3f} -- <1 is the 1024px round trip)")

    print("\n=== 5. end-to-end: stacks -> summary.csv -> main.tex ===")
    S = {r["model"]: r for r in csv.DictReader(open("benchmark/results/test/summary.csv"))}
    tex = open("paper_assets/paper/main.tex").read()
    i = tex.index("\\label{tab:main}")
    blk = tex[tex.rindex("\\begin{table}", 0, i):tex.index("\\end{table}", i)]
    n_frames = 0
    for m, d in MODELS.items():
        rows = [evaluate_stack(load(d, v), gt(v), name=v,
                               with_cbdice=True, skip_first_frame=True) for v in vids]
        if m == "ours": n_frames = sum(len(r.per_frame["iou"]) for r in rows)
        for k, col in zip(("iou", "dice", "cbdice", "cldice"), COLS):
            got = float(np.mean([r.mean(k) for r in rows]))
            if abs(got - float(S[m][col])) > 5e-5:
                fail.append(f"CSV {m}/{col}: stacks {got:.4f} vs csv {S[m][col]}")
    cells = 0
    for line in blk.split("\n"):
        if not line.strip().endswith("\\\\") or "multicolumn" in line or "Method &" in line:
            continue
        nm = line.split("&")[0].replace("\\textbf{", "").replace("}", "").strip()
        if nm not in TEX: continue
        for c, col in zip([re.sub(r"[^0-9.]", "", x) for x in line.split("&")[1:5]], COLS):
            cells += 1
            exp = Decimal(S[TEX[nm]][col]).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            if exp != Decimal(c): fail.append(f"TEX {nm}/{col}: {c} should be {exp}")
    print(f"   {cells} table cells checked against the scored CSV")
    print(f"   scored frames (frame 0 excluded): {n_frames}"
          + ("  ok" if n_frames == 328 else "  EXPECTED 328"))
    if n_frames != 328: fail.append(f"frame count {n_frames}, expected 328")

    print("\n" + "=" * 60)
    print("AUDIT PASSED" if not fail else f"AUDIT FAILED -- {len(fail)} problem(s):")
    for f in fail: print("  " + f)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
