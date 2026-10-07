#!/usr/bin/env python3
"""
aggregate_folds.py -- Table 1 and Table 2 over the cross-validation folds
(src/folds.py), as mean +- SD across folds.

Every number comes from the prediction stacks and is computed the way the
single-split tables were: frame 0 excluded (it is the prompt), metrics per frame
at native resolution, averaged within a recording, recordings weighted equally.
A fold's score is that average over its four test recordings. The table reports
the mean of the fold scores +- their SD (n = 3 models, ddof = 1); with four
recordings per fold, the mean over folds is also the mean over all twelve.

Table 2 (temporal drift) splits each recording's per-frame cbDice into four
contiguous quartiles, as drift_quartiles.py does; decay is Q4 - Q1 per fold.

    venv/bin/python benchmark/aggregate_folds.py            # folds 1-3
    venv/bin/python benchmark/aggregate_folds.py --folds 1  # the paper's tables

A method is tabulated only when it has a prediction for every test recording of
every requested fold; anything missing is listed, by recording, at the end.

Writes benchmark/results/folds/{table1,table2,per_recording}.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import tifffile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from benchmark.data import Dataset
from benchmark.eval_masks import evaluate_stack
from src.folds import FOLDS, test_videos

PRED = os.path.join(REPO, "benchmark", "predictions")
OUT = os.path.join(REPO, "benchmark", "results", "folds")

# label -> predictions subdirectory, or {fold: subdirectory} when every fold has
# its own model. Baselines are not trained on this data, so one directory holds
# them for all folds. Paper order.
OURS = {1: "ours_23rec_topo", 2: "ours_fold2_topo", 3: "ours_fold3_topo"}
METHODS = [
    ("NeuroVestigo", OURS),
    ("SAM 2 (zero-shot)", "sam2_zeroshot"),
    ("SAM 3 (mask)", "sam3_maskfirst"),
    ("muSAM (box)", "usam_single"),
    ("Cellpose-SAM", "cellpose_sam"),
    ("NeuroQuantify", "neuroquantify"),
    ("Omnipose", "omnipose"),
    ("SAM 3.1 (text)", "sam3_text_final"),
    # Not in Table 1: the IoU-best checkpoint, for the checkpoint-selection note.
    ("NeuroVestigo (IoU-best ckpt)", {1: "ours_23rec", 2: "ours_fold2", 3: "ours_fold3"}),
]
TABLE2_SKIP = {"SAM 3.1 (text)"}            # not in the paper's Table 2 either
METRICS = ["iou", "dice", "cbdice", "cldice"]


def pred_dir(spec, fold: int) -> str:
    return spec[fold] if isinstance(spec, dict) else spec


def provenance_problem(directory: str, fold: int) -> str | None:
    """score_fold.sbatch drops the model's training_config.json next to its
    predictions; if it is there, the model must have held out this fold."""
    p = os.path.join(PRED, directory, "training_config.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        held_out = json.load(f)["data"]["test_videos"]
    if sorted(held_out) != sorted(test_videos(fold)):
        return (f"{directory} comes from a model that held out {held_out}, "
                f"not fold {fold}'s {test_videos(fold)}")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", nargs="+", type=int, default=sorted(FOLDS),
                    choices=sorted(FOLDS))
    args = ap.parse_args()
    folds = args.folds

    gt_cache: dict[str, np.ndarray] = {}
    per_rec = []                # (label, fold, recording, {metric: mean}, quartiles)
    table1, table2, missing = [], [], []
    for label, spec in METHODS:
        fold_scores, fold_quarts, absent = {}, {}, []
        for fold in folds:
            d = pred_dir(spec, fold)
            bad = provenance_problem(d, fold)
            if bad:
                raise SystemExit(f"refusing to score: {bad}")
            ds = Dataset(fold=fold)
            vids = ds.split("test")
            paths = {v: os.path.join(PRED, d, f"{v}.tif") for v in vids}
            gone = [f"{d}/{v}" for v, p in paths.items() if not os.path.exists(p)]
            if gone:
                absent += gone
                continue
            rec_means, rec_quarts = [], []
            for v in vids:
                if v not in gt_cache:
                    gt_cache[v] = ds.load_gt(v)
                r = evaluate_stack(tifffile.imread(paths[v]), gt_cache[v], name=v,
                                   with_cbdice=True, skip_first_frame=True)
                m = {k: r.mean(k) for k in METRICS}
                q = [x.mean() for x in np.array_split(np.asarray(r.per_frame["cbdice"]), 4)]
                rec_means.append(m)
                rec_quarts.append(q)
                per_rec.append((label, fold, v, m, q))
            fold_scores[fold] = {k: float(np.mean([m[k] for m in rec_means])) for k in METRICS}
            fold_quarts[fold] = np.mean(rec_quarts, axis=0)
        if absent:
            missing.append((label, absent))
            continue
        n = len(folds)
        sd = (lambda xs: float(np.std(xs, ddof=1)) if n > 1 else float("nan"))
        row = {"method": label, "n_folds": n}
        for k in METRICS:
            xs = [fold_scores[f][k] for f in folds]
            row[f"{k}_mean"], row[f"{k}_sd"] = float(np.mean(xs)), sd(xs)
            for f in folds:
                row[f"{k}_fold{f}"] = fold_scores[f][k]
        table1.append(row)
        if label not in TABLE2_SKIP:
            qs = np.array([fold_quarts[f] for f in folds])         # folds x 4
            decays = qs[:, 3] - qs[:, 0]
            table2.append({"method": label, "n_folds": n,
                           **{f"q{i + 1}": float(qs[:, i].mean()) for i in range(4)},
                           "decay_mean": float(decays.mean()), "decay_sd": sd(decays),
                           **{f"decay_fold{f}": float(dv) for f, dv in zip(folds, decays)}})

    os.makedirs(OUT, exist_ok=True)
    pm = (lambda r, k: f"{r[f'{k}_mean']:.3f}" + (f" +- {r[f'{k}_sd']:.3f}" if len(folds) > 1 else ""))
    print(f"\nTable 1 -- folds {folds}, {len(folds) * 4} test recordings, frame 0 excluded")
    print(f"{'method':<30}" + "".join(f"{k:>18}" for k in ("IoU", "Dice", "cbDice", "clDice")))
    for r in table1:
        print(f"{r['method']:<30}" + "".join(f"{pm(r, k):>18}" for k in METRICS))
    if len(folds) > 1:
        print("\nper-fold IoU / cbDice:")
        for r in table1:
            print(f"  {r['method']:<30}" + "  ".join(
                f"f{f} {r[f'iou_fold{f}']:.3f}/{r[f'cbdice_fold{f}']:.3f}" for f in folds))

    print(f"\nTable 2 -- mean cbDice by recording quartile, decay = Q4 - Q1")
    print(f"{'method':<30}{'Q1':>8}{'Q2':>8}{'Q3':>8}{'Q4':>8}{'decay':>20}")
    for r in sorted(table2, key=lambda r: -r["decay_mean"]):
        dec = f"{r['decay_mean']:+.3f}" + (f" +- {r['decay_sd']:.3f}" if len(folds) > 1 else "")
        print(f"{r['method']:<30}" + "".join(f"{r[f'q{i}']:>8.3f}" for i in range(1, 5))
              + f"{dec:>20}")

    for name, rows in (("table1", table1), ("table2", table2)):
        if rows:
            with open(os.path.join(OUT, f"{name}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    with open(os.path.join(OUT, "per_recording.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "fold", "recording"] + METRICS + ["q1", "q2", "q3", "q4"])
        for label, fold, v, m, q in per_rec:
            w.writerow([label, fold, v] + [f"{m[k]:.4f}" for k in METRICS]
                       + [f"{x:.4f}" for x in q])
    print(f"\nwrote {os.path.relpath(OUT, REPO)}/table1.csv, table2.csv, per_recording.csv")

    if missing:
        print("\nNOT TABULATED -- predictions missing:")
        for label, gone in missing:
            print(f"  {label}: {' '.join(gone)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
