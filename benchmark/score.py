#!/usr/bin/env python3
"""
score.py -- score a model's predicted mask stacks against ground truth and
record the result in benchmark/results/. One entry point for EVERY model
(ours, sam2_zeroshot, cellpose_sam, usam, cellsam, deepsea) so the comparison
table and figures all read from the same place.

    python benchmark/score.py --pred-dir benchmark/predictions/ours --model ours --cbdice

Writes:
    benchmark/results/<model>.csv          per-sequence IoU/Dice/clDice(/cbDice)
    benchmark/results/summary.csv          one appended/updated row per model
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import tifffile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from benchmark.data import Dataset
from benchmark.eval_masks import evaluate_stack, aggregate

METRICS = ["iou", "dice", "cldice"]


def score_model(pred_dir: str, model: str, split: str = "test", videos=None,
                with_cbdice=False, skip_first_frame=False):
    ds = Dataset()
    videos = videos or ds.split(split)
    RESULTS_DIR = os.path.join(REPO, "benchmark", "results", split)
    metrics = METRICS + (["cbdice"] if with_cbdice else [])
    results, missing = [], []
    for v in videos:
        p = os.path.join(pred_dir, f"{v}.tif")
        if not os.path.exists(p):
            missing.append(v)
            continue
        gt = ds.load_gt(v)
        pred = tifffile.imread(p)
        results.append(evaluate_stack(pred, gt, name=v, with_cbdice=with_cbdice,
                                      skip_first_frame=skip_first_frame))
    if missing:
        print(f"  [warn] {model}: no prediction for {missing}")
    if not results:
        raise SystemExit(f"no predictions found in {pred_dir}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    per_seq = os.path.join(RESULTS_DIR, f"{model}.csv")
    with open(per_seq, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence"] + [f"{m}_mean" for m in metrics] + ["n_frames"])
        for r in results:
            w.writerow([r.name] + [f"{r.mean(m):.4f}" for m in metrics]
                       + [len(r.per_frame["iou"])])

    agg = aggregate(results)
    # print
    hdr = {"iou": "IoU", "dice": "Dice", "cldice": "clDice", "cbdice": "cbDice"}
    line = f"{'sequence':<14}" + "".join(f"{hdr[m]:>9}" for m in metrics)
    print("\n" + line + "\n" + "-" * len(line))
    for r in results:
        print(f"{r.name:<14}" + "".join(f"{r.mean(m):>9.4f}" for m in metrics))
    print("-" * len(line))
    print(f"{'MEAN':<14}" + "".join(f"{agg[m]['mean']:>9.4f}" for m in metrics))
    print(f"{'STD':<14}" + "".join(f"{agg[m]['std']:>9.4f}" for m in metrics)
          + f"   (n={agg[metrics[0]]['n']})")

    # upsert into summary.csv
    summary = os.path.join(RESULTS_DIR, "summary.csv")
    cols = ["model"] + [f"{m}_mean" for m in metrics] + [f"{m}_std" for m in metrics] + ["n_seq"]
    rows = {}
    if os.path.exists(summary):
        with open(summary) as f:
            for row in csv.DictReader(f):
                rows[row["model"]] = row
    rows[model] = {"model": model, "n_seq": agg[metrics[0]]["n"],
                   **{f"{m}_mean": f"{agg[m]['mean']:.4f}" for m in metrics},
                   **{f"{m}_std": f"{agg[m]['std']:.4f}" for m in metrics}}
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for m in rows:
            w.writerow({c: rows[m].get(c, "") for c in cols})
    print(f"\nwrote {per_seq} and updated {summary}")
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--model", required=True, help="model label (row key in results)")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"],
                    help="which held-out split these predictions cover (default: test)")
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--cbdice", action="store_true")
    ap.add_argument("--skip-first-frame", action="store_true")
    args = ap.parse_args()
    score_model(args.pred_dir, args.model, split=args.split, videos=args.videos,
                with_cbdice=args.cbdice, skip_first_frame=args.skip_first_frame)


if __name__ == "__main__":
    main()
