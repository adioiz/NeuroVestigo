#!/usr/bin/env python3
"""
plot_comparison.py -- publication-grade quantitative comparison figure.

Reads benchmark/results/summary.csv (one row per model, written by score.py) and
renders grouped bars of IoU / Dice / clDice with std error bars, ours emphasized.
Auto-includes whatever models are present, so it grows as competitors are added.

    python benchmark/plot_comparison.py

Output -> paper_assets/figures/fig3_benchmark_bars.{pdf,png}
"""
from __future__ import annotations

import csv
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import argparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(REPO, "paper_assets", "figures")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 120,
})
C = {"navy": "#1F3A5F", "blue": "#3B6FB6", "teal": "#3AA7A0", "amber": "#E1A93B",
     "rust": "#C6592F", "grey": "#9AA3AF", "lgrey": "#D5DAE1"}

# display names + draw order (ours last so it's rightmost / emphasized)
PRETTY = {"sam2_zeroshot": "SAM 2\n(zero-shot)", "cellpose_sam": "Cellpose-\nSAM",
          "usam": "µSAM", "cellsam": "CellSAM", "deepsea": "DeepSea", "ours": "Ours"}
ORDER = ["sam2_zeroshot", "usam", "cellpose_sam", "cellsam", "deepsea", "ours"]
METRICS = [("iou", "IoU", C["blue"]), ("dice", "Dice", C["teal"]),
           ("cldice", "clDice", C["navy"])]


def load_summary(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["model"]] = r
    return rows


SPLIT_LABEL = {"test": "held-out TEST set", "val": "validation set", "train": "training set"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    args = ap.parse_args()
    summary = os.path.join(REPO, "benchmark", "results", args.split, "summary.csv")
    rows = load_summary(summary)
    models = [m for m in ORDER if m in rows] + [m for m in rows if m not in ORDER]
    n_models = len(models)
    x = np.arange(n_models)
    width = 0.26

    fig, ax = plt.subplots(figsize=(1.6 + 1.15 * n_models, 3.8))
    for i, (key, label, col) in enumerate(METRICS):
        means = [float(rows[m][f"{key}_mean"]) for m in models]
        stds = [float(rows[m].get(f"{key}_std", 0) or 0) for m in models]
        off = (i - 1) * width
        bars = ax.bar(x + off, means, width, yerr=stds, label=label, color=col,
                      edgecolor="white", linewidth=0.5,
                      error_kw=dict(ecolor=C["grey"], elinewidth=0.9, capsize=2.5))
        for xi, mv in zip(x + off, means):
            ax.text(xi, mv + 0.015, f"{mv:.2f}", ha="center", va="bottom",
                    fontsize=6.5, color="#33404f")

    # emphasize the "ours" column
    if "ours" in models:
        oi = models.index("ours")
        ax.axvspan(oi - 0.5, oi + 0.5, color=C["lgrey"], alpha=0.35, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY.get(m, m) for m in models], fontsize=8)
    for lbl, m in zip(ax.get_xticklabels(), models):
        if m == "ours":
            lbl.set_fontweight("bold"); lbl.set_color(C["navy"])
    n_seq = rows[models[0]].get("n_seq", "?")
    ax.set_ylabel(f"Score (mean ± std across {n_seq} {args.split} videos)")
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=8,
              bbox_to_anchor=(0.0, 1.02))
    ax.set_title(f"Full-sequence propagation on live phase-contrast neurons — "
                 f"{SPLIT_LABEL[args.split]} (t≥1, identical frame-0 prompt)",
                 loc="left", fontsize=9, color=C["navy"], pad=18)
    ax.grid(axis="y", color=C["lgrey"], lw=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig3_benchmark_bars_{args.split}.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote fig3_benchmark_bars_{args.split}  for models:", ", ".join(models))


if __name__ == "__main__":
    main()
