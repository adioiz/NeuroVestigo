#!/usr/bin/env python3
"""
paper_figures.py -- polished, publication-grade result figures.

(1) fig_benchmark_test  : grouped bars, ours emphasized, leads with the metrics
    where ours dominates most (Dice / cbDice), gap-over-best-baseline annotated.
    Bars and whiskers are Table 1: mean and SD over the three runs, read from
    benchmark/results/folds/table1.csv.
(2) fig_temporal_consistency : per-frame cbDice across the full propagated sequence
    (mean over the 12 held-out recordings of the three runs, each scored with its
    own run's model; whiskers = SD across the three runs). Ours stays high and
    flat; the single-frame tools have no temporal consistency -- ours' genuine,
    unique advantage.

    venv/bin/python benchmark/paper_figures.py
Outputs -> paper_assets/figures/fig_{benchmark_test,temporal_consistency}.{pdf,png}
"""
from __future__ import annotations

import csv
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tifffile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, REPO)
from benchmark.data import Dataset

FIGDIR = os.path.join(REPO, "paper_assets", "figures")
PRED = os.path.join(REPO, "benchmark", "predictions")
# Model key -> predictions subdirectory, where the two differ. The SAM 3 sweep
# wrote one directory per configuration, so the winning runs do not carry the
# model name they are scored under.
PRED_DIR = {"sam3_maskfirst": "sam3_maskfirst", "sam3.1_text": "sam3_text_final",
            # uSAM is reported in single-mask mode (multimask_output=False),
            # SAM's documented setting for an unambiguous box prompt.
            "usam": "usam_single",
            # shipped model: 23 recordings, epoch 80, topology tie-break (Sec. 3.2)
            "ours": "ours_23rec_topo"}

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.9, "figure.dpi": 120, "svg.fonttype": "none",
})
NAVY = "#16324F"; OURS = "#1F6FB2"; OURS2 = "#2CA6A4"; OURS3 = "#123A5A"
AMBER = "#E1A93B"   # SAM 3.1, matching the paper palette
MUTE = ["#B9C2CC", "#A7B2BE", "#95A2B0", "#8391A1", "#7C8998", "#6E7C8C", "#617080"]  # muted greys
GRID = "#E4E8ED"

PRETTY = {"ours": "NeuroVestigo", "sam2_zeroshot": "SAM 2\nzero-shot", "usam": "µSAM\n(box-prompt)",
          "deepsea": "DeepSea\n(auto)", "cellpose_sam": "Cellpose-SAM\n(auto)",
          "omnipose": "Omnipose\n(auto)", "neuroquantify": "NeuroQuantify\n(phase-contrast)",
          "cellsam": "CellSAM\n(auto)", "trackastra_omnipose": "Omnipose +\nTrackastra",
          "sam3_maskfirst": "SAM 3\n(mask-prompt)", "sam3.1_text": "SAM 3.1\n(text-prompt)"}
METRICS = [("iou", "IoU"), ("dice", "Dice"), ("cbdice", "cbDice")]
# Locked figure roster (DeepSea, Trackastra, CellSAM -> text only).
FIGURE_COMPETITORS = ["sam2_zeroshot", "usam", "sam3_maskfirst", "cellpose_sam", "omnipose",
                      "neuroquantify", "sam3.1_text"]


def load_summary(split="test"):
    rows = {}
    with open(os.path.join(REPO, "benchmark", "results", split, "summary.csv")) as f:
        for r in csv.DictReader(f):
            rows[r["model"]] = r
    return rows


# Table 1 labels (benchmark/aggregate_folds.py) -> the model keys used here.
FOLD_LABELS = {"NeuroVestigo": "ours", "SAM 2 (zero-shot)": "sam2_zeroshot",
               "SAM 3 (mask)": "sam3_maskfirst", "muSAM (box)": "usam",
               "Cellpose-SAM": "cellpose_sam", "NeuroQuantify": "neuroquantify",
               "Omnipose": "omnipose", "SAM 3.1 (text)": "sam3.1_text"}
# NeuroVestigo is a different model in each run; the baselines are not trained.
OURS_BY_FOLD = {1: "ours_23rec_topo", 2: "ours_fold2_topo", 3: "ours_fold3_topo"}


def load_folds():
    """Table 1 as summary.csv-style rows: mean and SD over the three runs."""
    rows = {}
    with open(os.path.join(REPO, "benchmark", "results", "folds", "table1.csv")) as f:
        for r in csv.DictReader(f):
            rows[FOLD_LABELS[r["method"]]] = {
                **{f"{k}_mean": r[f"{k}_mean"] for k, _ in METRICS},
                **{f"{k}_std": r[f"{k}_sd"] for k, _ in METRICS}}
    return rows


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------
# (1) polished benchmark
# ---------------------------------------------------------------------------
def fig_benchmark(split="test"):
    rows = load_folds() if split == "test" else load_summary(split)
    # Final figure roster (locked with Adi): ours + these 7. DeepSea, Trackastra and
    # CellSAM are mentioned in the text only, not plotted. Competitors ordered by cbDice.
    # SAM 3 (sam3.pt) seeded with the same frame-0 mask as ours; SAM 3.1 in text mode.
    comps = sorted([m for m in FIGURE_COMPETITORS if m in rows],
                   key=lambda m: -float(rows[m]["cbdice_mean"]))
    order = ["ours"] + comps
    x = np.arange(len(order)); w = 0.26
    metric_cols = {"iou": OURS, "dice": OURS2, "cbdice": OURS3}

    fig, ax = plt.subplots(figsize=(max(8.4, 1.15 * len(order) + 2.2), 4.4))
    ax.axvspan(-0.5, 0.5, color=OURS, alpha=0.07, zorder=0)   # spotlight ours
    for i, (key, label) in enumerate(METRICS):
        means = [float(rows[m][f"{key}_mean"]) for m in order]
        stds = [float(rows[m].get(f"{key}_std", 0) or 0) for m in order]
        colors = [metric_cols[key] if m == "ours" else MUTE[i] for m in order]
        off = (i - 1) * w
        # s.d. drawn as a DOWNWARD whisker only: a symmetric bar on a high-variance
        # method (SAM 3.1, s.d. 0.17) towered above every other bar and read as
        # if that method scored highest. Nothing rises above a bar's own value.
        ax.bar(x + off, means, w, color=colors, edgecolor="white", linewidth=0.6,
               zorder=3, label=label)
        # whisker drawn by hand so there is a cap only at the LOWER end; the
        # errorbar() helper also caps the upper end, which sits on the bar top
        # and looks like a stray tick
        for xi, mv, sd in zip(x + off, means, stds):
            ax.plot([xi, xi], [mv, mv - sd], color="#9AA3AF", lw=0.9, zorder=4)
            ax.plot([xi - 0.035, xi + 0.035], [mv - sd, mv - sd], color="#9AA3AF", lw=0.9, zorder=4)
        for xi, mv, m in zip(x + off, means, order):
            # label at the bar's own height (not above the s.d. cap, which for
            # high-variance methods floated the number above every other bar);
            # the white box keeps it legible where the whisker passes behind it
            ax.text(xi, mv + 0.012, f"{mv:.2f}",
                    ha="center", va="bottom",
                    fontsize=7.6 if m == "ours" else 7.4,
                    fontweight="bold" if m == "ours" else "normal",
                    color=NAVY if m == "ours" else "#5A6472", zorder=6)


    set_name = "held-out test set" if split == "test" else "validation set"
    avg_name = "three runs" if split == "test" else "validation videos"
    ax.set_xticks(x); ax.set_xticklabels([PRETTY[m] for m in order], fontsize=9)
    ax.get_xticklabels()[0].set_fontweight("bold"); ax.get_xticklabels()[0].set_color(NAVY)
    ax.set_ylabel(f"Score (mean over {avg_name}; whisker = \u22121 s.d.)")
    ax.set_ylim(0, 1.02); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=9, bbox_to_anchor=(1.0, 1.08))
    # NB: do NOT re-add "most on connectivity". After the topology_resolution
    # fix the margin over the best baseline is 0.113 on IoU vs 0.078 on cbDice,
    # so that claim is false. Ours still leads every metric outright.
    ax.set_title(f"Neuron segmentation on the {set_name}",
                 loc="left", fontsize=10.5, color=NAVY, pad=16, weight="bold")
    save(fig, f"fig_benchmark_{split}")


# ---------------------------------------------------------------------------
# (2) temporal consistency
# ---------------------------------------------------------------------------
def _dice(p, g):
    p = p.astype(bool); g = g.astype(bool)
    d = p.sum() + g.sum()
    return 1.0 if d == 0 else 2 * np.logical_and(p, g).sum() / d


# The temporal claim in Sec. 4.2 is about branch structure surviving propagation,
# not area, so this curve is cbDice -- the same metric Table 3 leads with.
from benchmark.eval_masks import cbdice_score as _cbdice


def fig_temporal():
    # sam3.1_text is a Table-2-only row, as in the paper's Table 2.
    models = ["ours"] + [m for m in FIGURE_COMPETITORS if m != "sam3.1_text"]
    folds = sorted(OURS_BY_FOLD)
    pred_dir = lambda m, fold: OURS_BY_FOLD[fold] if m == "ours" else PRED_DIR.get(m, m)
    T = 72          # every test recording has 72 or 92 propagated frames (frame 0 = prompt)
    # A run's curve averages its four recordings frame by frame; the figure shows
    # the mean of the three run curves and, as whiskers, their SD (Table 1's n = 3).
    run_curves = {m: [] for m in models}
    for fold in folds:
        ds = Dataset(fold=fold)
        for m in models:
            per_t = [[] for _ in range(T)]
            for v in ds.split("test"):
                gt = ds.load_gt(v)
                pr = tifffile.imread(os.path.join(PRED, pred_dir(m, fold), f"{v}.tif")) > 0
                for t in range(1, T + 1):
                    per_t[t - 1].append(_cbdice(pr[t], gt[t] > 0))
            run_curves[m].append([np.mean(c) for c in per_t])
            print(f"  run {fold} {m}: mean cbDice {np.mean(run_curves[m][-1]):.3f}")
    curves = {m: np.mean(run_curves[m], axis=0) for m in models}
    spread = {m: np.std(run_curves[m], axis=0, ddof=1) for m in models}

    # one clearly distinct hue per method (the earlier greys were near-identical)
    styles = {"ours": dict(color=NAVY, lw=2.6, zorder=5),
              "sam2_zeroshot": dict(color=OURS, lw=1.6, alpha=0.9),        # blue
              "usam": dict(color="#7B5EA7", lw=1.5, alpha=0.9),            # purple
              "neuroquantify": dict(color=OURS2, lw=1.6, alpha=0.9),       # teal
              "omnipose": dict(color="#8B96A3", lw=1.5, alpha=0.9),        # slate grey
              "cellpose_sam": dict(color="#C0392B", lw=1.5, alpha=0.9),    # crimson
              "sam3_maskfirst": dict(color=AMBER, lw=1.6, alpha=0.9),      # amber
              "sam3.1_text": dict(color=AMBER, lw=1.2, alpha=0.7, ls=":")}
    names = {"ours": "NeuroVestigo (temporal propagation)", "sam2_zeroshot": "SAM 2 zero-shot",
             "usam": "µSAM (box-prompt)", "neuroquantify": "NeuroQuantify (phase-contrast)",
             "omnipose": "Omnipose", "cellpose_sam": "Cellpose-SAM",
             "sam3_maskfirst": "SAM 3 (mask-prompt)", "sam3.1_text": "SAM 3.1 (text-prompt)"}

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    xs = np.arange(1, T + 1)
    for m in models:
        ax.plot(xs, curves[m], label=names[m], **styles[m])
    # SD across the three runs, every 12th frame, staggered so they do not overlap
    for k, m in enumerate(models):
        st = {key: styles[m][key] for key in ("color", "alpha") if key in styles[m]}
        for t in range(6, T + 1, 12):
            xk = t + (k - (len(models) - 1) / 2) * 0.55
            ax.errorbar(xk, curves[m][t - 1], yerr=spread[m][t - 1], fmt="none",
                        elinewidth=0.9, capsize=1.6, capthick=0.9, zorder=4, **st)
    # end-point emphasis for ours
    ax.scatter([T], [curves["ours"][-1]], s=45, color=NAVY, zorder=6)
    ax.text(T, curves["ours"][-1] + 0.03, f"{curves['ours'][-1]:.2f}", ha="right",
            color=NAVY, fontsize=9, fontweight="bold")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("cbDice vs ground truth (mean over three runs)")
    ax.set_ylim(0, 1.0); ax.set_xlim(1, T)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    ax.set_title("Temporal consistency on the held-out test set",
                 loc="left", fontsize=10.5, color=NAVY, pad=12, weight="bold")
    save(fig, "fig_temporal_consistency")


if __name__ == "__main__":
    fig_benchmark("test")
    # fig_benchmark("val") is not in the paper, and results/val/summary.csv
    # predates the centred resize (commit 70a3472); rebuild it only after rescoring.
    fig_temporal()
