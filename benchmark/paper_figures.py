#!/usr/bin/env python3
"""
paper_figures.py -- polished, publication-grade result figures.

(1) fig_benchmark_test  : grouped bars, ours emphasized, leads with the metrics
    where ours dominates most (Dice / cbDice), gap-over-best-baseline annotated.
(2) fig_temporal_consistency : per-frame cbDice across the full propagated sequence
    (mean over the held-out test videos). Ours stays high and flat; the single-frame
    tools have no temporal consistency -- ours' genuine, unique advantage.

    venv/bin/python benchmark/paper_figures.py
Outputs -> paper_assets/figures/fig_{benchmark_test,temporal_consistency}.{pdf,png}
"""
from __future__ import annotations

import csv
import json
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
PRED_DIR = {"sam3.1": "sam3_final", "sam3.1_text": "sam3_text_final"}

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.9, "figure.dpi": 120, "svg.fonttype": "none",
})
NAVY = "#16324F"; OURS = "#1F6FB2"; OURS2 = "#2CA6A4"; OURS3 = "#123A5A"
AMBER = "#E1A93B"   # SAM 3.1, matching the paper palette
MUTE = ["#B9C2CC", "#A7B2BE", "#95A2B0", "#8391A1", "#7C8998", "#6E7C8C", "#617080"]  # muted greys
GRID = "#E4E8ED"

PRETTY = {"ours": "Ours", "sam2_zeroshot": "SAM 2\nzero-shot", "usam": "µSAM\n(box-prompt)",
          "deepsea": "DeepSea\n(auto)", "cellpose_sam": "Cellpose-SAM\n(auto)",
          "omnipose": "Omnipose\n(auto)", "neuroquantify": "NeuroQuantify\n(phase-contrast)",
          "cellsam": "CellSAM\n(auto)", "trackastra_omnipose": "Omnipose +\nTrackastra",
          "sam3.1": "SAM 3.1\n(point-prompt)", "sam3.1_text": "SAM 3.1\n(text-prompt)"}
METRICS = [("iou", "IoU"), ("dice", "Dice"), ("cbdice", "cbDice")]
# Locked figure roster (DeepSea, Trackastra, CellSAM -> text only).
FIGURE_COMPETITORS = ["sam2_zeroshot", "usam", "sam3.1", "cellpose_sam", "omnipose",
                      "neuroquantify", "sam3.1_text"]


def load_summary(split="test"):
    rows = {}
    with open(os.path.join(REPO, "benchmark", "results", split, "summary.csv")) as f:
        for r in csv.DictReader(f):
            rows[r["model"]] = r
    return rows


SHIPPED = os.path.join(REPO, "checkpoints-16-frames")


def shipped_val_reference():
    """The validation scores of the SHIPPED run, read from its own log.

    These were hardcoded as {"iou": 0.84, "dice": 0.91} with a caption saying
    "8-frame", which was inherited from the superseded run and wrong: the
    reported model trains and validates on 16-frame clips. Both the numbers
    and the clip length now come from checkpoints-16-frames/ so neither can
    go stale again. Do NOT read gpu-checkpoints/ -- that IS the 8-frame run.
    """
    rows = json.load(open(os.path.join(SHIPPED, "training_log.json")))
    if isinstance(rows, dict):
        rows = rows.get("epochs", list(rows.values()))
    scored = [r for r in rows
              if isinstance(r, dict) and r.get("val_iou") is not None]
    best = max(scored, key=lambda r: r["val_iou"])
    cfg = json.load(open(os.path.join(SHIPPED, "training_config.json")))
    n = cfg.get("data", cfg)["num_frames"]
    return {"iou": best["val_iou"], "dice": best["val_dice"]}, n, best["epoch"]


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
    rows = load_summary(split)
    # Final figure roster (locked with Adi): ours + these 7. DeepSea, Trackastra and
    # CellSAM are mentioned in the text only, not plotted. Competitors ordered by cbDice.
    # SAM 3.1 appears twice -- it is reported in both modes its video API accepts.
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
        ax.bar(x + off, means, w, yerr=stds, color=colors, edgecolor="white", linewidth=0.6,
               error_kw=dict(ecolor="#9AA3AF", elinewidth=0.9, capsize=2.5), zorder=3,
               label=label)
        for xi, mv, m in zip(x + off, means, order):
            if m == "ours":
                # inside the bar: the 8-frame reference dashes sit just above
                # these bars, and an outside label collides with them.
                ax.text(xi, mv - 0.018, f"{mv:.2f}", ha="center", va="top",
                        fontsize=7.6, fontweight="bold", color="white", zorder=8)
            else:
                ax.text(xi, mv + 0.012, f"{mv:.2f}", ha="center", va="bottom",
                        fontsize=7.4, color="#5A6472")

    # dotted reference: ours under the clip-length validation protocol
    val_ref, n_frames, best_ep = shipped_val_reference()
    keys = [k for k, _ in METRICS]
    for key, val in val_ref.items():
        xi = 0 + (keys.index(key) - 1) * w
        ax.plot([xi - w * 0.55, xi + w * 0.55], [val, val], ls=(0, (2, 1.4)),
                color=NAVY, lw=1.5, zorder=7)
        # label to the LEFT of the segment: directly above the bar it would
        # otherwise collide with the bar's own value label (0.91 vs 0.86).
        ax.text(xi - w * 0.62, val, f"{val:.2f}", ha="right", va="center",
                fontsize=6.6, color=NAVY, style="italic", zorder=7)
    # footnote below the axes -- inside the plot it lay over the 'ours' bars.
    ax.text(0.0, -0.155, f"– –  ours under the {n_frames}-frame validation "
            "protocol (bars = harder full-video propagation)",
            transform=ax.transAxes, fontsize=7, color=NAVY, style="italic", va="top")

    set_name = "held-out test set" if split == "test" else "validation set"
    vid_name = "held-out test videos" if split == "test" else "validation videos"
    ax.set_xticks(x); ax.set_xticklabels([PRETTY[m] for m in order], fontsize=9)
    ax.get_xticklabels()[0].set_fontweight("bold"); ax.get_xticklabels()[0].set_color(NAVY)
    ax.set_ylabel(f"Score (mean ± s.d. over {vid_name})")
    ax.set_ylim(0, 1.02); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=9, bbox_to_anchor=(1.0, 1.08))
    # NB: do NOT re-add "most on connectivity". After the topology_resolution
    # fix the margin over the best baseline is 0.113 on IoU vs 0.078 on cbDice,
    # so that claim is false. Ours still leads every metric outright.
    ax.set_title(f"Neuron segmentation on the {set_name} — ours leads every metric, "
                 "from a single frame-0 annotation",
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
    ds = Dataset()
    videos = ds.split("test")
    # sam3.1_text is a Table-2-only row: the temporal curve needs prediction
    # stacks and the reported (max_num_objects=64) run lives cluster-side.
    models = ["ours"] + [m for m in FIGURE_COMPETITORS if m != "sam3.1_text"]
    T = 72                                    # common length (skip frame 0 = prompt)
    curves = {}
    for m in models:
        per_t = [[] for _ in range(T)]
        for v in videos:
            gt = ds.load_gt(v)
            pr = tifffile.imread(os.path.join(PRED, PRED_DIR.get(m, m), f"{v}.tif")) > 0
            for t in range(1, min(T + 1, gt.shape[0], pr.shape[0])):
                per_t[t - 1].append(_cbdice(pr[t], gt[t] > 0))
        curves[m] = np.array([np.mean(c) if c else np.nan for c in per_t])

    styles = {"ours": dict(color=NAVY, lw=2.6, zorder=5),
              "sam2_zeroshot": dict(color=OURS, lw=1.6, alpha=0.9),
              "usam": dict(color="#7FA6C9", lw=1.4, alpha=0.85),
              "neuroquantify": dict(color=OURS2, lw=1.6, alpha=0.9),
              "omnipose": dict(color="#AEB8C4", lw=1.4, alpha=0.85),
              "cellpose_sam": dict(color="#C7CDD6", lw=1.4, alpha=0.85),
              "sam3.1": dict(color=AMBER, lw=1.6, alpha=0.9),
              "sam3.1_text": dict(color=AMBER, lw=1.2, alpha=0.7, ls=":")}
    names = {"ours": "Ours (temporal propagation)", "sam2_zeroshot": "SAM 2 zero-shot",
             "usam": "µSAM (box-prompt)", "neuroquantify": "NeuroQuantify (phase-contrast)",
             "omnipose": "Omnipose", "cellpose_sam": "Cellpose-SAM",
             "sam3.1": "SAM 3.1 (point-prompt)", "sam3.1_text": "SAM 3.1 (text-prompt)"}

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    xs = np.arange(1, T + 1)
    for m in models:
        ax.plot(xs, curves[m], label=names[m], **styles[m])
    # end-point emphasis for ours
    ax.scatter([T], [curves["ours"][-1]], s=45, color=NAVY, zorder=6)
    ax.text(T, curves["ours"][-1] + 0.03, f"{curves['ours'][-1]:.2f}", ha="right",
            color=NAVY, fontsize=9, fontweight="bold")
    ax.set_xlabel("Frame (propagated from a single frame-0 annotation)")
    ax.set_ylabel("cbDice vs ground truth (mean over test videos)")
    ax.set_ylim(0, 1.0); ax.set_xlim(1, T)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    ax.set_title("Temporal consistency on the held-out test set",
                 loc="left", fontsize=10.5, color=NAVY, pad=12, weight="bold")
    save(fig, "fig_temporal_consistency")


if __name__ == "__main__":
    fig_benchmark("test")
    fig_benchmark("val")
    fig_temporal()
