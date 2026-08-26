#!/usr/bin/env python3
"""
make_figures.py -- method/positioning figures in the SAME unified palette as the
result figures (paper_figures.py). Produces:
  fig_capability_matrix  : where ours sits vs competitors (qualitative)
  fig_param_budget       : dual-rank LoRA+ trainable fraction
  fig_ablation           : cumulative effect of the design choices (IoU)

    venv/bin/python benchmark/make_figures.py
Outputs -> paper_assets/figures/fig_{capability_matrix,param_budget,ablation}.{pdf,png}
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(HERE), "paper_assets", "figures")

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.9, "figure.dpi": 120, "svg.fonttype": "none",
})
# unified palette (matches paper_figures.py)
NAVY = "#16324F"; BLUE = "#1F6FB2"; TEAL = "#2CA6A4"; DARK = "#123A5A"
GREEN = "#4F9D5D"; AMBER = "#E1A93B"; LGREY = "#D9DEE5"; GRID = "#E4E8ED"; MUTE = "#8B96A3"


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------
# capability matrix
# ---------------------------------------------------------------------------
def fig_capability_matrix():
    models = ["SAM 2 (base)", "µSAM", "Cellpose-SAM", "Omnipose", "NeuroQuantify", "Ours"]
    caps = ["Native temporal\ntracking", "Phase-contrast\ntrained", "Neuron / neurite\nspecialized",
            "Topology-aware\nloss", "Promptable", "Foundation-model\nbackbone"]
    M = np.array([
        [2, 0, 0, 0, 2, 2],   # SAM 2
        [1, 1, 0, 0, 2, 2],   # µSAM  (light-microscopy generalist)
        [0, 1, 0, 0, 0, 2],   # Cellpose-SAM
        [0, 2, 0, 0, 0, 0],   # Omnipose  (phase-contrast, elongated-morphology model)
        [0, 2, 2, 0, 0, 0],   # NeuroQuantify  (phase-contrast neuron+neurite U-Net)
        [2, 2, 2, 2, 2, 2],   # Ours
    ])
    cmap = {0: LGREY, 1: AMBER, 2: GREEN}
    sym = {0: "–", 1: "~", 2: "✓"}
    nr, nc = M.shape
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for i in range(nr):
        for j in range(nc):
            v = M[nr - 1 - i, j]
            ax.add_patch(FancyBboxPatch((j + 0.06, i + 0.06), 0.88, 0.88,
                         boxstyle="round,pad=0,rounding_size=0.10",
                         facecolor=cmap[v], edgecolor="white", linewidth=0, zorder=2))
            ax.text(j + 0.5, i + 0.5, sym[v], ha="center", va="center", fontsize=15,
                    color="white", fontweight="bold", zorder=3, fontfamily="DejaVu Sans")
    # spotlight the Ours row (bottom row, i index nr-1 -> y=0)
    ax.add_patch(FancyBboxPatch((0.02, 0.02), nc - 0.04, 0.96,
                 boxstyle="round,pad=0,rounding_size=0.12", fill=False,
                 edgecolor=NAVY, linewidth=2.4, zorder=4))
    ax.set_xlim(0, nc); ax.set_ylim(0, nr)
    ax.set_xticks(np.arange(nc) + 0.5); ax.set_xticklabels(caps, fontsize=8.2)
    ax.set_yticks(np.arange(nr) + 0.5); ax.set_yticklabels(models[::-1], fontsize=9.5)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() == "Ours":
            lbl.set_fontweight("bold"); lbl.set_color(NAVY)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Only a neuron-specialized, topology-aware, temporally-consistent method "
                 "fills every column", loc="left", fontsize=10.5, color=NAVY, pad=12, weight="bold")
    leg = [Patch(facecolor=GREEN, label="yes / native"), Patch(facecolor=AMBER, label="partial / external"),
           Patch(facecolor=LGREY, label="no")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3,
              frameon=False, fontsize=9)
    save(fig, "fig_capability_matrix")


# ---------------------------------------------------------------------------
# parameter budget
# ---------------------------------------------------------------------------
def fig_param_budget():
    trainable, total = 5_557_250, 39_876_930
    frozen = total - trainable
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.axis("equal")
    ax.pie([frozen, trainable], colors=[LGREY, NAVY], startangle=90, counterclock=False,
           wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2))
    ax.text(0, 0.14, "13.9%", ha="center", fontsize=22, fontweight="bold", color=NAVY)
    ax.text(0, -0.12, "trainable", ha="center", fontsize=10.5, color=NAVY)
    ax.text(0, -0.32, "5.56M / 39.88M params", ha="center", fontsize=8.5, color=MUTE)
    rows = [(BLUE, "LoRA r=16 (α=32) · 68 layers", "mask decoder + memory attention"),
            (TEAL, "LoRA r=8 (α=16) · 8 layers", "Hiera last-2 trunk blocks"),
            (NAVY, "unfrozen temporal / object embeddings", "")]
    y0 = -0.78
    for i, (c, a, b) in enumerate(rows):
        yy = y0 - i * 0.17
        ax.scatter([-1.35], [yy], s=42, color=c, zorder=5)
        ax.text(-1.24, yy, a, fontsize=8.3, va="center", fontweight="bold", color=NAVY)
        if b:
            ax.text(-1.24, yy - 0.085, b, fontsize=7.6, va="center", color=MUTE)
    ax.set_title("Dual-rank LoRA+ parameter budget", loc="center", fontsize=11, color=NAVY, weight="bold")
    ax.set_ylim(-1.5, 1.15)
    save(fig, "fig_param_budget")


# ---------------------------------------------------------------------------
# cumulative ablation (the design-choice journey)
# ---------------------------------------------------------------------------
def fig_ablation():
    runs = ["Baseline\n(rank 8)", "+ regularise", "+ cbDice wt\n− supervoxel", "+ val swap\n+ schedule",
            "+ Path B\n+ obj-fix + rank16", "+ Hiera\nLoRA", "+ 20 videos\n(shipped)"]
    ious = [0.6951, 0.7218, 0.7468, 0.7755, 0.8295, 0.8295, 0.8401]
    x = np.arange(len(runs))
    # colour: build-up in graded blues, final navy, the Path-B jump highlighted teal
    cols = [LGREY, "#C3CBD4", "#AAB6C2", "#93A2B2", TEAL, "#3E7CA8", NAVY]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.bar(x, ious, width=0.66, color=cols, edgecolor="white", linewidth=0.6, zorder=3)
    for xi, v in zip(x, ious):
        ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=8,
                fontweight="bold" if xi in (4, 6) else "normal",
                color=NAVY if xi in (4, 6) else "#5A6472")
    # highlight the big jump (Path B)
    ax.annotate("+0.054 — training the memory-attention adapters\n(SAM 2 video path)",
                xy=(4, 0.8295), xytext=(2.15, 0.895),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.3),
                fontsize=8, color="#1c7a78", fontweight="bold", ha="left")
    ax.set_xticks(x); ax.set_xticklabels(runs, fontsize=7.6)
    ax.set_ylabel("Validation IoU")
    ax.set_ylim(0.65, 0.90); ax.set_yticks(np.arange(0.65, 0.901, 0.05))
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.set_title("Cumulative effect of the design choices (held-out validation IoU)",
                 loc="left", fontsize=10.5, color=NAVY, pad=10, weight="bold")
    save(fig, "fig_ablation")


if __name__ == "__main__":
    fig_capability_matrix()
    fig_param_budget()
    fig_ablation()
