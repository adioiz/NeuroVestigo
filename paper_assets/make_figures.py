#!/usr/bin/env python3
"""
Publication-quality draft figures for the neuron SAM2 paper.
All data is REAL and on-disk: gpu-checkpoints/training_log.json + REPORT.tex Run-7 numbers.
Figures that require a competitor benchmark run (comparison grid, clDice bars, CTC) are
NOT produced here -- they need eval_masks.py + competitor inference first.

Outputs -> paper_assets/figures/*.pdf and *.png (300 dpi)
"""
import json, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 120,
})

# Nature-ish colorblind-friendly palette
C = {
    "navy":  "#1F3A5F",
    "blue":  "#3B6FB6",
    "teal":  "#3AA7A0",
    "amber":  "#E1A93B",
    "rust":  "#C6592F",
    "grey":  "#9AA3AF",
    "lgrey": "#D5DAE1",
    "green": "#4F9D5D",
}

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

log = json.load(open(os.path.join(ROOT, "gpu-checkpoints", "training_log.json")))
val = [e for e in log if "val_iou" in e]
best = max(val, key=lambda e: e["val_iou"])

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", name)

# ----------------------------------------------------------------------------
# FIG 1 — Training journey / cumulative ablation (from REPORT.tex Table journey)
# ----------------------------------------------------------------------------
runs   = ["Run 1\nbaseline", "Run 2\n+reg", "Run 3\ncbDice+, sv off",
         "Run 4\nval swap", "Run 5\nPath B\n+obj-fix\n+rank16", "Run 6\n+Hiera\nLoRA",
         "Run 7\n20 vids\n(ship)"]
ious   = [0.6951, 0.7218, 0.7468, 0.7755, 0.8295, 0.8295, 0.8401]
colors = [C["grey"]]*4 + [C["rust"], C["amber"], C["navy"]]

fig, ax = plt.subplots(figsize=(7.2, 3.4))
x = np.arange(len(runs))
bars = ax.bar(x, ious, color=colors, width=0.66, edgecolor="white", linewidth=0.6)
for xi, v in zip(x, ious):
    ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=8,
            fontweight="bold" if xi in (4, 6) else "normal")
# highlight the two big jumps
ax.annotate("", xy=(4, 0.8295), xytext=(3, 0.7755),
            arrowprops=dict(arrowstyle="->", color=C["rust"], lw=1.4))
ax.text(3.5, 0.845, "+0.054\nmemory-attn\nadapters trained", ha="center",
        va="bottom", fontsize=7, color=C["rust"])
ax.set_xticks(x); ax.set_xticklabels(runs, fontsize=7)
ax.set_ylabel("Validation IoU")
ax.set_ylim(0.65, 0.88)
ax.set_title("Cumulative effect of design changes (in-house 5-video val split)", loc="left")
ax.axhline(0.8401, color=C["navy"], ls=":", lw=0.8, zorder=0)
save(fig, "fig1_training_journey")

# ----------------------------------------------------------------------------
# FIG 2 — Temporal degradation (per-frame IoU, Run-7 best epoch, from REPORT)
# ----------------------------------------------------------------------------
# canonical Run-7 numbers from REPORT.tex
pf_report = [0.946, 0.837, 0.825, 0.828, 0.827, 0.819, 0.818, 0.822]
pf_log    = best["per_frame_iou"]
t = np.arange(8)

fig, ax = plt.subplots(figsize=(5.0, 3.4))
ax.plot(t, pf_report, "-o", color=C["navy"], lw=2, ms=6, label="fine-tuned SAM 2 (ours)")
ax.axvspan(-0.3, 0.3, color=C["lgrey"], alpha=0.6, zorder=0)
ax.text(0, 0.9, "prompt\nframe", ha="center", va="top", fontsize=7, color=C["grey"])
ax.annotate(f"−{pf_report[0]-pf_report[7]:.3f} over 7 propagated frames",
            xy=(7, pf_report[7]), xytext=(3.4, 0.90),
            arrowprops=dict(arrowstyle="->", color=C["rust"], lw=1.1),
            fontsize=8, color=C["rust"])
ax.set_xlabel("Frame index (propagated from frame 0)")
ax.set_ylabel("Mean IoU across val videos")
ax.set_ylim(0.80, 0.96)
ax.set_xticks(t)
ax.set_title("Temporal propagation stability", loc="left")
ax.legend(loc="upper right", frameon=False)
save(fig, "fig2_temporal_degradation")

# ----------------------------------------------------------------------------
# FIG 3 — Per-video IoU (Run-7 best epoch, from REPORT)
# ----------------------------------------------------------------------------
vids = ["A91_C1_2", "A31_C1", "A52_C2", "A22_C2", "A42_C1"]
pv   = [0.793, 0.873, 0.844, 0.840, 0.850]
mean = 0.8401
fig, ax = plt.subplots(figsize=(5.0, 3.4))
xx = np.arange(len(vids))
cols = [C["rust"] if v < 0.80 else C["blue"] for v in pv]
ax.bar(xx, pv, color=cols, width=0.6, edgecolor="white")
for xi, v in zip(xx, pv):
    ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
ax.axhline(mean, color=C["navy"], ls="--", lw=1)
ax.text(len(vids)-0.5, mean + 0.004, f"mean {mean:.3f}", ha="right", color=C["navy"], fontsize=8)
ax.set_xticks(xx); ax.set_xticklabels(vids, rotation=20, ha="right", fontsize=7.5)
ax.set_ylabel("Validation IoU")
ax.set_ylim(0.70, 0.90)
ax.set_title("Per-video validation IoU (5 held-out videos)", loc="left")
save(fig, "fig3_per_video")

# ----------------------------------------------------------------------------
# FIG 4 — Learning curves over epochs (from training_log.json, real on-disk)
# ----------------------------------------------------------------------------
ep   = [e["epoch"] for e in val]
viou = [e["val_iou"] for e in val]
vdice= [e["val_dice"] for e in val]
fig, ax = plt.subplots(figsize=(6.0, 3.4))
ax.plot(ep, viou, "-o", color=C["navy"], ms=4, lw=1.6, label="val IoU")
ax.plot(ep, vdice, "-s", color=C["teal"], ms=4, lw=1.6, label="val Dice")
bi = best["epoch"]
ax.scatter([bi], [best["val_iou"]], s=70, facecolor="none", edgecolor=C["rust"], lw=1.6, zorder=5)
ax.annotate(f"best {best['val_iou']:.4f} @ ep {bi}", xy=(bi, best["val_iou"]),
            xytext=(bi-14, best["val_iou"]-0.006), fontsize=7.5, color=C["rust"])
ax.set_xlabel("Epoch"); ax.set_ylabel("Metric")
ax.set_title("Validation learning curves (on-disk training log)", loc="left")
ax.legend(loc="lower right", frameon=False)
save(fig, "fig4_learning_curves")

# ----------------------------------------------------------------------------
# FIG 5 — Parameter budget donut (from REPORT: 5,557,250 / 39,876,930 = 13.9%)
# ----------------------------------------------------------------------------
trainable = 5_557_250
total     = 39_876_930
frozen    = total - trainable
fig, ax = plt.subplots(figsize=(4.2, 3.6))
ax.axis("equal")
w = 0.42
ax.pie([frozen, trainable], colors=[C["lgrey"], C["navy"]],
       startangle=90, counterclock=False,
       wedgeprops=dict(width=w, edgecolor="white", linewidth=1.5))
ax.text(0, 0.12, "13.9%", ha="center", fontsize=18, fontweight="bold", color=C["navy"])
ax.text(0, -0.14, "trainable", ha="center", fontsize=9, color=C["navy"])
ax.text(0, -0.34, "5.56M / 39.88M", ha="center", fontsize=8, color=C["grey"])
# legend text for adapter placement
lines = [
    ("●", C["navy"], "LoRA r=16 (α=32): 68 layers"),
    ("",       C["navy"], "  mask decoder + memory attention"),
    ("●", C["blue"], "LoRA r=8 (α=16): 8 layers"),
    ("",       C["blue"], "  Hiera last-2 trunk blocks"),
    ("●", C["teal"], "unfrozen temporal/obj embeddings"),
]
y0 = -0.75
for i,(m,c,txt) in enumerate(lines):
    ax.text(-1.35, y0 - i*0.16, m, color=c, fontsize=9, transform=ax.transData)
    ax.text(-1.18, y0 - i*0.16, txt, fontsize=7.5, va="center", transform=ax.transData)
ax.set_title("Dual-rank LoRA+ parameter budget", loc="center")
ax.set_ylim(-1.6, 1.15)
save(fig, "fig5_param_budget")

# ----------------------------------------------------------------------------
# FIG 6 — Weighted val loss decomposition at best epoch (from training_log.json)
# ----------------------------------------------------------------------------
lc = best["loss_components"]
w  = {"loss_mask":5.0, "loss_dice":2.0, "loss_cbdice":5.0, "loss_supervoxel":0.3,
      "loss_iou":1.0, "loss_class":1.0}
names  = ["focal", "Dice", "cbDice", "supervoxel"]
keys   = ["loss_mask", "loss_dice", "loss_cbdice", "loss_supervoxel"]
weighted = [w[k]*lc[k] for k in keys]
raw      = [lc[k] for k in keys]
fig, ax = plt.subplots(figsize=(5.2, 3.2))
xx = np.arange(len(names))
ax.bar(xx, weighted, color=[C["blue"], C["navy"], C["teal"], C["amber"]], width=0.6, edgecolor="white")
for xi,(wv,rv,k) in enumerate(zip(weighted, raw, keys)):
    ax.text(xi, wv + 0.004, f"w={w[k]:g}\nraw={rv:.3f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(xx); ax.set_xticklabels(names)
ax.set_ylabel("Weighted contribution to val loss")
ax.set_title("Loss-component decomposition at best epoch", loc="left")
ax.set_ylim(0, max(weighted)*1.35)
save(fig, "fig6_loss_decomposition")

# ----------------------------------------------------------------------------
# FIG 7 — Capability matrix (qualitative facts from the competitor research)
# ----------------------------------------------------------------------------
# Cells: 2 = yes/native, 1 = partial/external/indirect, 0 = no
models = ["SAM 2\n(base)", "µSAM", "Cellpose-\nSAM", "CellSAM", "DeepSea", "Ours"]
caps   = ["Native temporal\ntracking", "Phase-contrast\ntrained", "Neuron/neurite\nspecialized",
          "Topology-aware\nloss", "Promptable", "Foundation-model\nbackbone"]
# rows = models, cols = caps
M = np.array([
    [2, 0, 0, 0, 2, 2],  # SAM 2
    [1, 2, 0, 0, 2, 2],  # µSAM (tracking via Trackastra linking; SAM1)
    [0, 2, 0, 0, 0, 2],  # Cellpose-SAM (automatic, single-frame)
    [0, 1, 0, 0, 2, 2],  # CellSAM (single-frame; phase-contrast one of many)
    [2, 2, 0, 0, 0, 0],  # DeepSea (own tracker; no foundation model)
    [2, 2, 2, 2, 2, 2],  # Ours
])
cmap = {0: C["lgrey"], 1: C["amber"], 2: C["green"]}
sym  = {0: "–", 1: "~", 2: "✓"}   # – ~ ✓
fig, ax = plt.subplots(figsize=(7.6, 4.2))
nr, nc = M.shape
for i in range(nr):
    for j in range(nc):
        val = M[i, j]
        face = cmap[val]
        # emphasize the "Ours" row
        ax.add_patch(plt.Rectangle((j, nr-1-i), 1, 1, facecolor=face,
                     edgecolor="white", lw=2, zorder=1))
        ax.text(j+0.5, nr-1-i+0.5, sym[val], ha="center", va="center",
                fontsize=14, color="white", fontweight="bold", zorder=2,
                fontfamily="DejaVu Sans")
ax.add_patch(plt.Rectangle((0, 0), nc, 1, fill=False, edgecolor=C["navy"], lw=2.4, zorder=3))
ax.set_xlim(0, nc); ax.set_ylim(0, nr)
ax.set_xticks(np.arange(nc)+0.5); ax.set_xticklabels(caps, fontsize=7.5)
ax.set_yticks(np.arange(nr)+0.5); ax.set_yticklabels(models[::-1], fontsize=8.5)
ax.set_yticklabels([m for m in models[::-1]], fontsize=8.5)
for lbl in ax.get_yticklabels():
    if lbl.get_text() == "Ours":
        lbl.set_fontweight("bold"); lbl.set_color(C["navy"])
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title("Capability comparison: only a neuron-specialized topology+temporal method fills every column",
             loc="left", fontsize=9, color=C["navy"], pad=10)
# legend
from matplotlib.patches import Patch
leg = [Patch(facecolor=C["green"], label="yes / native"),
       Patch(facecolor=C["amber"], label="partial / external"),
       Patch(facecolor=C["lgrey"], label="no")]
ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.12),
          ncol=3, frameon=False, fontsize=8)
save(fig, "fig7_capability_matrix")

print("\nAll real-data figures written to", FIGDIR)
