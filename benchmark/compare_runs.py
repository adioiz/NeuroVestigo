#!/usr/bin/env python3
"""INTERNAL, NOT FOR THE PAPER -- visual diff of the two training runs.

Left column  : the predecessor, trained with the INERT cbDice (predictions
               preserved in benchmark/predictions/_ours_oldmodel).
Right column : the retrained epoch-72 model, repaired cbDice at weight 1.0.
Bottom strip : per-frame clDice for both, over every scored test frame.

Overlay semantics match make_proof_panel.py:
    yellow = correct, green = missed ground truth, red = extra prediction.
Frames are chosen automatically as those where the two runs disagree most on
clDice, so the panel shows the real differences rather than a flattering pick.

    python benchmark/compare_runs.py
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from benchmark.make_proof_panel import load_gray, load_gt, overlay, bbox  # noqa: E402
from benchmark.eval_masks import cldice_score  # noqa: E402

OLD = os.path.join(REPO, "benchmark", "predictions", "_ours_oldmodel")
NEW = os.path.join(REPO, "benchmark", "predictions", "ours_cb03")
TEST = ["A71_C1", "A81_C1", "A12_C3_100k", "A12_C1_100k_2"]
OUT = os.path.join(REPO, "paper_assets", "figures", "_internal_A_vs_D_worstcase.png")


def main():
    # ---- per-frame clDice for both runs -------------------------------
    rows = []
    for v in TEST:
        po = tifffile.imread(os.path.join(OLD, f"{v}.tif"))
        pn = tifffile.imread(os.path.join(NEW, f"{v}.tif"))
        gts = sorted(glob.glob(os.path.join(REPO, "training_data", "masks", v, "*.png")))
        for t in range(1, min(len(po), len(pn))):
            from PIL import Image
            g = np.array(Image.open(gts[t])) > 0
            rows.append((v, t, cldice_score(po[t].astype(bool), g),
                         cldice_score(pn[t].astype(bool), g)))
    arr_old = np.array([r[2] for r in rows])
    arr_new = np.array([r[3] for r in rows])
    diff = arr_new - arr_old
    # Paper convention (summary.csv / Table 3): mean within a recording, then
    # recordings weighted equally. Pooling frames instead over-weights the two
    # 92-frame recordings, which are also the two weaker ones.
    pv_old = np.mean([np.mean([r[2] for r in rows if r[0] == v]) for v in TEST])
    pv_new = np.mean([np.mean([r[3] for r in rows if r[0] == v]) for v in TEST])

    # 3 frames where NEW loses most, 1 where it gains most
    order = np.argsort(diff)
    picks = [rows[i] for i in list(order[:3]) + [order[-1]]]

    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(5, 4, height_ratios=[1, 1, 1, 1, 0.85], hspace=0.32, wspace=0.06)

    for col, (v, t, c_old, c_new) in enumerate(picks):
        gray, gt = load_gray(v, t), load_gt(v, t)
        m_old = tifffile.imread(os.path.join(OLD, f"{v}.tif"))[t].astype(bool)
        m_new = tifffile.imread(os.path.join(NEW, f"{v}.tif"))[t].astype(bool)
        y0, y1, x0, x1 = bbox([gt, m_old, m_new], margin=14)
        for row, (mask, cl, tag) in enumerate(
                [(m_old, c_old, "OLD  (inert cbDice)"), (m_new, c_new, "NEW  (cbDice 0.3, ep84)")]):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(overlay(gray, gt, mask)[y0:y1, x0:x1])
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(tag, fontsize=10, fontweight="bold")
            ax.set_title(f"{v}  t={t}\nclDice {cl:.3f}", fontsize=9,
                         color=("#c0392b" if row == 1 and c_new < c_old else "#1e8449"
                                if row == 1 else "#333333"))
        # difference map: where the two runs disagree
        ax = fig.add_subplot(gs[2, col])
        d = np.zeros(gray.shape + (3,))
        d[m_old & ~m_new] = [0.85, 0.2, 0.2]     # lost by the new model
        d[m_new & ~m_old] = [0.2, 0.5, 0.95]     # gained by the new model
        d[m_old & m_new] = [0.30, 0.30, 0.30]
        ax.imshow(d[y0:y1, x0:x1]); ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("difference", fontsize=10, fontweight="bold")
        ax.set_title("red = only OLD   blue = only NEW", fontsize=8, color="#555555")
        # ground truth for reference
        ax = fig.add_subplot(gs[3, col])
        ax.imshow(gt[y0:y1, x0:x1], cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("ground truth", fontsize=10, fontweight="bold")

    # ---- temporal profile: the real story ------------------------------
    ax = fig.add_subplot(gs[4, :])
    import matplotlib.ticker as mtick
    qo = [[] for _ in range(4)]; qn = [[] for _ in range(4)]
    for v in TEST:
        co = np.array([r[2] for r in rows if r[0] == v])
        cn = np.array([r[3] for r in rows if r[0] == v])
        for i, idx in enumerate(np.array_split(np.arange(len(co)), 4)):
            qo[i] += list(co[idx]); qn[i] += list(cn[idx])
    x = np.arange(4)
    mo = [np.mean(q) for q in qo]; mn = [np.mean(q) for q in qn]
    ax.plot(x, mo, "o-", color="#8B96A3", lw=2, ms=8, label=f"OLD (inert cbDice)   mean {pv_old:.4f}")
    ax.plot(x, mn, "o-", color="#1F6FB2", lw=2, ms=8, label=f"NEW (cbDice 0.3 + focal 3, ep84)  mean {pv_new:.4f}")
    for i in range(4):
        d = mn[i] - mo[i]
        ax.annotate(f"{d:+.3f}", (i, max(mo[i], mn[i]) + 0.006), ha="center", fontsize=10,
                    color=("#1e8449" if d > 0 else "#c0392b"), fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["Q1 (early)", "Q2", "Q3", "Q4 (late)"])
    ax.set_xlabel("position within the recording, pooled over the four held-out videos", fontsize=9)
    ax.set_ylabel("clDice", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title("A (inert cbDice, currently in the paper) vs D (cbDice 0.3 + focal 3, epoch 84)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.suptitle("INTERNAL -- A (inert cbDice) vs D (cbDice 0.3): the THREE WORST frames for D",
                 fontsize=13, fontweight="bold", y=0.985)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")
    print(f"  per-video (Table 3): OLD {pv_old:.4f}  NEW {pv_new:.4f}  diff {pv_new-pv_old:+.4f}")
    print(f"  pooled-frame       : OLD {arr_old.mean():.4f}  NEW {arr_new.mean():.4f}  "
          f"diff {diff.mean():+.4f}")
    print(f"  frames shown: {[(p[0], p[1]) for p in picks]}")


if __name__ == "__main__":
    main()
