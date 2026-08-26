#!/usr/bin/env python3
"""INTERNAL -- a REPRESENTATIVE visual comparison of the two runs.

compare_runs.py deliberately shows the three worst frames, which makes the
gap look larger than it typically is. This script instead samples frames at
fixed positions through each recording (10%, 40%, 70%, 95% of its length), so
what you see is the ordinary trajectory rather than the tail.

One row per held-out recording; four columns spanning early -> late.
Each cell shows the OLD run above the NEW one, with the chosen metric for both.

    python benchmark/compare_runs_fair.py
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from benchmark.make_proof_panel import load_gray, load_gt, overlay, bbox  # noqa: E402
from benchmark.eval_masks import cldice_score, cbdice_score  # noqa: E402

TEST = ["A71_C1", "A81_C1", "A12_C3_100k", "A12_C1_100k_2"]
FRACS = [0.10, 0.40, 0.70, 0.95]


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default="ours", help="prediction dir under predictions/")
    ap.add_argument("--new", default="ours_fixed")
    ap.add_argument("--label-old", default="D")
    ap.add_argument("--label-new", default="fixed")
    ap.add_argument("--metric", choices=("cbdice", "cldice"), default="cbdice",
                    help="cbDice is the paper's primary metric")
    ap.add_argument("--videos", nargs="*", default=TEST)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    OLD = os.path.join(REPO, "benchmark", "predictions", args.old)
    NEW = os.path.join(REPO, "benchmark", "predictions", args.new)
    TEST_V = args.videos
    score = cbdice_score if args.metric == "cbdice" else cldice_score
    mlabel = "cbDice" if args.metric == "cbdice" else "clDice"
    OUT = args.out or os.path.join(REPO, "paper_assets", "figures",
                                   f"_internal_{args.label_old}_vs_{args.label_new}_fair.png")
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(len(TEST_V) * 2, len(FRACS), hspace=0.30, wspace=0.05)

    for r, v in enumerate(TEST_V):
        po = tifffile.imread(os.path.join(OLD, f"{v}.tif"))
        pn = tifffile.imread(os.path.join(NEW, f"{v}.tif"))
        gts = sorted(glob.glob(os.path.join(REPO, "training_data", "masks", v, "*.png")))
        T = min(len(po), len(pn))
        for c, f in enumerate(FRACS):
            t = max(1, min(T - 1, int(round(f * (T - 1)))))
            g = np.array(Image.open(gts[t])) > 0
            m_old, m_new = po[t].astype(bool), pn[t].astype(bool)
            gray = load_gray(v, t)
            y0, y1, x0, x1 = bbox([g, m_old, m_new], margin=14)
            c_old = score(m_old, g)
            c_new = score(m_new, g)
            for k, (mask, cl, tag) in enumerate(
                    [(m_old, c_old, args.label_old), (m_new, c_new, args.label_new)]):
                ax = fig.add_subplot(gs[r * 2 + k, c])
                ax.imshow(overlay(gray, g, mask)[y0:y1, x0:x1])
                ax.set_xticks([]); ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(f"{v}\n{tag}" if k == 0 else tag,
                                  fontsize=9, fontweight="bold")
                if k == 0:
                    ax.set_title(f"t = {t} / {T-1}   ({int(f*100)}%)",
                                 fontsize=9, color="#333333")
                col = "#c0392b" if (k == 1 and c_new < c_old - 0.01) else \
                      "#1e8449" if (k == 1 and c_new > c_old + 0.01) else "#555555"
                ax.text(0.02, 0.02, f"{mlabel} {cl:.3f}", transform=ax.transAxes,
                        fontsize=9, color="white", fontweight="bold",
                        bbox=dict(facecolor=col, alpha=0.85, pad=2, edgecolor="none"))

    fig.suptitle(f"INTERNAL -- {args.label_old} (top) vs {args.label_new} (bottom)\n"
                 "representative frames: 10/40/70/95% through each recording\n"
                 "yellow = correct   green = missed   red = extra",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.savefig(OUT, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
