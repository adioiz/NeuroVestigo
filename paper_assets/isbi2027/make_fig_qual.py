#!/usr/bin/env python3
"""
make_fig_qual.py -- Fig. 2 of the ISBI 2027 paper.

Runs benchmark/make_proof_panel.py unchanged in what it draws (the same
predictions, overlay colors, crop and IoU labels), with four presentation
changes for a 7-inch-wide, two-column page:
  * the figure canvas is scaled by SCALE, so every label is 1/SCALE times
    larger relative to the panels (at 0.55 the column titles print at ~6.5 pt
    instead of ~3.3 pt);
  * shorter column titles, so they fit the narrower panels;
  * no suptitle (the LaTeX caption carries it);
  * output goes to paper_assets/isbi2027/figures/, never over the long
    paper's proof_<video>.png.

    venv/bin/python paper_assets/isbi2027/make_fig_qual.py            # A71_C1, the paper's Fig. 2
    venv/bin/python paper_assets/isbi2027/make_fig_qual.py B12_C1_100k
"""
import os
import sys

import matplotlib.figure
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "benchmark"))
import make_proof_panel as mpp  # noqa: E402

PRESETS = {
    # The paper's Fig. 2: run 1's test recording, drawn with run 1's model (the
    # released one), at the draft's frames 8 and 62.
    "A71_C1": dict(frames=["8", "62"], ours="ours_23rec_topo", scale=0.55,
                   min_aspect=0.0, sam2="SAM 2 (zero-shot)"),
    # Run 2's test recording whose NeuroVestigo IoU is closest to that fold's
    # mean (0.711 against 0.724), so the figure shows typical rather than
    # best-case performance; A61_C1, the fold's best at 0.800, would be
    # cherry-picking. Frames keep the draft's relative positions, 8/72 and 62/72
    # of the sequence. B12's neuron grows down the frame, so the tight crop is
    # portrait; widening it with background keeps nine columns to a figure
    # height the page can hold.
    "B12_C1_100k": dict(frames=["10", "79"], ours="ours_fold2_topo", scale=0.64,
                        min_aspect=0.75, sam2="SAM 2"),
}
VIDEO = sys.argv[1] if len(sys.argv) > 1 else "A71_C1"
P = PRESETS[VIDEO]
SCALE, MIN_ASPECT, FRAMES = P["scale"], P["min_aspect"], P["frames"]

mpp.FIGDIR = os.path.join(HERE, "figures")
mpp.PRETTY.update({"usam": "µSAM (box)", "sam3_maskfirst": "SAM 3 (mask)",
                   "sam2_zeroshot": P["sam2"]})
# make_proof_panel refuses to draw a recording this model trained on.
mpp.PRED_DIR["ours"] = P["ours"]

_subplots = plt.subplots
def _scaled_subplots(*a, figsize=None, **kw):
    return _subplots(*a, figsize=(figsize[0] * SCALE, figsize[1] * SCALE), **kw)
plt.subplots = _scaled_subplots
matplotlib.figure.Figure.suptitle = lambda self, *a, **kw: None

sys.argv = [sys.argv[0], "--video", VIDEO, "--frames", *FRAMES,
            "--min-aspect", str(MIN_ASPECT), "--suffix", "_isbi"]
mpp.main()
