"""One check, shared by the figure scripts: was this recording held out?

A qualitative figure may only show our prediction on a recording the model that
produced it never trained or validated on. With three cross-validation folds
(src/folds.py) the prediction directories differ in exactly that, and the
directory name is the only thing distinguishing them -- easy to point at the
wrong one.

The trainer writes training_config.json into its run directory at the start of
training, so for anything trained since then the split is read from the run
itself rather than from whatever the fold table says today. Directories that
predate that (the fold-1 runs, ours_23rec_topo among them) fall back to fold 1
in src/folds.py, which is the split every one of them used; the fallback says
so out loud, because it is an assumption about the directory rather than a
record from the run.
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from src.folds import FOLDS, VAL                                    # noqa: E402

PRED = os.path.join(REPO, "benchmark", "predictions")


def split_of(pred_dir):
    """(test, val, source) for a prediction directory."""
    cfg = os.path.join(PRED, pred_dir, "training_config.json")
    if os.path.exists(cfg):
        with open(cfg) as f:
            data = json.load(f)["data"]
        return set(data["test_videos"]), set(data["val_videos"]), "training_config.json"
    return set(FOLDS[1]), set(VAL), "no training_config.json: fold 1 assumed"


def require(videos, pred_dir):
    """Exit unless every recording was held out of the run in `pred_dir`."""
    if isinstance(videos, str):
        videos = [videos]
    test, val, source = split_of(pred_dir)
    if source.startswith("no training_config"):
        print(f"  !! {pred_dir} predates training_config.json; assuming it is a "
              f"fold-1 run (test {sorted(test)})")
    seen = sorted(set(videos) - test)
    if seen:
        role = ("validation" if set(seen) <= val else "training or validation")
        raise SystemExit(
            f"{', '.join(seen)} is {role} data for {pred_dir} (it held out "
            f"{sorted(test)}, per {source}). Refusing to draw a prediction on a "
            f"recording the model learned from.")
