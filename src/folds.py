"""
Cross-validation folds: three disjoint held-out test sets around one fixed
validation set.

Fold 1 is the split the ISBI draft reports (23 train / 5 val / 4 test). Folds 2
and 3 rerun the identical recipe with a different four-recording test set, so
results can be reported as mean +- SD over three models instead of one.

How the new test sets were chosen (check_folds() enforces the mechanical rules):

  * Only from fold 1's TRAINING recordings. Never from validation: every
    hyperparameter -- ours, and the baselines' input scales -- was tuned on it,
    so a validation recording in a test set would carry that tuning into the
    score. That is also why VAL stays the same in every fold.
  * Disjoint from each other and from fold 1: three folds score 12 recordings.
  * Fold 1's mix: two from the 1.5 px/um cortex plates, two from the
    0.75 px/um seeding-density series.
  * No test recording shares frames with a recording its model trains or
    validates on (SAME_FRAMES: one crop, two target neurons).
  * Each cortex pick is the only recording from its field of view. The density
    series has two such recordings (B11, B12); fold 3 takes the two full-length
    A11 cells and leaves A11_C2/C3 in training -- the same-field situation fold
    1 has with A12.
  * B28_C4_200k is the only 200k recording and trains in every fold, as in 1.
  * Every fold scores exactly 328 frames (2 x 72 + 2 x 92, frame 0 excluded).

Fold 1 itself predates these rules and breaks one: its test recording
A12_C1_100k_2 is the same crop as A12_C1_100k_1, which it trains on.

    python -m src.folds                       # print the folds, check the rules
    python -m src.folds --check-data          # cluster preflight: is training_data/ current?
    python -m src.folds --write-fingerprints  # after editing any GT mask (then commit)

Only the standard library at import time: the baseline runners import this from
their own environments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "data_manifest.json")
FINGERPRINTS = os.path.join(REPO, "src", "data_fingerprints.json")

# Model selection (best val IoU, Betti-0 tie-break) and every tuned setting.
VAL = ["A91_C1_2", "A31_C1", "A52_C2", "A22_C2", "A42_C1"]

FOLDS = {
    1: ["A71_C1", "A81_C1", "A12_C3_100k", "A12_C1_100k_2"],    # the paper's split
    2: ["A51_C1", "A61_C1", "B11_C1_100k", "B12_C1_100k"],
    3: ["A11_C1", "A52_C1_right", "A11_C1_100k", "A11_C5_100k"],
}

# Recordings cut from the same image stack: identical frames, different target.
SAME_FRAMES = [
    ("A91_C1_1", "A91_C1_2"),
    ("A32_C1_1", "A32_C1_2"),
    ("A12_C1_100k_1", "A12_C1_100k_2"),
]


def test_videos(fold: int) -> list[str]:
    if fold not in FOLDS:
        raise ValueError(f"no fold {fold!r}; folds are {sorted(FOLDS)}")
    return list(FOLDS[fold])


def manifest_names() -> list[str]:
    with open(MANIFEST) as f:
        return sorted(e["name"] for e in json.load(f))


def train_videos(fold: int, names: list[str] | None = None) -> list[str]:
    held_out = set(VAL) | set(test_videos(fold))
    return [n for n in (names or manifest_names()) if n not in held_out]


def check_folds(names: list[str] | None = None) -> None:
    """Raise ValueError on the first fold that breaks a rule above."""
    names = set(names or manifest_names())
    unknown = sorted((set(VAL) | {v for t in FOLDS.values() for v in t}) - names)
    if unknown:
        raise ValueError(f"not in the manifest: {unknown}")
    for fold, test in FOLDS.items():
        if len(test) != 4 or len(set(test)) != 4:
            raise ValueError(f"fold {fold}: test set must be 4 distinct recordings")
        if set(test) & set(VAL):
            raise ValueError(f"fold {fold}: tests on validation recordings "
                             f"{sorted(set(test) & set(VAL))}")
        for other, other_test in FOLDS.items():
            if other < fold and set(test) & set(other_test):
                raise ValueError(f"folds {other} and {fold} share "
                                 f"{sorted(set(test) & set(other_test))}")
        if fold == 1:
            continue            # published split, kept as is (see module docstring)
        for pair in SAME_FRAMES:
            if len(set(pair) & set(test)) == 1:
                raise ValueError(f"fold {fold}: {pair} share frames but are "
                                 f"split across test and train/val")


# --- data preflight ---------------------------------------------------------
# training_data/ is not in git: it is rsynced to the cluster by hand, so a GT
# fix made on the laptop can silently never arrive (A71_C1 did exactly that).
# Fingerprints are over DECODED pixels, so a PNG re-encoded by another Pillow
# still matches; only a real pixel change, or a missing frame, does not.

def _fingerprint(directory: str) -> tuple[int, str]:
    import numpy as np
    from PIL import Image
    files = sorted(f for f in os.listdir(directory) if f.endswith(".png"))
    h = hashlib.sha256()
    for f in files:
        a = np.asarray(Image.open(os.path.join(directory, f)))
        h.update(f.encode())
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return len(files), h.hexdigest()


def fingerprint_data(data_dir: str, names: list[str]) -> dict:
    out = {}
    for n in names:
        rec = {}
        for kind in ("images", "masks"):
            d = os.path.join(data_dir, kind, n)
            rec[kind] = list(_fingerprint(d)) if os.path.isdir(d) else None
        out[n] = rec
    return out


def check_data(data_dir: str) -> list[str]:
    """Problems with data_dir against the committed fingerprints ([] = OK)."""
    with open(FINGERPRINTS) as f:
        want = json.load(f)
    names = manifest_names()
    problems = []
    if sorted(want) != names:
        problems.append("src/data_fingerprints.json is out of date with "
                        "data_manifest.json -- rerun --write-fingerprints")
    img_root = os.path.join(data_dir, "images")
    present = sorted(d for d in os.listdir(img_root)
                     if os.path.isdir(os.path.join(img_root, d))) \
        if os.path.isdir(img_root) else []
    extra = sorted(set(present) - set(names))
    if extra:
        # get_video_names() lists directories, not the manifest: an extra
        # directory would quietly join the training set.
        problems.append(f"{img_root}/ has recordings outside the manifest, "
                        f"which training would pick up: {extra}")
    have = fingerprint_data(data_dir, names)
    for n in names:
        for kind in ("images", "masks"):
            got, exp = have[n][kind], want.get(n, {}).get(kind)
            if got is None:
                problems.append(f"{kind}/{n}: missing")
            elif exp is not None and got[0] != exp[0]:
                problems.append(f"{kind}/{n}: {got[0]} frames, expected {exp[0]}")
            elif exp is not None and got[1] != exp[1]:
                problems.append(f"{kind}/{n}: pixels differ from the committed "
                                f"fingerprint (stale copy, or an unsynced edit)")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(REPO, "training_data"))
    ap.add_argument("--check-data", action="store_true")
    ap.add_argument("--write-fingerprints", action="store_true")
    args = ap.parse_args()

    check_folds()
    if args.write_fingerprints:
        fp = fingerprint_data(args.data_dir, manifest_names())
        missing = [f"{k}/{n}" for n, r in fp.items() for k, v in r.items() if v is None]
        if missing:
            print(f"refusing to write: missing {missing}")
            return 1
        with open(FINGERPRINTS, "w") as f:
            json.dump(fp, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"wrote {os.path.relpath(FINGERPRINTS, REPO)} ({len(fp)} recordings)")
        return 0
    if args.check_data:
        problems = check_data(args.data_dir)
        for p in problems:
            print(f"  DATA PROBLEM  {p}")
        if problems:
            print(f"\n{len(problems)} problem(s) in {args.data_dir}. Sync it from "
                  f"the laptop, where it was fingerprinted, before training.")
            return 1
        print(f"training_data OK: all {len(manifest_names())} recordings match "
              f"the committed fingerprints")
        return 0

    names = manifest_names()
    print(f"val ({len(VAL)}): {VAL}")
    for fold in FOLDS:
        tr = train_videos(fold, names)
        print(f"fold {fold}: test {FOLDS[fold]}  |  train {len(tr)}")
    print("fold rules OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
