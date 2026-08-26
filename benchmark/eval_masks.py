#!/usr/bin/env python3
"""
eval_masks.py -- segmentation metrics for the neuron benchmark.

Computes IoU, Dice, and clDice (the connectivity-sensitive metric) between a
predicted mask stack and a ground-truth mask stack, per-frame and per-sequence.
clDice is the reported connectivity METRIC that exposes fragmented-neurite failures
IoU/Dice miss (losing a thin branch barely moves IoU/Dice but sharply drops clDice).
Note: the model is TRAINED with cbDice (an EDT-weighted extension of clDice) + a
supervoxel loss; clDice is used only for evaluation, kept independent of the training
objective on purpose. cbDice can be reported as a secondary metric if desired.

Masks may be binary or integer-labelled. By default every non-zero pixel is
foreground (the target neuron); pass a specific label to score one instance.

Usage
-----
    # score two multi-page TIFF stacks (pred vs GT), 2D+time (frames on axis 0)
    python eval_masks.py --pred pred.tif --gt gt.tif

    # score whole directories of per-sequence TIFFs matched by filename stem
    python eval_masks.py --pred preds/ --gt gts/ --out results.csv

    # prove the metric works: connectivity self-test (no data needed)
    python eval_masks.py --selftest

The module is import-friendly: use iou_score / dice_score / cldice_score on 2D
arrays, or evaluate_stack / evaluate_pair on (T,H,W) stacks.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from dataclasses import dataclass, field

import numpy as np
from skimage.morphology import skeletonize
from skimage.transform import resize as sk_resize

EPS = 1e-7
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------
def _binarize(mask: np.ndarray, target_label: int | None = None) -> np.ndarray:
    """Boolean foreground: any non-zero pixel, or exactly `target_label`."""
    arr = np.asarray(mask)
    if target_label is None:
        return arr > 0
    return arr == target_label


def _match_shape(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Nearest-neighbour resize a boolean `pred` onto `gt`'s HxW (order=0)."""
    if pred.shape == gt.shape:
        return pred
    out = sk_resize(pred.astype(np.uint8), gt.shape, order=0,
                    preserve_range=True, anti_aliasing=False)
    return out.astype(bool)


# ---------------------------------------------------------------------------
# 2D metrics (operate on a single frame)
# ---------------------------------------------------------------------------
def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:          # both empty -> perfect agreement
        return 1.0
    return float(inter / union)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(p, g).sum() / denom)


def cldice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Hard clDice (Shit et al., CVPR 2021) on a single 2D frame.

    Tprec = |skel(pred) & gt| / |skel(pred)|   (is the predicted centerline inside GT?)
    Tsens = |skel(gt) & pred| / |skel(gt)|     (is the GT centerline covered by pred?)
    clDice = harmonic mean(Tprec, Tsens).
    """
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() and not g.any():
        return 1.0
    if not p.any() or not g.any():
        return 0.0
    sp = skeletonize(p)
    sg = skeletonize(g)
    sp_sum, sg_sum = sp.sum(), sg.sum()
    if sp_sum == 0 or sg_sum == 0:
        return 0.0
    tprec = np.logical_and(sp, g).sum() / (sp_sum + EPS)
    tsens = np.logical_and(sg, p).sum() / (sg_sum + EPS)
    if tprec + tsens < EPS:
        return 0.0
    return float(2.0 * tprec * tsens / (tprec + tsens))


METRIC_FNS = {"iou": iou_score, "dice": dice_score, "cldice": cldice_score}


# -- secondary metric: cbDice via the project's OWN training loss ------------
# Reported alongside clDice for completeness. Uses src.loss_fns.SoftCBDiceLoss
# so the number is exactly consistent with the training objective (no re-impl).
_CBDICE_LOSS = None


def _exact_skeleton(img, num_iter: int = 5):
    """Exact morphological thinning, as a drop-in for soft_skel when SCORING.

    clDice is already evaluated with skimage's skeletonize; cbDice must be too,
    or the two topology metrics are not measuring centrelines the same way.
    soft_skel is five morphological openings -- it cannot fully thin a thick
    soma, and how badly it under-skeletonises depends on how blobby or
    fragmented the prediction is, so it biases the comparison BETWEEN methods
    (exact-vs-soft on real predictions: Omnipose +0.169, mu-SAM -0.004).
    The loss keeps soft_skel (reference behaviour); only scoring changes.
    """
    import torch
    a = img.detach().cpu().numpy()
    out = np.zeros_like(a)
    flat, o = a.reshape(-1, *a.shape[-2:]), out.reshape(-1, *a.shape[-2:])
    for i in range(flat.shape[0]):
        o[i] = skeletonize(flat[i] > 0.5).astype(a.dtype)
    return torch.from_numpy(out).to(img.device)


def _get_cbdice_loss(num_iter: int = 5, topology_resolution: int = 512):
    """topology_resolution MUST exceed the native mask size (max 485 px here).

    SoftCBDiceLoss downsamples to a square topo_res x topo_res whenever either
    side exceeds it. At the training default of 256 that squashes a 307x485
    mask non-uniformly and thins away 1-2 px neurites -- the very structure
    cbDice exists to measure. Measured consequence: cbDice(gt, gt) came out at
    0.90-1.00 instead of 1.0, i.e. the metric was not on a 0-1 scale. At 512
    no resize happens and cbDice(gt, gt) == 1.0000 on every frame tested.
    Downsampling is fine for the TRAINING loss (speed, and only gradients
    matter); it is not acceptable for a reported metric.
    """
    global _CBDICE_LOSS
    if _CBDICE_LOSS is None:
        import sys
        sys.path.insert(0, REPO_ROOT)
        from src.loss_fns import SoftCBDiceLoss
        _CBDICE_LOSS = SoftCBDiceLoss(num_iter=num_iter,
                                      topology_resolution=topology_resolution,
                                      skeleton_fn=_exact_skeleton).eval()
    return _CBDICE_LOSS


def cbdice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Secondary connectivity metric = cbDice value (1 - SoftCBDiceLoss) on hard
    masks. Requires torch + src.loss_fns. Empty-mask handling matches clDice."""
    p, g = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    if not p.any() and not g.any():
        return 1.0
    if not p.any() or not g.any():
        return 0.0
    import torch
    loss_fn = _get_cbdice_loss()
    with torch.no_grad():
        pl = torch.from_numpy((p.astype(np.float32) * 20.0) - 10.0)[None, None]  # logits [1,1,H,W]
        tg = torch.from_numpy(g.astype(np.float32))[None, None]
        loss = float(loss_fn(pl, tg))
    return 1.0 - loss


# ---------------------------------------------------------------------------
# stack-level evaluation (T, H, W)
# ---------------------------------------------------------------------------
@dataclass
class SeqResult:
    name: str
    per_frame: dict[str, list[float]] = field(default_factory=dict)

    def mean(self, metric: str) -> float:
        vals = self.per_frame.get(metric, [])
        return float(np.mean(vals)) if vals else float("nan")

    def std(self, metric: str) -> float:
        vals = self.per_frame.get(metric, [])
        return float(np.std(vals)) if vals else float("nan")


def evaluate_stack(pred: np.ndarray, gt: np.ndarray, name: str = "seq",
                   target_label: int | None = None,
                   skip_first_frame: bool = False,
                   with_cbdice: bool = False) -> SeqResult:
    """Per-frame IoU/Dice/clDice (and optionally cbDice) over a 2D+time stack.

    `skip_first_frame` excludes t=0 (the prompt frame, which trivially matches
    GT for a promptable model) from the reported per-frame lists.
    `with_cbdice` adds the secondary cbDice metric (needs torch + src.loss_fns).
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.ndim == 2:
        pred, gt = pred[None], gt[None]
    T = min(pred.shape[0], gt.shape[0])
    keys = list(METRIC_FNS) + (["cbdice"] if with_cbdice else [])
    res = SeqResult(name=name, per_frame={k: [] for k in keys})
    start = 1 if skip_first_frame else 0
    for t in range(start, T):
        gt_t = _binarize(gt[t], target_label)
        pred_t = _match_shape(_binarize(pred[t], target_label), gt_t)
        for m, fn in METRIC_FNS.items():
            res.per_frame[m].append(fn(pred_t, gt_t))
        if with_cbdice:
            res.per_frame["cbdice"].append(cbdice_score(pred_t, gt_t))
    return res


def evaluate_pair(pred_path: str, gt_path: str, **kw) -> SeqResult:
    import tifffile
    pred = tifffile.imread(pred_path)
    gt = tifffile.imread(gt_path)
    name = os.path.splitext(os.path.basename(gt_path))[0]
    return evaluate_stack(pred, gt, name=name, **kw)


def aggregate(results: list[SeqResult]) -> dict[str, dict[str, float]]:
    """Mean +/- std across sequences (each sequence weighted equally)."""
    keys: list[str] = []
    for r in results:
        for k in r.per_frame:
            if k not in keys:
                keys.append(k)
    out = {}
    for m in keys:
        seq_means = [r.mean(m) for r in results if not np.isnan(r.mean(m))]
        out[m] = {
            "mean": float(np.mean(seq_means)) if seq_means else float("nan"),
            "std": float(np.std(seq_means)) if seq_means else float("nan"),
            "n": len(seq_means),
        }
    return out


# ---------------------------------------------------------------------------
# connectivity self-test -- the paper's thesis as an assertion
# ---------------------------------------------------------------------------
def _synthetic_neuron() -> np.ndarray:
    """A soma with a main process and two thin branches -- a toy neuron.

    Thin branches are little AREA but a large fraction of the SKELETON, which is
    exactly why dropping them barely moves IoU/Dice yet sharply drops clDice.
    """
    H, W = 140, 240
    gt = np.zeros((H, W), bool)
    yy, xx = np.ogrid[:H, :W]
    gt |= (yy - 70) ** 2 + (xx - 34) ** 2 <= 15 ** 2   # soma disk, r=15 (~700 px)
    gt[68:71, 34:210] = True                            # main process, 3 px x ~176
    for i in range(55):                                 # upper thin branch, 1 px
        gt[69 - i, 120 + i] = True
    for i in range(50):                                 # lower thin branch, 1 px
        gt[69 + i, 160 + i] = True
    return gt


def selftest() -> None:
    """Why clDice belongs in the benchmark: it catches lost thin neurites that
    IoU/Dice shrug off, and (honestly) it is only mildly moved by a cosmetic gap.
    """
    gt = _synthetic_neuron()

    db = gt.copy()                                       # a generalist's failure
    for i in range(55):
        db[69 - i, 120 + i] = False                     # erase upper branch
    for i in range(50):
        db[69 + i, 160 + i] = False                     # erase lower branch

    missing_tip = gt.copy()
    missing_tip[:, 195:210] = False                      # drop distal 15 px of main

    small_gap = gt.copy()
    small_gap[:, 100:102] = False                        # cosmetic 2-px gap

    cases = [("identical", gt),
             ("dropped 2 thin branches", db),
             ("missing distal tip", missing_tip),
             ("cosmetic 2-px gap", small_gap)]

    print("\nclDice self-test  (GT = toy neuron: soma + main process + 2 thin branches)")
    print(f"{'prediction':<26}{'IoU':>8}{'Dice':>8}{'clDice':>9}")
    print("-" * 51)
    for label, pred in cases:
        print(f"{label:<26}{iou_score(pred, gt):>8.3f}"
              f"{dice_score(pred, gt):>8.3f}{cldice_score(pred, gt):>9.3f}")

    iou_drop = 1.0 - iou_score(db, gt)
    cld_drop = 1.0 - cldice_score(db, gt)
    print("-" * 51)
    print(f"dropping 2 thin branches:  IoU -{iou_drop:.3f}   clDice -{cld_drop:.3f}   "
          f"(clDice reacts {cld_drop / max(iou_drop, EPS):.1f}x harder)")
    assert cld_drop > 1.8 * iou_drop, "clDice should react far harder to lost thin branches than IoU"
    print("PASS: clDice exposes lost thin neurites that IoU/Dice under-penalise.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _collect_pairs(pred: str, gt: str) -> list[tuple[str, str]]:
    """Match pred/GT: either two files, or two dirs paired by filename stem."""
    if os.path.isfile(pred) and os.path.isfile(gt):
        return [(pred, gt)]
    pred_files = {os.path.splitext(os.path.basename(f))[0].replace("_masks", ""): f
                  for f in glob.glob(os.path.join(pred, "*.tif*"))}
    pairs = []
    for gf in sorted(glob.glob(os.path.join(gt, "*.tif*"))):
        stem = os.path.splitext(os.path.basename(gf))[0].replace("_masks", "")
        if stem in pred_files:
            pairs.append((pred_files[stem], gf))
        else:
            print(f"  [warn] no prediction for GT sequence '{stem}'")
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", help="predicted mask .tif or directory of .tif")
    ap.add_argument("--gt", help="ground-truth mask .tif or directory of .tif")
    ap.add_argument("--target-label", type=int, default=None,
                    help="score only this GT label (default: any non-zero pixel)")
    ap.add_argument("--skip-first-frame", action="store_true",
                    help="exclude t=0 (prompt frame) from per-frame scores")
    ap.add_argument("--out", default=None, help="write per-sequence CSV here")
    ap.add_argument("--cbdice", action="store_true",
                    help="also compute cbDice (secondary metric via src.loss_fns.SoftCBDiceLoss)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the connectivity demonstration and exit")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.pred or not args.gt:
        ap.error("--pred and --gt are required (or use --selftest)")

    pairs = _collect_pairs(args.pred, args.gt)
    if not pairs:
        ap.error("no matching pred/GT pairs found")

    results = [evaluate_pair(p, g, target_label=args.target_label,
                             skip_first_frame=args.skip_first_frame,
                             with_cbdice=args.cbdice)
               for p, g in pairs]

    metrics = list(METRIC_FNS) + (["cbdice"] if args.cbdice else [])
    labels = {"iou": "IoU", "dice": "Dice", "cldice": "clDice", "cbdice": "cbDice"}
    header = f"{'sequence':<18}" + "".join(f"{labels[m]:>9}" for m in metrics) + "   frames"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        n = len(r.per_frame["iou"])
        print(f"{r.name:<18}" + "".join(f"{r.mean(m):>9.4f}" for m in metrics) + f"   {n}")
    print("-" * len(header))
    agg = aggregate(results)
    print(f"{'MEAN':<18}" + "".join(f"{agg[m]['mean']:>9.4f}" for m in metrics))
    print(f"{'STD':<18}" + "".join(f"{agg[m]['std']:>9.4f}" for m in metrics)
          + f"   (n={agg[metrics[0]]['n']} sequences)")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence"] + [f"{m}_mean" for m in metrics]
                       + [f"{m}_std" for m in metrics] + ["n_frames"])
            for r in results:
                w.writerow([r.name] + [r.mean(m) for m in metrics]
                           + [r.std(m) for m in metrics] + [len(r.per_frame["iou"])])
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
