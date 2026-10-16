#!/usr/bin/env python
"""Build the propagation filmstrip: real phase-contrast frames with the frame-0
ground-truth prompt and the model's own propagated masks overlaid.

Frame 0 carries the ground-truth mask -- that is the prompt the model is given.
Every later frame carries the PREDICTION of the model named by --pred-dir.
Nothing here is ground truth except frame 0, so the panel shows what the model
actually produces.

Before drawing, the script re-scores the stack against the ground truth and
refuses to write the figure if the mean IoU does not match the number recorded
for that run -- the guard against silently plotting a superseded stack. It also
refuses outright to draw a recording the model did not hold out, read from the
prediction directory's own training_config.json.

    venv/bin/python paper_assets/paper/make_filmstrip.py                 # A71_C1, fold 1
    venv/bin/python paper_assets/paper/make_filmstrip.py --video B12_C1_100k

Recording-specific geometry lives in RECORDINGS; --crop, --frames, --px-per-um,
--erase-band and --out override it.
"""
import argparse
import csv
import os
import sys

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage
from skimage import measure

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
PRED_ROOT = os.path.join(REPO, "benchmark", "predictions")
RESULTS = os.path.join(REPO, "benchmark", "results")
sys.path.insert(0, os.path.join(REPO, "benchmark"))
import held_out  # noqa: E402

CELL_W, CELL_H = 400, 400        # px per frame in the output strip (A71 default)
GAP_RATIO = 342 / 400            # gap between frames, as a fraction of cell width
NAVY = np.array([22, 50, 79])    # the frame-0 prompt
TEAL = np.array([44, 166, 164])  # the model's own output
SCALE_BAR_UM = 50

# Per recording: the frames shown (12 h apart), the crop, the calibration, and
# the band holding IncuCyte's burnt-in scale bar. The bar is bright-on-dark in
# both recordings but sits in a different place, and it moves a little from
# frame to frame because the movies are registered, so the band is generous.
RECORDINGS = {
    # Rotated during registration (black corners) and carrying a burnt-in scale
    # bar and "0.79 x 0.58 mm" caption near the bottom. Rows/cols 22..250 are
    # free of both across all seven frames and hold the whole target.
    "A71_C1": dict(
        frames=[0, 12, 24, 36, 48, 60, 72],
        crop=(22, 250, 22, 250),              # y0, y1, x0, x1
        px_per_um=1.497,                      # 292 px = 195 um
        erase_band=(200, 270, 0, None),
        shared_levels=False,                  # crop holds no padding: stretch per frame
        pred_dir="ours_23rec_topo",
        scores=os.path.join(RESULTS, "test", "ours_23rec_topo.csv"),
        out=os.path.join(HERE, "fig2_filmstrip.png"),
    ),
    # Fold 2's test recording closest to that fold's mean IoU. The neuron grows
    # straight down the frame, so the crop is portrait; the burnt-in "400 um"
    # bar sits bottom left and is erased as in A71. Registration moves the
    # imaged field by up to 20 px between frames, so a crop holding the whole
    # target (rows 82..350) cannot avoid the rotated black corner entirely:
    # frames 36 and 72 keep a ~6 px sliver along the top edge.
    "B12_C1_100k": dict(
        frames=[0, 12, 24, 36, 48, 60, 72, 84],
        crop=(82, 357, 21, 262),
        px_per_um=0.75,                       # 300 px = 400 um (data_manifest.json)
        erase_band=(265, 350, 15, 100),
        shared_levels=True,                   # see levels(): padding inside the crop
        pred_dir="ours_fold2_topo",
        scores=os.path.join(RESULTS, "folds", "per_recording.csv"),
        out=os.path.join(REPO, "paper_assets", "isbi2027", "figures",
                         "fig2_filmstrip.png"),
    ),
}


def gt(video, frame):
    m = np.array(Image.open(os.path.join(REPO, "training_data", "masks", video,
                                         f"{frame:05d}.png")))
    if m.ndim == 3:
        m = m[..., 0]
    labels = [v for v in np.unique(m) if v != 0]
    return m == max(labels, key=lambda v: (m == v).sum())


def recorded_iou(scores, video):
    """The mean IoU this run is reported with, from either results layout."""
    with open(scores) as f:
        rows = list(csv.DictReader(f))
    if rows and "sequence" in rows[0]:                     # results/test/<run>.csv
        for row in rows:
            if row["sequence"] == video:
                return float(row["iou_mean"])
    else:                                                  # results/folds/per_recording.csv
        for row in rows:
            if row["method"] == "NeuroVestigo" and row["recording"] == video:
                return float(row["iou"])
    sys.exit(f"no row for {video} in {scores}")


def check_stack(stack, video, scores):
    """Refuse to plot a stack that is not the run the paper reports."""
    ious = []
    for t in range(1, stack.shape[0]):
        p, g = stack[t] > 0, gt(video, t)
        inter, union = (p & g).sum(), (p | g).sum()
        ious.append(inter / union if union else 1.0)
    got, want = float(np.mean(ious)), recorded_iou(scores, video)
    if abs(got - want) > 5e-4:
        sys.exit(f"MISMATCH: recomputed IoU {got:.4f} but {scores} records "
                 f"{want:.4f}. Refusing to draw a stack that is not the "
                 f"reported run.")
    print(f"  verified: mean IoU over t>=1 is {got:.4f}, matches "
          f"{os.path.basename(scores)} ({want:.4f})")
    return got


def levels(raw):
    """Display range of one frame, over imaged pixels only.

    Registration rotates the field, so part of some frames inside the crop is
    zero padding. Letting those zeros into the percentile drags the low end to
    0 and washes that frame out -- next to its neighbours in the strip it then
    looks like a different exposure. Recordings whose crop is free of padding
    (A71_C1) keep the original per-frame stretch over the crop itself.
    """
    valid = raw[raw > 0]
    return np.percentile(valid if valid.size else raw, (1.0, 99.0))


def strip_overlay(im, keep, band_box, levels_=None):
    """Erase IncuCyte's burnt-in scale bar and caption.

    Only very bright pixels inside the band are touched, and never a pixel
    belonging to the drawn mask (`keep`), so no part of the neuron is altered.
    Erased pixels are refilled from a median of their neighbourhood, i.e. with
    the local background texture.
    """
    y0, y1, x0, x1 = band_box
    band = np.zeros(im.shape, bool)
    band[y0:y1, x0:x1] = True
    for _ in range(2):
        lo, hi = levels_ if levels_ is not None else np.percentile(im, (1.0, 99.0))
        norm = np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1)
        ov = ndimage.binary_dilation((norm > 0.90) & band, np.ones((3, 3)),
                                     iterations=2) & ~keep
        if not ov.any():
            break
        im = np.where(ov, ndimage.median_filter(im, size=11), im)
    return im


def load_raw(video, frame):
    return np.array(Image.open(os.path.join(REPO, "training_data", "images", video,
                                            f"{frame:05d}.png")).convert("L")).astype(float)


def cell(video, frame, binary, color, cfg, size, display):
    """One frame: contrast-stretched phase contrast + translucent mask + outline."""
    y0, y1, x0, x1 = cfg["crop"]
    raw = load_raw(video, frame)
    im = strip_overlay(raw, binary, cfg["erase_band"], display)[y0:y1, x0:x1]
    binary = binary[y0:y1, x0:x1]
    lo, hi = display if display is not None else np.percentile(im, (1.0, 99.0))
    im = np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1)
    w, h = size
    rgb = np.array(Image.fromarray((np.dstack([im] * 3) * 255).astype(np.uint8))
                   .resize((w, h), Image.LANCZOS)).astype(float)

    big = np.array(Image.fromarray(binary.astype(np.uint8) * 255)
                   .resize((w, h), Image.NEAREST)) > 127
    rgb[big] = 0.62 * rgb[big] + 0.38 * color            # translucent fill

    canvas = Image.fromarray(rgb.astype(np.uint8))
    from PIL import ImageDraw
    d = ImageDraw.Draw(canvas)
    sy, sx = h / binary.shape[0], w / binary.shape[1]
    for c in measure.find_contours(binary.astype(float), 0.5):
        pts = [(x * sx, y * sy) for y, x in c]
        if len(pts) > 2:
            d.line(pts + [pts[0]], fill=tuple(int(v) for v in color), width=2)
    d.rectangle([0, 0, w - 1, h - 1], outline=(150, 150, 150), width=2)
    return np.array(canvas)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default="A71_C1", choices=sorted(RECORDINGS))
    ap.add_argument("--frames", type=int, nargs="+")
    ap.add_argument("--crop", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"))
    ap.add_argument("--px-per-um", type=float)
    ap.add_argument("--erase-band", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"))
    ap.add_argument("--pred-dir")
    ap.add_argument("--out")
    args = ap.parse_args()

    cfg = dict(RECORDINGS[args.video])
    for key in ("frames", "crop", "px_per_um", "erase_band", "pred_dir", "out"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    video = args.video

    held_out.require(video, cfg["pred_dir"])
    pred = os.path.join(PRED_ROOT, cfg["pred_dir"], f"{video}.tif")
    print(f"reading {pred}")
    stack = tifffile.imread(pred)
    print(f"  stack {stack.shape}, dtype {stack.dtype}")
    check_stack(stack, video, cfg["scores"])

    y0, y1, x0, x1 = cfg["crop"]
    size = (CELL_W, int(round(CELL_W * (y1 - y0) / (x1 - x0))))
    # One display range for the whole strip, so the frames are comparable.
    display = (np.median([levels(load_raw(video, f)) for f in cfg["frames"]], axis=0)
               if cfg["shared_levels"] else None)
    cells = [cell(video, f, gt(video, 0) if f == 0 else stack[f] > 0,
                  NAVY if f == 0 else TEAL, cfg, size, display)
             for f in cfg["frames"]]

    bar = int(round(SCALE_BAR_UM * cfg["px_per_um"] * size[0] / (x1 - x0)))
    last = cells[-1]
    ytop, xend = size[1] - 34, size[0] - 26
    last[ytop:ytop + 7, xend - bar:xend] = np.array([20, 20, 20])

    n, gap = len(cells), int(round(CELL_W * GAP_RATIO))
    strip = np.full((size[1], n * size[0] + (n - 1) * gap, 3), 255, dtype=np.uint8)
    for i, c in enumerate(cells):
        strip[:, i * (size[0] + gap):i * (size[0] + gap) + size[0]] = c
    os.makedirs(os.path.dirname(cfg["out"]), exist_ok=True)
    Image.fromarray(strip).save(cfg["out"], optimize=True)
    print(f"wrote {cfg['out']}  {strip.shape[1]}x{strip.shape[0]}  "
          f"{os.path.getsize(cfg['out']) / 1e6:.2f} MB")
    print(f"  {len(cells)} frames {cfg['frames']}, cell {size[0]}x{size[1]} px, "
          f"scale bar {bar} px = {SCALE_BAR_UM} um")


if __name__ == "__main__":
    main()
