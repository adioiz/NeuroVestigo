#!/usr/bin/env python
"""Build the Figure 2(a) filmstrip: real phase-contrast frames of A71_C1 with
the frame-0 ground-truth prompt and the model's own propagated masks overlaid.

Frame 0 carries the ground-truth mask -- that is the prompt the model is given.
Every later frame carries the PREDICTION of the shipped 16-frame model, read
from benchmark/predictions/ours_16f/.  Nothing here is ground truth except
frame 0, so the panel shows what the model actually produces.

Before drawing, the script re-scores the stack against the ground truth and
refuses to write the figure if the mean IoU does not match the number recorded
in benchmark/results/test/ours_16f.csv -- that is the guard against silently
plotting a superseded run.
"""
import csv
import os
import sys

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage
from skimage import measure

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEQ = "A71_C1"
FRAMES = [0, 12, 24, 36, 48, 60, 72]
PRED = os.path.join(REPO, "benchmark", "predictions", "ours_16f", f"{SEQ}.tif")
SCORES = os.path.join(REPO, "benchmark", "results", "test", "ours_16f.csv")
IMAGES = os.path.join(REPO, "training_data", "images", SEQ)
MASKS = os.path.join(REPO, "training_data", "masks", SEQ)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig2_filmstrip.png")

# The raw frames are rotated during registration (black corners) and carry a
# burnt-in IncuCyte scale bar and "0.79 x 0.58 mm" caption near the bottom.
# Rows/cols 22..250 are free of both across all seven frames and contain the
# whole target (rows 47..223, cols 27..246).
CROP = (22, 250)
CELL = 400                       # px per frame in the output strip
GAP = 342                        # px between frames (matches the TikZ spacing)
NAVY = np.array([22, 50, 79])    # the frame-0 prompt
TEAL = np.array([44, 166, 164])  # the model's own output
PX_PER_UM = 1.497                # 292 px = 195 um


def gt(frame):
    m = np.array(Image.open(os.path.join(MASKS, f"{frame:05d}.png")))
    if m.ndim == 3:
        m = m[..., 0]
    labels = [v for v in np.unique(m) if v != 0]
    return m == max(labels, key=lambda v: (m == v).sum())


def check_stack(stack):
    """Refuse to plot a stack that is not the run the paper reports."""
    ious = []
    for t in range(1, stack.shape[0]):
        p, g = stack[t] > 0, gt(t)
        inter, union = (p & g).sum(), (p | g).sum()
        ious.append(inter / union if union else 1.0)
    got = float(np.mean(ious))
    want = None
    for row in csv.DictReader(open(SCORES)):
        if row["sequence"] == SEQ:
            want = float(row["iou_mean"])
    if want is None:
        sys.exit(f"no row for {SEQ} in {SCORES}")
    if abs(got - want) > 5e-4:
        sys.exit(f"MISMATCH: recomputed IoU {got:.4f} but {SCORES} records "
                 f"{want:.4f}. Refusing to draw a stack that is not the "
                 f"reported run.")
    print(f"  verified: mean IoU over t>=1 is {got:.4f}, matches "
          f"ours_16f.csv ({want:.4f})")
    return got


def strip_overlay(im, keep):
    """Erase the IncuCyte scale bar and caption burnt into rows ~200-270.

    Only very bright pixels inside that band are touched, and never a pixel
    belonging to the drawn mask (`keep`), so no part of the neuron is altered;
    across the seven frames the overlay and the mask overlap by 44 px, all of
    them in the dilation halo.  Erased pixels are refilled from a median of
    their neighbourhood, i.e. with the local background texture.
    """
    band = np.zeros(im.shape, bool)
    band[200:270, :] = True
    for _ in range(2):
        lo, hi = np.percentile(im, (1.0, 99.0))
        norm = np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1)
        ov = ndimage.binary_dilation((norm > 0.90) & band, np.ones((3, 3)),
                                     iterations=2) & ~keep
        if not ov.any():
            break
        im = np.where(ov, ndimage.median_filter(im, size=11), im)
    return im


def crop(a):
    lo, hi = CROP
    return a[lo:hi, lo:hi]


def cell(frame, binary, color):
    """One frame: contrast-stretched phase contrast + translucent mask + outline."""
    raw = np.array(Image.open(os.path.join(IMAGES, f"{frame:05d}.png"))
                   .convert("L")).astype(float)
    im = crop(strip_overlay(raw, binary))
    binary = crop(binary)
    lo, hi = np.percentile(im, (1.0, 99.0))
    im = np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.dstack([im] * 3)
    rgb = Image.fromarray((rgb * 255).astype(np.uint8)).resize(
        (CELL, CELL), Image.LANCZOS)
    rgb = np.array(rgb).astype(float)

    s = CELL / binary.shape[0]
    big = np.array(Image.fromarray(binary.astype(np.uint8) * 255).resize(
        (CELL, CELL), Image.NEAREST)) > 127
    rgb[big] = 0.62 * rgb[big] + 0.38 * color            # translucent fill

    canvas = Image.fromarray(rgb.astype(np.uint8))
    from PIL import ImageDraw
    d = ImageDraw.Draw(canvas)
    for c in measure.find_contours(binary.astype(float), 0.5):
        pts = [(x * s, y * s) for y, x in c]
        if len(pts) > 2:
            d.line(pts + [pts[0]], fill=tuple(int(v) for v in color), width=2)
    d.rectangle([0, 0, CELL - 1, CELL - 1], outline=(150, 150, 150), width=2)
    return np.array(canvas)


def main():
    print(f"reading {PRED}")
    stack = tifffile.imread(PRED)
    print(f"  stack {stack.shape}, dtype {stack.dtype}")
    check_stack(stack)

    cells = []
    for f in FRAMES:
        if f == 0:
            cells.append(cell(f, gt(0), NAVY))          # the given prompt
        else:
            cells.append(cell(f, stack[f] > 0, TEAL))   # the model's output

    # 50 um scale bar, burnt into the last frame
    bar = int(round(50 * PX_PER_UM * CELL / (CROP[1] - CROP[0])))
    last = cells[-1]
    y0, x1 = CELL - 34, CELL - 26
    last[y0:y0 + 7, x1 - bar:x1] = np.array([20, 20, 20])

    strip = np.full((CELL, len(FRAMES) * CELL + (len(FRAMES) - 1) * GAP, 3),
                    255, dtype=np.uint8)
    for i, c in enumerate(cells):
        x = i * (CELL + GAP)
        strip[:, x:x + CELL] = c
    Image.fromarray(strip).save(OUT, optimize=True)
    print(f"wrote {OUT}  {strip.shape[1]}x{strip.shape[0]}  "
          f"{os.path.getsize(OUT) / 1e6:.2f} MB")
    print(f"  scale bar {bar} px = 50 um")


if __name__ == "__main__":
    main()
