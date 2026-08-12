"""Regression tests: augmentation must keep image and label aligned.

The elastic warp used to deform the image with a freshly-sampled random field
while the mask used a pre-computed one, so the label no longer matched the
image (measured agreement only IoU ~0.67) and every frame in a clip got a
different warp. Both are training-data corruption that no loss function can
compensate for, so they are asserted here.

Run: ``KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. pytest tests/test_augmentation_alignment.py``
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.augmentations import (  # noqa: E402
    VideoAugmentation,
    elastic_transform,
    elastic_transform_mask,
    make_elastic_field,
)


def _synthetic_clip(n=4, h=96, w=128):
    """A clip whose 'image' is the mask itself, so misalignment is measurable.

    Using the label as the image content means a correctly-aligned pipeline
    must return frame == mask (up to interpolation); any divergence is
    geometric misalignment rather than appearance.
    """
    m = np.zeros((h, w), np.float32)
    m[h // 2 - 8:h // 2 + 8, w // 2 - 8:w // 2 + 8] = 1.0   # soma
    m[h // 2 - 1:h // 2 + 2, w // 2:int(w * 0.9)] = 1.0     # thin neurite
    frames = [torch.from_numpy(m).unsqueeze(0).repeat(3, 1, 1) for _ in range(n)]
    masks = [torch.from_numpy(m).long() for _ in range(n)]
    return frames, masks


def test_elastic_uses_same_field_for_image_and_mask():
    """Same field in -> image and label land on the same geometry."""
    rng = np.random.RandomState(0)
    m = np.zeros((96, 128), np.float32)
    m[40:56, 40:110] = 1.0
    dx, dy = make_elastic_field(m.shape, alpha=50.0, sigma=5.0)

    warped_img = elastic_transform(m, 50.0, 5.0, dx, dy) > 0.5
    warped_msk = elastic_transform_mask(m, 50.0, 5.0, dx, dy) > 0.5

    inter = np.logical_and(warped_img, warped_msk).sum()
    union = np.logical_or(warped_img, warped_msk).sum()
    iou = inter / union if union else 1.0
    # order=1 vs order=0 differ only on boundary pixels; geometry must match.
    assert iou > 0.97, f"image/mask agreement only IoU={iou:.3f} — fields differ"


def test_elastic_field_is_deterministic_given_field():
    """Repeated calls with the same field must be identical (no hidden RNG)."""
    m = np.zeros((64, 64), np.float32)
    m[20:40, 20:40] = 1.0
    dx, dy = make_elastic_field(m.shape, alpha=50.0, sigma=5.0)
    a = elastic_transform(m, 50.0, 5.0, dx, dy)
    b = elastic_transform(m, 50.0, 5.0, dx, dy)
    assert np.array_equal(a, b), "elastic_transform re-sampled its own field"


def test_clip_stays_aligned_and_temporally_consistent():
    """End-to-end: with elastic forced on, every frame must match its mask
    and all frames must receive the SAME spatial warp."""
    frames, masks = _synthetic_clip()
    aug = VideoAugmentation(
        image_size=128,
        # Isolate elastic: no flips/affine/intensity randomness.
        horizontal_flip_p=0.0, vertical_flip_p=0.0,
        rotation_degrees=0.0, shear=0.0,
        brightness=0.0, contrast=0.0,
        scale_limit=(1.0, 1.0),
        gaussian_blur_p=0.0, gaussian_noise_p=0.0,
        elastic_transform_p=1.0,          # always fire
        normalize_mean=[0.0, 0.0, 0.0], normalize_std=[1.0, 1.0, 1.0],
    )
    out_frames, out_masks = aug(frames, masks)

    ious = []
    for f, mk in zip(out_frames, out_masks):
        fb = (f[0] > 0.5)
        mb = (mk > 0)
        inter = torch.logical_and(fb, mb).sum().item()
        union = torch.logical_or(fb, mb).sum().item()
        ious.append(inter / union if union else 1.0)
    assert min(ious) > 0.90, f"frame/mask misaligned within clip: IoUs={ious}"

    # All masks in the clip must be identical -> one shared warp, not per-frame.
    for mk in out_masks[1:]:
        assert torch.equal(mk, out_masks[0]), "clip is not temporally consistent"


if __name__ == "__main__":
    test_elastic_uses_same_field_for_image_and_mask()
    test_elastic_field_is_deterministic_given_field()
    test_clip_stays_aligned_and_temporally_consistent()
    print("all augmentation alignment tests passed")
