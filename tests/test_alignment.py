"""
Pixel alignment of the resizes that bring predictions back to native size.

A model that predicts exactly the truth must score IoU 1 after the round trip
native -> model resolution -> native. The old run_ours.py order (threshold at
1024, then F.interpolate(mode="nearest")) failed this: "nearest" samples the
top-left pixel of each block, so masks came back about a third of a pixel off
and thin neurites moved a whole pixel. Found by an outside reproduction, worth
+0.04 IoU on run 1 with the same logits.
"""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "benchmark"))

from run_ours import logits_to_native  # noqa: E402

SIZES = [(148, 318), (292, 292), (498, 452), (353, 615)]    # the recordings' range


def thin_mask(H, W):
    """1- and 2-pixel-wide lines at assorted offsets, plus a small blob."""
    m = np.zeros((H, W), np.uint8)
    for c in (7, W // 3, W // 2 + 1, W - 9):
        m[:, c] = 1
    for r in (5, H // 2):
        m[r:r + 2, :] = 1
    m[H // 4:H // 4 + 6, W // 5:W // 5 + 9] = 1
    return m


def perfect_logits(mask, S=1024):
    """What a perfect model would output at S x S: the truth as a +-1 field,
    enlarged centre-aligned like the image the model sees."""
    x = torch.from_numpy(mask.astype(np.float32) * 2 - 1)[None, None]
    return F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False)[0]


@pytest.mark.parametrize("H,W", SIZES)
def test_logits_to_native_is_aligned(H, W):
    m = thin_mask(H, W)
    out = logits_to_native(perfect_logits(m), H, W)[0]
    assert out.shape == (H, W)
    assert np.array_equal(out, m), f"{(out != m).sum()} pixels differ at {H}x{W}"


@pytest.mark.parametrize("H,W", SIZES)
def test_old_resize_was_misaligned(H, W):
    """Documents the bug the fix removes: the old order loses thin lines."""
    m = thin_mask(H, W)
    binary = (perfect_logits(m) > 0).float()[:, None]
    old = (F.interpolate(binary, size=(H, W), mode="nearest")[:, 0, ].numpy() > 0.5)[0]
    assert not np.array_equal(old, m)


# --- automatic baselines (benchmark/native_res.py) ---------------------------

def _thin_scene(H, W, rng, n=14):
    """Straight segments at random sub-pixel position, angle and width 0.8-2.5 px."""
    segs = [(rng.uniform(5, H - 5), rng.uniform(5, W - 5), rng.uniform(0, np.pi),
             rng.uniform(10, 40), rng.uniform(0.8, 2.5)) for _ in range(n)]

    def inside(y, x):
        m = np.zeros(y.shape, bool)
        for cy, cx, a, L, w in segs:
            dy, dx = y - cy, x - cx
            along = dx * np.cos(a) + dy * np.sin(a)
            across = -dx * np.sin(a) + dy * np.cos(a)
            m |= (np.abs(along) <= L / 2) & (np.abs(across) <= w / 2)
        return m
    return inside


@pytest.mark.parametrize("s", [2, 3, 4])
def test_baseline_target_comes_back_centred(s):
    """A perfect segmenter on the s-times grid must come back better than the
    old corner-sampling shrink, and unshifted."""
    from scipy import ndimage
    from native_res import target_at_native

    def match(inst, gt):                       # the runners' union matcher
        labs = np.unique(inst[gt])
        return np.isin(inst, labs[labs > 0])

    def iou(a, b):
        return np.logical_and(a, b).sum() / np.logical_or(a, b).sum()

    rng = np.random.default_rng(s)
    H, W = 120, 160
    new_iou, old_iou = [], []
    for _ in range(10):
        inside = _thin_scene(H, W, rng)
        yy, xx = np.mgrid[0:H, 0:W].astype(float)
        gt = inside(yy, xx)                                          # pixel-centre GT
        Y, X = np.mgrid[0:H * s, 0:W * s].astype(float)
        inst_up = ndimage.label(inside((Y + 0.5) / s - 0.5, (X + 0.5) / s - 0.5))[0]
        new = target_at_native(inst_up, gt, match)
        old = match(inst_up[::s, ::s], gt)                           # = cv2.INTER_NEAREST
        new_iou.append(iou(new, gt))
        old_iou.append(iou(old, gt))
        nudges = {(dy, dx): iou(np.roll(np.roll(new, dy, 0), dx, 1), gt)
                  for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
        assert max(nudges, key=nudges.get) == (0, 0)
    assert np.mean(new_iou) > np.mean(old_iou) + 0.05
    assert np.mean(new_iou) > 0.85


# --- training labels and the frame-0 prompt (src/augmentations.py) -----------

NO_AUGMENTATION = dict(horizontal_flip_p=0, vertical_flip_p=0, rotation_degrees=0, shear=0,
                       brightness=0, contrast=0, scale_limit=(1.0, 1.0), gaussian_blur_p=0,
                       gaussian_noise_p=0, elastic_transform_p=0)


@pytest.mark.parametrize("H,W", SIZES)
@pytest.mark.parametrize("transform", ["train", "val"])
def test_labels_are_enlarged_onto_the_image(H, W, transform):
    """A label enlarged to 1024 must cover what the image enlarged to 1024
    shows. Legacy "nearest" (every run before 2026-11-02) shifts it half an
    output pixel, which on these thin lines leaves only 70-87% overlap;
    "nearest-exact" puts it on the image."""
    from src.augmentations import VideoAugmentation, VideoValidationTransform
    m = thin_mask(H, W)
    frame = torch.from_numpy(m.astype(np.float32))[None].repeat(3, 1, 1)   # image == label
    plain = dict(normalize_mean=[0.0] * 3, normalize_std=[1.0] * 3)

    def overlap(mask_resize):
        if transform == "train":
            tf = VideoAugmentation(1024, mask_resize=mask_resize, **NO_AUGMENTATION, **plain)
        else:
            tf = VideoValidationTransform(1024, mask_resize=mask_resize, **plain)
        frames, masks = tf([frame], [torch.from_numpy(m).long()])
        image, label = frames[0][0].numpy() > 0.5, masks[0].numpy() > 0
        return (image & label).sum() / (image | label).sum()

    assert overlap("nearest-exact") > 0.99
    assert overlap("nearest") < 0.9


def _run_dirs(tmp_path):
    """A run from before 2026-11-02 and one from after, as the trainer leaves them."""
    import dataclasses
    import json
    from src.config import TrainingConfig
    old, new = tmp_path / "old_run", tmp_path / "new_run"
    old.mkdir()
    new.mkdir()
    (old / "training_config.json").write_text(json.dumps({"backbone": "sam2", "data": {"image_size": 1024}}))
    (new / "training_config.json").write_text(json.dumps(dataclasses.asdict(TrainingConfig())))
    return old, new


def test_inference_prompts_each_checkpoint_as_it_was_trained(tmp_path):
    """run_ours.py enlarges the frame-0 prompt the way the checkpoint's labels
    were enlarged: legacy for every run before 2026-11-02 (so the paper's
    numbers cannot move), centred for new runs; stock SAM 2 stays legacy
    whatever sits beside the default --ckpt."""
    from run_ours import prompt_mask_resize
    old, new = _run_dirs(tmp_path)
    assert prompt_mask_resize(str(old / "lora_weights_topo.pt"), load_lora=True) == "nearest"
    assert prompt_mask_resize(str(new / "lora_weights_topo.pt"), load_lora=True) == "nearest-exact"
    assert prompt_mask_resize(str(new / "lora_weights_topo.pt"), load_lora=False) == "nearest"
    assert prompt_mask_resize(str(tmp_path / "none" / "lora_weights.pt"), load_lora=True) == "nearest"


def test_training_mode_new_runs_centred_resumed_runs_unchanged(tmp_path):
    """A fresh run trains on centred labels; a resumed run keeps the mode it
    started with, so a run begun before 2026-11-02 never switches mid-way."""
    from src.config import TrainingConfig, run_mask_resize
    old, new = _run_dirs(tmp_path)
    fresh = TrainingConfig()
    assert run_mask_resize(fresh) == "nearest-exact"
    resumed_old = TrainingConfig()
    resumed_old.resume_from = str(old / "train_state.pt")
    assert run_mask_resize(resumed_old) == "nearest"
    resumed_new = TrainingConfig()
    resumed_new.resume_from = str(new / "train_state.pt")
    assert run_mask_resize(resumed_new) == "nearest-exact"


def test_transforms_refuse_to_guess_the_mask_mode():
    """No default: a caller that forgets mask_resize fails at once instead of
    silently enlarging labels or prompts the wrong way."""
    from src.augmentations import VideoAugmentation, VideoValidationTransform
    for make in (VideoAugmentation, VideoValidationTransform):
        with pytest.raises(TypeError):
            make(1024)
        with pytest.raises(ValueError):
            make(1024, mask_resize="bilinear")
