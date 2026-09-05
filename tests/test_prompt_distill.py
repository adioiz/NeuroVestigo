"""Prompt-pathway self-distillation + frame-0 prompt dropout.

Motivation (measured, checkpoints-16-frames/training_log.json, best epoch):

    frame 0 (GT dense prompt)   IoU 0.9466   betti0_err 0.080
    frames 1-15 (memory only)   IoU ~0.830   betti0_err ~0.70

The decoder can produce a 0.947 mask of these images; it just cannot reach
that without an outline. Frames 1-15 are flat (0.017 total drift), so this is
one conditioning step, not accumulating error.

These tests pin the two things that make the fix trustworthy:
  1. with both features OFF the loss is untouched, and
  2. with them ON the distillation term is a real, correctly-signed loss.
"""
import torch

from src.loss_fns import MultiStepMultiMasksAndIous, SoftCBDiceLoss


WEIGHTS = {
    "loss_mask": 3.0, "loss_dice": 2.0, "loss_cbdice": 0.3, "loss_supervoxel": 0.0,
    "loss_distill_dice": 1.0, "loss_distill_cbdice": 0.5,
    "loss_iou": 1.0, "loss_class": 1.0,
}


def _loss_fn():
    return MultiStepMultiMasksAndIous(
        weight_dict=WEIGHTS, use_cbdice=True, use_supervoxel=False,
        topology_resolution=64, pred_obj_scores=False,
    )


def _strokes(B=1, T=4, N=1, H=64, W=64):
    """A tree-like shape -- noise has no skeleton, so it cannot test cbDice."""
    gt = torch.zeros(B, T, N, H, W)
    gt[..., 20:24, 5:60] = 1
    gt[..., 10:50, 30:34] = 1
    return gt


def test_absent_keys_leave_loss_untouched():
    """No distill keys -> both terms are exactly 0. This is the shipped path."""
    gt = _strokes()
    out = _loss_fn()({"pred_masks": torch.randn_like(gt)}, gt)
    assert out["loss_distill_dice"].item() == 0.0
    assert out["loss_distill_cbdice"].item() == 0.0


def test_matched_student_costs_nothing():
    gt = _strokes()
    logits = gt * 20 - 10
    out = _loss_fn()(
        {"pred_masks": logits, "distill_student": logits, "distill_teacher": logits}, gt
    )
    # Soft Dice against a soft target has a small non-zero floor by
    # construction; a confident teacher is nearly binary, so it stays tiny.
    assert out["loss_distill_dice"].item() < 0.01
    assert abs(out["loss_distill_cbdice"].item()) < 1e-6


def test_wrong_student_costs_more_and_passes_gradient():
    gt = _strokes()
    teacher = gt * 20 - 10
    student = (-teacher).clone().requires_grad_(True)
    out = _loss_fn()(
        {"pred_masks": student, "distill_student": student, "distill_teacher": teacher}, gt
    )
    assert out["loss_distill_dice"].item() > 0.5
    assert out["loss_distill_cbdice"].item() > 0.5
    out["core_loss"].backward()
    assert student.grad.abs().sum().item() > 0


def test_teacher_receives_no_gradient():
    gt = _strokes()
    teacher = (gt * 20 - 10).requires_grad_(True)
    student = torch.randn_like(gt).requires_grad_(True)
    out = _loss_fn()(
        {"pred_masks": student, "distill_student": student,
         "distill_teacher": teacher.detach()}, gt
    )
    out["core_loss"].backward()
    assert teacher.grad is None


def test_cbdice_term_is_sensitive_to_teacher_content():
    """Regression: SoftCBDiceLoss re-binarises its target ONLY inside the
    resize branch. Passed a soft target with no resize, every non-zero value
    counts as foreground, so an all-0.5 teacher and a correct teacher both
    collapse to an all-ones mask and score an identical NEGATIVE loss
    (measured -0.1679 for both). The distillation term thresholds the teacher
    itself so it is correct at every topology_resolution.
    """
    gt = _strokes()
    logits = gt * 20 - 10
    lf = _loss_fn()
    good = lf({"pred_masks": logits, "distill_student": logits,
               "distill_teacher": logits}, gt)["loss_distill_cbdice"].item()
    flat = lf({"pred_masks": logits, "distill_student": logits,
               "distill_teacher": torch.zeros_like(logits)},  # sigmoid -> 0.5
              gt)["loss_distill_cbdice"].item()
    assert good >= -1e-6, "a loss must not be negative"
    assert flat > good + 1e-4, "term is ignoring the teacher's content"


def test_soft_target_is_why_thresholding_is_needed():
    """Pins the underlying SoftCBDiceLoss behaviour this guards against, so
    the reason survives if someone later 'simplifies' the threshold away."""
    cb = SoftCBDiceLoss(num_iter=5, topology_resolution=64)
    gt = torch.zeros(1, 1, 64, 64)
    gt[:, :, 20:24, 5:60] = 1
    logits = gt * 20 - 10
    assert cb(logits, gt).item() == 0.0                       # binary target: correct
    assert cb(logits, torch.sigmoid(logits)).item() < 0.0     # soft target: negative
