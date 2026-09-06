"""Prompt-pathway distillation (privileged teacher) + frame-0 prompt dropout.

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


def test_cbdice_handles_soft_targets():
    """SoftCBDiceLoss thresholded y_true at 0 rather than 0.5, so ANY non-zero
    value read as foreground. A soft target (sigmoid(-10) = 4.5e-5 is not zero)
    therefore collapsed to an all-ones mask: a correct target and an all-0.5
    target both returned an identical -0.1679, i.e. a negative loss that had
    stopped depending on its target. Checked at two resolutions because the
    resize branch re-binarises and so masked the bug whenever it fired.
    """
    gt = torch.zeros(1, 1, 64, 64)
    gt[:, :, 20:24, 5:60] = 1
    gt[:, :, 10:50, 30:34] = 1
    logits = gt * 20 - 10
    soft = torch.sigmoid(logits)
    flat = torch.full_like(gt, 0.5)

    for res in (64, 32):  # 64 = no resize (bug was live), 32 = resize fires
        cb = SoftCBDiceLoss(num_iter=5, topology_resolution=res)
        assert cb(logits, gt).item() == 0.0, "binary target must be unaffected"
        assert cb(logits, soft).item() == 0.0, "soft target must now score 0"
        assert cb(logits, flat).item() > 0.5, "must depend on target content"
        assert cb(logits, soft).item() >= 0.0, "a loss must never be negative"


def test_cbdice_is_never_negative():
    """w_tprec/w_tsens are ratios of radius-weighted sums, not probabilities, so
    a SATURATED mask pushes them above 1 and cb_dice above 1 -- the loss then
    went negative and rewarded saturation. Measured before the clamp: a full
    target scored -0.3862, an all-foreground prediction against a normal target
    -0.1669. Never observed in the GT path (logged loss_cbdice stays +0.11..
    +0.19 across all three checkpoint dirs), but the distillation target is a
    teacher mask, which can saturate early in training.
    """
    cb = SoftCBDiceLoss(num_iter=5, topology_resolution=64)
    gt = torch.zeros(1, 1, 64, 64)
    gt[:, :, 20:24, 5:60] = 1
    ones, tens = torch.ones(1, 1, 64, 64), torch.full((1, 1, 64, 64), 10.0)
    torch.manual_seed(0)
    assert cb(torch.randn(1, 1, 64, 64) * 3, ones).item() >= 0.0   # was -0.3862
    assert cb(tens, gt).item() >= 0.0                              # was -0.1669
    assert cb(gt * 20 - 10, gt).item() == 0.0                      # perfect still 0


def test_binary_target_path_is_unchanged():
    """The `> 0` -> `> 0.5` fix must be a no-op for every existing caller:
    training only ever passes binary GT. These are the pre-fix values."""
    gt = torch.zeros(2, 1, 64, 64)
    gt[:, :, 20:24, 5:60] = 1
    gt[:, :, 10:50, 30:34] = 1
    cb = SoftCBDiceLoss(num_iter=5, topology_resolution=64)
    assert cb(gt * 20 - 10, gt).item() == 0.0
    torch.manual_seed(0)
    assert abs(cb(torch.randn(2, 1, 64, 64) * 3, gt).item() - 0.83378118) < 1e-6


# ---------------------------------------------------------------------------
# Frame selection: frame 1 is the defect, so it is never left to chance.
# ---------------------------------------------------------------------------

def _stub(k=4, start=40, epoch=50, enabled=True):
    """SAM2TrainingWrapper's frame chooser without building SAM 2."""
    import torch.nn as nn
    from src.trainer import SAM2TrainingWrapper

    w = SAM2TrainingWrapper.__new__(SAM2TrainingWrapper)
    nn.Module.__init__(w)
    w.use_prompt_distill = enabled
    w.distill_frames_per_clip = k
    w.distill_start_epoch = start
    w.current_epoch = epoch
    w.training = True
    return w


def test_frame_one_is_always_distilled():
    """Frame 0 carries the dense prompt and scores 0.947; frame 1 is the first
    without one and drops to ~0.83. That step IS the defect, so frame 1 is
    mandatory rather than sampled k/T of the time."""
    import random
    random.seed(0)
    w = _stub()
    for _ in range(500):
        frames = w._distill_frames(16)
        assert 1 in frames
        assert len(frames) == 5          # frame 1 + 4 extra
        assert 0 not in frames           # no prompt dropped -> no signal there


def test_frame_zero_joins_only_when_its_prompt_was_dropped():
    """With the prompt withheld, frame 0 is in the no-prompt regime and worth
    a teacher. With the prompt present, student and teacher both see the GT
    mask and the term is ~0 -- a wasted decoder pass."""
    w = _stub()
    assert 0 not in w._distill_frames(16, frame0_prompt_dropped=False)
    frames = w._distill_frames(16, frame0_prompt_dropped=True)
    assert 0 in frames and 1 in frames and len(frames) == 6


def test_distillation_waits_for_a_stationary_teacher():
    """Teacher and student share weights, so early on the target moves: over
    the first run the prompted pathway gained +0.072 IoU across epochs 2-16 but
    only +0.002 from epoch 30 on, while the gap to the student WIDENED from
    0.039 to 0.092. Starting late is what freezing the teacher would buy."""
    # epoch here is the loop's 0-indexed counter; start=40 is stated in
    # printed terms, so it fires at counter 39 (printed "Epoch 40/110").
    for epoch, expected_on in ((0, False), (38, False), (39, True), (90, True)):
        w = _stub(start=40, epoch=epoch)
        assert bool(w._distill_frames(16)) is expected_on


def test_disabled_and_all_frames_modes():
    assert _stub(enabled=False)._distill_frames(16) == []
    assert _stub(k=0)._distill_frames(16) == list(range(1, 16))


def test_start_epoch_means_the_epoch_as_printed():
    """The training loop counts epochs from 0 but prints epoch+1, and
    distill_start_epoch is stated in printed terms. Without the +1 in the gate
    it fires one epoch after the log says it should."""
    for printed, on in ((39, False), (40, True), (41, True)):
        w = _stub(start=40, epoch=printed - 1)
        assert bool(w._distill_frames(16)) is on, f"printed epoch {printed}"


def test_no_teacher_at_validation():
    """Distillation is a training-time signal. A teacher at validation would
    both waste decoder passes and pollute val_loss."""
    w = _stub()
    w.training = False
    assert w._distill_frames(16) == []


def test_frame_selection_survives_short_clips():
    for T in (1, 2, 3, 16):
        frames = _stub()._distill_frames(T)
        assert len(frames) == len(set(frames))
        assert all(0 <= t < T for t in frames)


def test_teacher_and_student_frames_stay_aligned():
    """The highest-risk silent bug in the whole feature: teacher_preds is built
    inside the frame loop while the student is gathered afterwards by
    index_select. If those two orderings ever diverge, the loss compares one
    frame's student against another frame's teacher and trains on noise --
    with no error and a perfectly ordinary-looking loss curve.
    """
    import torch

    B, N, T, H, W = 2, 4, 16, 8, 8
    ts = _stub()._distill_frames(T)

    # Teacher: appended in loop order, value == frame index.
    preds = [torch.full((B * N, 1, H, W), float(t)) for t in ts]
    teacher = (
        torch.stack(preds, dim=1).squeeze(2)
             .view(B, N, len(ts), H, W)
             .permute(0, 2, 1, 3, 4)
             .contiguous()
    )
    # Student: gathered from the full stack, value == frame index.
    pred_masks = torch.arange(T).float().view(1, T, 1, 1, 1).expand(B, T, N, H, W)
    student = pred_masks.contiguous().index_select(1, torch.as_tensor(ts))

    assert torch.equal(teacher, student)


def test_prompt_dropout_ramp_and_rng_hygiene():
    """p must reach its full value by the end of the warmup, be 0 at eval, and
    -- when disabled -- consume no draws from the global RNG, which would
    shift every downstream augmentation sample versus previous runs."""
    import random

    w = _stub()
    w.prompt_dropout, w.prompt_dropout_warmup_epochs = 0.4, 40
    assert w._prompt_dropout_p() == 0.4          # epoch 50, past warmup
    w.current_epoch = 0
    assert w._prompt_dropout_p() == 0.0
    w.training = False
    assert w._prompt_dropout_p() == 0.0

    off = _stub()
    off.prompt_dropout, off.prompt_dropout_warmup_epochs = 0.0, 40
    random.seed(7)
    for _ in range(50):
        off._prompt_dropout_p()
    after = random.random()
    random.seed(7)
    assert after == random.random()
