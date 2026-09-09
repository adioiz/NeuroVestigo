"""Training configuration for SAM 2 LoRA fine-tuning on neuron data"""

from dataclasses import dataclass, field

from typing import Dict, List, Optional, Tuple


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.10
    # Hiera encoder last-stage LoRA adaptation (Run 6 experiment).
    # When True, applies a SECOND LoRA pass on the last 2 blocks of the
    # Hiera trunk (the deepest, highest-dim features). This lets the model
    # adapt its visual representation to microscopy without retraining the
    # full encoder. Uses a smaller rank than the main LoRA because the
    # 768-dim Hiera layers give more absolute capacity per rank unit.
    hiera_last_stage: bool = True
    hiera_rank: int = 8
    hiera_alpha: float = 16.0  # keeps alpha/rank = 2.0 like the main LoRA
    # How many TRAILING Hiera trunk blocks to adapt. Default 2 = stage 3 only
    # (blocks 10-11 of 12), which is the historical behaviour.
    #
    # Why raising this is the best-motivated capacity experiment left: with a
    # ground-truth prompt the model scores t0 = 0.947, but the very next frame
    # drops to 0.843 and then stays essentially flat (-0.0012 IoU/frame out to
    # t15). The deficit is therefore not temporal drift -- it is LOCALISING the
    # neuron without being told where it is, which is an encoder/memory
    # representation problem, not a mask-decoding one. The decoder already gets
    # 68 layers at rank 16; the encoder gets 8 layers at rank 8, and 10 of its
    # 12 blocks are untouched. That asymmetry points the wrong way for the
    # failure mode we actually have.
    #
    # hiera_blocks=4 reaches blocks 8-11, picking up block 9 -- a GLOBAL
    # attention block (trunk global_att_blocks = [5, 7, 9]) that attends over
    # the whole image instead of a local window.
    hiera_blocks: int = 2


@dataclass
class DataConfig:
    """Directory containing converted training data (PNG images + PNG masks)"""
    training_data_dir: str = "training_data"
    num_frames: int = 8 # Number of frames per training clip
    # Frames per VALIDATION clip. None = follow num_frames.
    # Pin this when experimenting with num_frames: val_iou is measured over a
    # propagation chain of this length, so raising num_frames alone silently
    # lengthens the chain validation has to survive and the number stops being
    # comparable to earlier runs (it would drop for a purely mechanical reason,
    # not because the model got worse). Keep it at 8 to stay comparable with the
    # 0.8529 baseline while training on longer clips.
    val_num_frames: Optional[int] = None
    image_size: int = 1024 # Image size for SAM 2 (must match pretrained config)
    max_objects: int = 3 # Maximum objects per clip
    val_fraction: float = 0.25 # Validation split (fraction of videos held out → 5 of 20)
    # Optional: explicitly pin which videos go in val. When set, overrides val_fraction
    # and the random seed-based split. Useful for val-swap experiments.
    val_videos: Optional[List[str]] = None
    # Held-out TEST set — NEVER trained or validated on. These recordings are
    # removed from the video pool before the train/val split, so they cannot
    # leak into training even under the random split. They are reserved for
    # final benchmarking only (mirrors benchmark/data.py TEST_VIDEOS). To train
    # on a video, remove it from this list.
    test_videos: List[str] = field(default_factory=lambda: [
        "A71_C1", "A81_C1", "A12_C3_100k", "A12_C1_100k_2",
    ])
    # Performance optimizations
    clip_stride: int = 4  # Stride between clip start frames (1=all clips, 4=every 4th)
    cache_in_memory: bool = True  # Cache all frames/masks in RAM at init
    precompute_features: bool = False  # Precompute frozen encoder features before training


@dataclass
class AugmentationConfig:
    """Spatial augmentations (applied consistently across frames) """
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    rotation_degrees: float = 25.0
    shear: float = 20.0
    # Intensity augmentations
    brightness: float = 0.15
    contrast: float = 0.15
    # Scale / blur / noise / elastic. These previously lived only as
    # VideoAugmentation defaults: the trainer never passed them, so they were
    # active but invisible — absent from this config, from the CLI and from the
    # startup banner. That is how a mis-implemented elastic warp (image and
    # mask deformed by two different random fields, label/image agreement only
    # IoU ≈ 0.67) ran unnoticed at p=0.3 for ~10 runs. Values below reproduce
    # the defaults that were in force; they are now logged and overridable.
    scale_limit: Tuple[float, float] = (0.85, 1.15)
    gaussian_blur_p: float = 0.3
    gaussian_blur_sigma: Tuple[float, float] = (0.1, 2.0)
    gaussian_noise_p: float = 0.2
    gaussian_noise_std: Tuple[float, float] = (0.01, 0.05)
    # Elastic deformation. Set elastic_transform_p = 0.0 to disable — worth
    # doing for one run to test whether label noise, not the loss recipe, set
    # the ~0.851 val-IoU ceiling that every loss configuration converged to.
    elastic_transform_p: float = 0.3
    elastic_alpha: float = 50.0
    elastic_sigma: float = 5.0
    # Normalization stats — set from stats.json (dataset-specific) by default.
    # Falls back to ImageNet stats if not overridden.
    # Use --use-dataset-stats to auto-load from stats.json (recommended).
    normalize_mean: Optional[List[float]] = None
    normalize_std: Optional[List[float]] = None


@dataclass
class OptimizerConfig:
    # Learning rates (LoRA+ asymmetric A/B rates, Hayou et al. ICML 2024)
    base_lr: float = 1e-5  # For non-LoRA trainable params
    lora_lr: float = 1e-4  # LoRA A matrix learning rate
    lora_b_lr_ratio: float = 16.0  # LoRA+ ratio: lr_B = lora_lr × ratio (B needs higher LR)
    weight_decay: float = 0.05
    # Gradient clipping
    max_grad_norm: float = 1.0
    # Scheduler
    # Warmup: 2 epochs (~3.3% of 60-epoch run) — was 1 epoch. Bumped to give
    # LoRA-B (zero init) more time to leave the sharp init region. Comparable
    # to GPT-3's ~0.13% warmup scaled to our shorter, fine-tuning regime.
    warmup_epochs: int = 2
    # Use cosine annealing after warmup
    use_cosine_schedule: bool = True
    # Minimum LR ratio — was 0.10 (matching GPT-3). Lowered to 0.05 in Run 7,
    # which produced a genuine late-phase gain at the floor (val IoU climbed
    # from 0.8397 @ ep58 → 0.8401 @ ep88 while LR sat at 5e-6). Run 8/9 briefly
    # tried 0.03, but both runs underperformed Run 7 so we reverted. Run 7's
    # 0.05 stays as the shipping value.
    min_lr_ratio: float = 0.05


@dataclass
class LossConfig:
    # Loss weights — Run 7 (shipping config, best val IoU = 0.8401 @ ep88).
    # Post-Run-7 experiments tried to push topology harder; both lost ground:
    #   - Run 8: dice 2→1.5, cbDice 5→10, supervoxel 0.3→2.0 (256 res) →
    #     val IoU dropped to 0.8229. The supervoxel loss hit a structural
    #     ceiling at val ≈ 0.05 — weight pumping a blind loss does nothing.
    #   - Run 9: topology_resolution 256→384, supervoxel 0.3→0.5 (other
    #     weights at Run 7 baseline) → val IoU = 0.8346. The resolution lift
    #     helped (closed most of Run 8's gap) but didn't beat Run 7 and cost
    #     2.5× wall-clock per step.
    # Conclusion: at 20 videos we've extracted what loss tuning can give.
    # Further gains need MORE DATA, not more loss engineering. Reverted here.
    # cbDice weight dropped 5.0 → 1.0 after the cbDice bug fix (commit d4edf6c):
    # the repaired term is ~20× larger than the old inert one (raw ≈0.13 vs
    # ≈0.008), so at weight 5.0 it contributed ~0.69 — ~65% of the total loss,
    # drowning out Dice/mask. At 1.0 its contribution (~0.13) is comparable to
    # Dice (~0.29). Baseline to beat: weight 5.0 reached val IoU 0.8473 @ ep88.
    #
    # Weight history and what each setting actually bought (val IoU, 25 videos):
    #   focal 5 / cbDice 1.0  → 0.8534 @ep72   (cbDice still had the resize bug)
    #   focal 3 / cbDice 0.3  → 0.8520 @ep84   (Run D, shipped)
    #   focal 3 / cbDice 0.3  → 0.8498 @ep54   (+ cbDice resize fix)
    #   focal 3 / cbDice 0.3  → 0.8529 @ep92   (+ resize fix + elastic fix)  ← BEST
    #   focal 5 / cbDice 1.0  → 0.8517 @ep104  (all fixes; WORSE on every metric)
    #
    # ⚠️ Raising cbDice HURTS topology. Matched tails (ep84-104, n=11), all fixes:
    #     cbDice 0.3 → IoU 0.8524, b0 0.631, b1 0.223
    #     cbDice 1.0 → IoU 0.8510, b0 0.755, b1 0.337   (b1 is +5.6x the scatter)
    # Sorted by how much correctly-computed cbDice influences training, b0 rises
    # monotonically: 0.513 (0.3 + resize bug) → 0.631 (0.3 fixed) → 0.755 (1.0).
    # So the b0 regression after the resize fix was not the fix breaking
    # something -- it was cbDice starting to work. Likely cause: cbDice
    # optimises SOFT-skeleton overlap (five openings, a crude proxy), and
    # pushing it rewards predictions whose skeleton matches while their
    # component structure does not. Do not raise this weight.
    # Every one of these is inside the ±0.003 run-to-run band. Nine
    # configurations and three genuine correctness fixes have not moved val IoU
    # off ~0.852. Treat any weight change as a topology experiment, not an
    # accuracy one, and read it on b0/b1 — never on IoU.
    #
    # ⚠️ Weights are NOT proportional to influence. Measured share of the
    # weighted gradient norm at the operating point (IoU 0.63-0.69):
    #     supervoxel (0.3) ≈ 75%   dice (2.0) ≈ 15%   cbdice ≈ 5%   focal ≈ 4%
    # The supervoxel term normalises each bucket by its own pixel count, so a
    # handful of critical pixels pull ~4.9e6× harder than an ordinary one. That
    # is why it dominates at weight 0.3, and why pushing it UP has failed twice
    # (Run 9: 0.5 → 0.8346; Run 8: 2.0 → 0.8229). If you test supervoxel, test
    # it downward.
    loss_mask_weight: float = 3.0  # Focal loss
    loss_dice_weight: float = 2.0  # Shape coverage
    loss_cbdice_weight: float = 0.3  # Skeleton topology
    loss_supervoxel_weight: float = 0.3  # Connectivity preservation
    loss_iou_weight: float = 1.0  # IoU prediction loss
    loss_class_weight: float = 1.0  # Object score loss
    # Focal loss params
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    # cbDice params (Shi et al., MICCAI 2024 - extends clDice with boundary weighting)
    use_cbdice: bool = True  # Enable skeleton topology-preserving loss (cbDice)
    cbdice_num_iter: int = 5  # Iterations for soft skeletonization (reduced from 10 for MPS memory)
    # Supervoxel params
    # Re-enabled at small weight (0.3) for Run 5. Previously disabled because
    # the component sat flat at 0.30 — but that was on 12 videos with a
    # different config. With 16 videos and the full memory_attention path
    # active (Path B), it should contribute usefully. The diagnostics will
    # tell us: if the per-component value drops below 0.20 across epochs,
    # the loss is learning; if it sits flat again, disable.
    use_supervoxel: bool = True  # Enable connectivity-preserving loss
    # Topology resolution: resolution at which topology losses are computed
    # (cbDice skeleton/EDT + supervoxel connectivity analysis).
    # Lower = faster. 256 is 16× faster than 1024. Run 9 tried 384: 2.5× slower
    # per step for no IoU gain, so this sat at 256 through Run D.
    #
    # Raised 256 → 512 alongside the cbDice interpolation-mode fix, to retain
    # more thin-neurite detail: the loss receives 1024² inputs, so 256 is a 4×
    # linear downsample (1-2 px neurites go sub-pixel) where 512 is only 2×.
    # Costs 4× the topology compute (measured 225 → 896 ms/forward at B*T*N=16).
    #
    # Two caveats worth recording, both measured, because they weaken the
    # original rationale for this change:
    #   - No aspect distortion is introduced here. augmentations.py already
    #     resizes frames and masks to a square [1024, 1024] grid, so this
    #     resize is square→square; the native 615×353 aspect was flattened
    #     upstream, not by the topology resize.
    #   - Component count (Betti-0) is preserved exactly at BOTH 256 and 512
    #     over 18 sampled frames, so masks are not fragmenting at 256.
    # The defensible claim is finer skeleton/EDT detail, not topology rescue.
    topology_resolution: int = 512

    # ------------------------------------------------------------------
    # Prompt-pathway distillation from a privileged-information teacher
    # ------------------------------------------------------------------
    # In _forward_memory, frame 0 receives the GT mask as a dense prompt and
    # frames 1..T-1 receive `None` -- memory is their only spatial context.
    # The per-frame validation metrics show what that costs:
    #
    #     frame 0 (prompted)      IoU 0.9466   betti0_err 0.080
    #     frames 1-15 (no prompt) IoU ~0.830   betti0_err ~0.70
    #
    # The IoU gap is 12%; the topology gap is ~9x. The decoder can clearly
    # produce a 0.947 mask of these images -- it just cannot reach that
    # without being handed an outline. Frames 1-15 are FLAT (0.017 total
    # drift), so this is one conditioning step, not accumulating error.
    #
    # Distillation closes it from the inside: run the decoder a second
    # time on frame t with that frame's GT as a dense prompt (the teacher,
    # no gradient), and train the ordinary memory-path prediction (the
    # student) to reproduce it. The target is the model's own stronger
    # branch, so nothing is imported from outside the network.
    #
    # The distillation term is topological (cbDice) as well as regional
    # (Dice), because the 9x figure says connectivity is what the no-prompt
    # pathway loses.
    #
    # This is NOT self-distillation. Teacher and student are the same weights
    # on the same image; the teacher differs only in receiving the GT mask as
    # a prompt -- information unavailable at inference. The correct frame is
    # learning using privileged information:
    #   Soft targets:           Hinton et al.,    arXiv:1503.02531
    #   Privileged information: Lopez-Paz et al., ICLR 2016, arXiv:1511.03643
    # OFF. Two runs measured it and both landed below the no-distillation
    # baseline; the code is kept so the negative result stays reproducible.
    #
    #     baseline (no distillation)   val IoU 0.8377
    #     v1  alpha ~36% of core_loss          0.8340
    #     v2  alpha ~10%, frame-1 anchored     0.8327
    #
    # Matched epochs 98-110: v2 0.8304 vs baseline 0.8373, t = -19.6, with
    # Betti-0 and Betti-1 both indistinguishable. Lowering alpha did not help.
    #
    # WHY, measured by benchmark/probe_teacher.py on the shipped baseline:
    #
    #     correct GT prompt 0.9378 | WRONG mask 0.3870 | NO prompt 0.0093
    #
    # Unprompted the decoder scores 0.009 -- it cannot find the neuron at all,
    # and handed a wrong mask it segments the wrong thing. So the teacher is a
    # mask REFINER, not a segmenter: its 0.94 is the ground truth handed to it
    # and cleaned up, and its entire advantage is the answer itself. There is
    # no transferable skill, at any weight, which is why the t0-t1 gap stayed
    # flat (0.0963 -> 0.0987) across both runs.
    #
    # A second reason it could only hurt: the distillation cbDice term is
    # effectively ~30% more cbDice pull (0.1 x 0.155 against 0.3 x 0.170), and
    # the sweep recorded above already established that raising cbDice makes
    # Betti-0 monotonically WORSE.
    use_prompt_distill: bool = False
    # Weights cut 1.0/0.5 -> 0.2/0.1 after the first run. At the old values the
    # two distillation terms were 32-37% of core_loss for the whole run -- a
    # third of the gradient budget aimed at a target the student cannot reach,
    # since the teacher is handed the ground-truth mask. That run peaked at
    # val_iou 0.8340 against the 0.8377 baseline and then sat flat for 44
    # epochs. At 0.2/0.1 the terms are ~10% of the loss: a regulariser, which
    # is what an auxiliary objective should be.
    distill_dice_weight: float = 0.2
    distill_cbdice_weight: float = 0.1
    # Distillation is OFF until this epoch. The teacher shares weights with the
    # student, so early in training it is a moving target -- measured over the
    # first run, the prompted pathway gained +0.072 IoU across epochs 2-16 and
    # only +0.002 from epoch 30 onward, while the student gained +0.024 overall
    # and the gap WIDENED from 0.039 to 0.092. Starting once the teacher is
    # stationary is what freezing it would buy, without a second copy of the
    # weights in memory.
    distill_start_epoch: int = 40

    # ------------------------------------------------------------------
    # Topology tie-break checkpoint  ->  lora_weights_topo.pt
    # ------------------------------------------------------------------
    # The primary checkpoint stays selected on val_iou, for comparability:
    # the baseline and every competitor were chosen that way, and switching
    # criteria after seeing the numbers is a researcher degree of freedom a
    # reviewer will rightly flag.
    #
    # But once IoU saturates, "best IoU" is close to arbitrary. Measured over
    # epochs 40-58 of the first v2 run: val_iou sd 0.0072 on a mean of 0.8252
    # (0.9% CV) -- every epoch within 0.006 of every other. Among models that
    # IoU cannot tell apart, preferring the one with better topology is a
    # TIE-BREAK, not a change of criterion.
    #
    # Betti-0 is noisy -- 47% CV over those epochs, swinging 0.676 to 2.232
    # between adjacent validations, 54x noisier than IoU in relative terms. A
    # trailing mean was tried to damp that and REJECTED: replayed on the real
    # v2 log it selected epoch 50 over epoch 48, although 48 was better on both
    # metrics (IoU 0.8324 vs 0.8274, b0 0.682 vs 0.759). The window at epoch 48
    # still contained epoch 44's outlier of 2.232 -- a DIFFERENT model -- so
    # smoothing across epochs misattributes one model's measurement to another.
    #
    # Raw Betti-0 it is. That does track noise, so this file is not a claim
    # about which checkpoint is better: it PRESERVES THE OPTION. Which of the
    # two wins is settled on the held-out test set, where clDice and cbDice are
    # scored, not on this rule.
    #
    # Saves to a SEPARATE file. lora_weights.pt is untouched, early stopping is
    # unaffected, and training dynamics are identical.
    save_topo_checkpoint: bool = True
    topo_iou_margin: float = 0.005   # "indistinguishable on IoU" band
    # EXTRA teacher frames per clip, on top of frame 1 which is always
    # distilled (and frame 0 when its prompt was dropped). Frame 1 is the
    # defect itself -- the first frame without a dense prompt, where IoU steps
    # from 0.947 down to ~0.83 -- so sampling it only k/T of the time spent
    # most of the budget on frames 2..15, which sit at frame 1's level anyway.
    # The image encoder is shared, so each extra frame costs one prompt-encoder
    # + mask-decoder pass, not a full forward. 0 = every remaining frame.
    distill_frames_per_clip: int = 4


@dataclass
class TrainingConfig:
    # Model
    sam2_model_id: str = "facebook/sam2.1-hiera-tiny"
    freeze_image_encoder: bool = True
    # "sam2" (default, and what every shipped checkpoint was trained with) or
    # "sam3" to put the same dual-rank LoRA+ on SAM 3.1's tracker instead.
    # Opt-in on purpose: SAM 3.1's tracker is a different size and its
    # checkpoints are NOT interchangeable with the SAM 2 ones, so a run must
    # say which backbone it used. Run benchmark/sam3_lora_preflight.py first.
    backbone: str = "sam2"

    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Data
    data: DataConfig = field(default_factory=DataConfig)

    # Augmentations
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    # Optimizer
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # Loss
    loss: LossConfig = field(default_factory=LossConfig)

    # Training
    batch_size: int = 2  # Small batch for implicit regularization on a small dataset
    num_epochs: int = 90  # Run 7 shipping value. Bumped from 60 to give 20 videos +
                          # rank-16 LoRA enough training. Best epoch lands ~80-88
                          # consistently (cosine floor phase). Run 8/9 tried 110 with
                          # extended floor — didn't change outcome, reverted.
    num_workers: int = 2
    seed: int = 42
    # Use mixed precision (bfloat16 on CUDA, float32 on MPS/CPU)
    use_amp: bool = True

    # torch.compile (PyTorch 2.x). Off by default — first call adds 30s-2min
    # of compile overhead and recompile loops can hurt throughput if shapes
    # vary. SAM2's own video predictor compiles both modules, so they're
    # known to work; enable when you want the speedup and can absorb the
    # one-time compile cost.
    compile_encoder: bool = False  # Compile model.image_encoder.forward
    compile_decoder: bool = False  # Compile model.sam_mask_decoder.forward

    # Validation
    val_every_n_epochs: int = 2  # More frequent validation
    val_max_batches: int = 0  # Max batches for validation (0 = all batches)

    # Early stopping
    early_stopping_patience: int = 10  # Stop if no improvement for N val checks

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True

    # SAM2Train settings
    prob_to_use_pt_input: float = 0.0  # Use mask input
    num_init_cond_frames: int = 1  # Condition on 1 frame

    # Forward-pass propagation path. When True, the wrapper uses SAM2's full
    # track_step API — memory_attention conditioning + memory_encoder + SAM
    # heads — so the LoRA adapters wrapped around memory_attention layers
    # actually receive gradients. When False, falls back to the simplified
    # path (interpolate previous prediction as a dense prompt on each frame),
    # which is faster but leaves the memory_attention LoRAs inactive. Keep
    # True unless you want to A/B compare against the legacy behavior.
    # Frame-0 prompt dropout (scheduled sampling, Bengio et al. NeurIPS 2015,
    # arXiv:1506.03099). Frame 0 currently ALWAYS gets the GT dense prompt, so
    # 1/T of the training signal is spent on a conditioning regime that does
    # not exist at inference -- classic exposure bias. With probability p the
    # prompt is withheld and frame 0 must be segmented from features alone,
    # which is the frames-1+ condition. Ramped from 0 over the warmup so an
    # early bad frame-0 prediction cannot poison a whole clip's memory bank.
    # OFF. It trains for a condition that DOES NOT EXIST at inference.
    #
    # Frame 0 always carries the mask in real use: benchmark/run_ours.py passes
    # `mask_inputs=(frame0 if t == 0 else None)` and the GUI has the user draw
    # frame 0 before propagating. Dropping that prompt during training creates
    # a third regime that occurs nowhere:
    #
    #     frame 0 normal    prompt, NO memory     happens at inference
    #     frames 1+         no prompt, MEMORY     happens at inference
    #     frame 0 dropped   no prompt, NO memory  happens NEVER
    #
    # The original justification was exposure bias -- that withholding the
    # prompt reproduced the frames-1+ situation. It does not: frames 1+ have
    # the memory bank and a de-prompted frame 0 has nothing. There was no
    # train/test mismatch at frame 0 to correct.
    #
    # It DID teach the decoder a real capability, measured against the
    # baseline, with the prompted pathway untouched:
    #
    #     NO prompt   0.0093 -> 0.5292   57x
    #     correct GT  0.9378 -> 0.9372   unchanged
    #     WRONG mask  0.3870 -> 0.3859   unchanged
    #
    # But that capability is never exercised, which is why v2 acquired it and
    # still scored WORSE on val IoU (0.8304 vs 0.8373 over matched epochs).
    #
    # Kept in the code, defaulted to 0.0. Related measurement worth preserving:
    # the same probe shows the memory bank does essentially all the work --
    # 0.009 with neither prompt nor memory, ~0.830 with memory and no prompt.
    prompt_dropout: float = 0.0
    prompt_dropout_warmup_epochs: int = 40

    # --- historical note on the values above -------------------------------
    # Superseded settings from the two distillation runs: p=0.4 over 40 epochs.
    # Probing v2 against the baseline gave,
    #
    #     NO prompt   baseline 0.0093  ->  v2 0.5292      57x
    #     correct GT  baseline 0.9378  ->  v2 0.9372      unchanged
    #     WRONG mask  baseline 0.3870  ->  v2 0.3859      unchanged
    #
    # Both refine a handed mask identically; the whole difference is standalone
    # segmentation, which normal training never acquires. Bimodal though --
    # A31_C1 0.808 and A52_C2 0.780, but the hardest recording A91_C1_2 stays
    # at 0.0001.
    #
    # which is recorded above. Set prompt_dropout back to 0.4 to reproduce
    # checkpoints-distill-v2's dropout behaviour.

    use_memory_attention_path: bool = True

    # Path to a train_state.pt checkpoint to resume from. When set, the
    # train loop restores model + optimiser + scheduler + counters and
    # continues from the saved epoch. Used to chain SLURM jobs together
    # under a 4-hour queue limit: run 1 trains epochs 0..30 then exits,
    # run 2 picks up at epoch 31. Set via --resume PATH on the CLI; leave
    # None for a fresh run.
    resume_from: Optional[str] = None

    # Logging
    log_every_n_steps: int = 10

    @property
    def loss_weight_dict(self) -> Dict[str, float]:
        return {
            "loss_mask": self.loss.loss_mask_weight,
            "loss_dice": self.loss.loss_dice_weight,
            "loss_cbdice": self.loss.loss_cbdice_weight,
            "loss_supervoxel": self.loss.loss_supervoxel_weight,
            "loss_distill_dice": self.loss.distill_dice_weight,
            "loss_distill_cbdice": self.loss.distill_cbdice_weight,
            "loss_iou": self.loss.loss_iou_weight,
            "loss_class": self.loss.loss_class_weight,
        }
