"""Training configuration for SAM 2 LoRA fine-tuning on neuron data"""

from dataclasses import dataclass, field

from typing import Dict, List, Optional


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


@dataclass
class DataConfig:
    """Directory containing converted training data (PNG images + PNG masks)"""
    training_data_dir: str = "training_data"
    num_frames: int = 8 # Number of frames per training clip
    image_size: int = 1024 # Image size for SAM 2 (must match pretrained config)
    max_objects: int = 3 # Maximum objects per clip
    val_fraction: float = 0.25 # Validation split (fraction of videos held out → 5 of 20)
    # Optional: explicitly pin which videos go in val. When set, overrides val_fraction
    # and the random seed-based split. Useful for val-swap experiments.
    val_videos: Optional[List[str]] = None
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
    # Minimum LR ratio — was 0.10 (matching GPT-3). Lowered to 0.05 for Run 7,
    # which produced a genuine late-phase gain at the floor (val IoU climbed
    # from 0.8397 @ ep58 → 0.8401 @ ep88 while LR sat at 5e-6). Lowered
    # further to 0.03 for Run 8 (B+C plan) to give the topology-rebalanced
    # loss regime even finer LR resolution during the long polishing phase.
    min_lr_ratio: float = 0.03


@dataclass
class LossConfig:
    # Loss weights — Run 9: resolution-driven topology fix.
    # Run 8 (val IoU=0.8229) tried weight-rebalancing (dice↓, cbDice↑, supervoxel
    # 6.7×↑) and LOST 0.017 IoU vs Run 7 (0.8401). Diagnosis: the supervoxel
    # loss hit a STRUCTURAL CEILING at val ≈ 0.05 because topology was computed
    # at 256 resolution — hair-thin neuron breaks at 1024 res were invisible to
    # the loss, so weight pumping had nothing to optimize. cbDice was already
    # saturated at val ≈ 0.005.
    # Run 9 fixes the underlying problem instead: bump topology_resolution
    # 256 → 384 (see below) so the losses can SEE the fine-scale breaks. Weight
    # bumps from Run 8 are mostly reverted because the resolution change is now
    # doing the heavy lifting — supervoxel held at a modest 0.5 (vs Run 7's 0.3)
    # since the loss now has real headroom to drop.
    loss_mask_weight: float = 5.0  # Focal loss
    loss_dice_weight: float = 2.0  # Back to Run 7 value
    loss_cbdice_weight: float = 5.0  # Back to Run 7 value (saturated regardless)
    loss_supervoxel_weight: float = 0.5  # Modest bump from Run 7's 0.3 — the resolution change is the real lever
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
    # Lower = faster. 256 is 16× faster than 1024.
    # Run 9: bumped 256 → 384 because supervoxel val plateaued at ~0.05 in Run 8
    # despite a 6.7× weight bump — the loss was blind to fine-scale breaks at 256.
    # 384 res is 2.25× more cells (384²/256² = 1.5²), so topology forward+backward
    # gets ~2.25× slower; expect ~1.6s/step → ~2.5s/step. Total epoch time
    # ~260s → ~400s, so 90 epochs ≈ 10 hours (was 7h in Run 7/8).
    topology_resolution: int = 384


@dataclass
class TrainingConfig:
    # Model
    sam2_model_id: str = "facebook/sam2.1-hiera-tiny"
    freeze_image_encoder: bool = True

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
    num_epochs: int = 110  # Run 8: bumped from 90 to extend the LR-floor polishing phase
                           # (Run 7 improved at the floor from ep78→ep88, so more floor time
                           # should help the new topology-heavy loss regime settle).
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
            "loss_iou": self.loss.loss_iou_weight,
            "loss_class": self.loss.loss_class_weight,
        }
