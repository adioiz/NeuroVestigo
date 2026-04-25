"""Training configuration for SAM 2 LoRA fine-tuning on neuron data"""

from dataclasses import dataclass, field

from typing import Dict, List, Optional


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.15  # Increased from 0.05 for better regularization


@dataclass
class DataConfig:
    """Directory containing converted training data (PNG images + PNG masks)"""
    training_data_dir: str = "training_data"
    num_frames: int = 8 # Number of frames per training clip
    image_size: int = 1024 # Image size for SAM 2 (must match pretrained config)
    max_objects: int = 3 # Maximum objects per clip
    val_fraction: float = 0.2 # Validation split (fraction of videos held out)
    # Performance optimizations
    clip_stride: int = 4  # Stride between clip start frames (1=all clips, 4=every 4th)
    cache_in_memory: bool = True  # Cache all frames/masks in RAM at init
    precompute_features: bool = True  # Precompute frozen encoder features before training


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
    base_lr: float = 1e-5  # For non-LoRA trainable params (increased from 5e-6)
    lora_lr: float = 1e-4  # LoRA A matrix learning rate (increased from 5e-5)
    lora_b_lr_ratio: float = 16.0  # LoRA+ ratio: lr_B = lora_lr × ratio (B needs higher LR)
    weight_decay: float = 0.1
    # Gradient clipping
    max_grad_norm: float = 1.0  # Increased from 0.1 (was too aggressive)
    # Scheduler
    warmup_epochs: int = 1  # Reduced from 5 - was too long for short training runs
    # Use cosine annealing after warmup
    use_cosine_schedule: bool = True
    # Minimum LR ratio - never let LR drop below this fraction of initial
    min_lr_ratio: float = 0.1


@dataclass
class LossConfig:
    # Loss weights - balanced for better generalization
    loss_mask_weight: float = 5.0  # Focal loss (reduced from 20.0 to balance with Dice/IoU)
    loss_dice_weight: float = 2.0  # Standard Dice for shape matching
    loss_cbdice_weight: float = 1.0  # cbDice for skeleton topology (upgraded from clDice)
    loss_supervoxel_weight: float = 1.0  # Supervoxel for connectivity preservation
    loss_iou_weight: float = 1.0  # IoU prediction loss
    loss_class_weight: float = 1.0  # Object score loss
    # Focal loss params
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    # cbDice params (Shi et al., MICCAI 2024 - extends clDice with boundary weighting)
    use_cbdice: bool = True  # Enable skeleton topology-preserving loss (cbDice)
    cbdice_num_iter: int = 5  # Iterations for soft skeletonization (reduced from 10 for MPS memory)
    # Supervoxel params
    use_supervoxel: bool = True  # Enable connectivity-preserving loss
    # Topology resolution: resolution at which topology losses are computed
    # (cbDice skeleton/EDT + supervoxel connectivity analysis).
    # Lower = faster. 256 is 16× faster than 1024 with minimal quality loss.
    topology_resolution: int = 256


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
    batch_size: int = 2  # Increased from 1 for more stable gradients
    num_epochs: int = 60
    num_workers: int = 2
    seed: int = 42
    # Use mixed precision (bfloat16 on CUDA, float32 on MPS/CPU)
    use_amp: bool = True

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
