#!/usr/bin/env python3
"""
Top-level entry point for SAM 2 LoRA fine-tuning on neuron data.

Usage:
    # Step 1: Prepare training data from TIF stacks
    python cli.py prepare --manifest data_manifest.json --output-dir training_data

    # Step 2: Train the model
    python cli.py train

    # Step 2 (with custom settings):
    python cli.py train --epochs 80 --lora-rank 16 --data-dir training_data

    # Step 2 (use ImageNet normalization instead of dataset stats):
    python cli.py train --use-imagenet-stats

Example data_manifest.json:
    [
        {
            "image_tif": "/Volumes/Adi-SSD/EfratData/.../A31 C2.tif",
            "mask_tif": "labeled_masks/A31_C2_masks.tif",
            "name": "A31_C2"
        },
        ...
    ]
"""

import argparse
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def cmd_prepare(args):
    """Prepare training data from TIF stacks."""
    from src.prepare_data import prepare_dataset

    with open(args.manifest) as f:
        manifest = json.load(f)

    prepare_dataset(
        manifest=manifest,
        output_dir=args.output_dir,
    )


def cmd_train(args):
    """Run the training loop.

    CLI flags override config.py defaults *only when explicitly provided*.
    Unspecified flags fall through to whatever is in src/config.py, so
    config.py is the single source of truth.
    """
    from src.config import TrainingConfig
    from src.trainer import train

    config = TrainingConfig()  # Start from config.py defaults

    # ----- Top-level overrides -----
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.val_every is not None:
        config.val_every_n_epochs = args.val_every
    if args.val_samples is not None:
        config.val_max_batches = args.val_samples
    if args.patience is not None:
        config.early_stopping_patience = args.patience

    # ----- LoRA -----
    if args.lora_rank is not None:
        config.lora.rank = args.lora_rank
    if args.lora_alpha is not None:
        config.lora.alpha = args.lora_alpha

    # ----- Data -----
    if args.data_dir is not None:
        config.data.training_data_dir = args.data_dir
    if args.num_frames is not None:
        config.data.num_frames = args.num_frames
    if args.clip_stride is not None:
        config.data.clip_stride = args.clip_stride
    if args.no_cache:
        config.data.cache_in_memory = False
    if args.no_precompute:
        config.data.precompute_features = False
    if args.val_videos:
        # Comma-separated list, e.g. --val-videos A51_C1,A52_C2,A31_C2
        config.data.val_videos = [v.strip() for v in args.val_videos.split(",") if v.strip()]

    # ----- torch.compile -----
    if args.compile_encoder:
        config.compile_encoder = True
    if args.compile_decoder:
        config.compile_decoder = True

    # ----- Optimizer -----
    if args.lora_lr is not None:
        config.optimizer.lora_lr = args.lora_lr
    if args.lora_b_lr_ratio is not None:
        config.optimizer.lora_b_lr_ratio = args.lora_b_lr_ratio
    if args.base_lr is not None:
        config.optimizer.base_lr = args.base_lr
    if args.warmup_epochs is not None:
        config.optimizer.warmup_epochs = args.warmup_epochs

    # ----- Augmentation / normalization -----
    if args.use_imagenet_stats:
        config.augmentation.normalize_mean = [0.485, 0.456, 0.406]
        config.augmentation.normalize_std = [0.229, 0.224, 0.225]

    # ----- Loss -----
    if args.no_cbdice or args.no_topology:
        config.loss.use_cbdice = False
    if args.no_supervoxel or args.no_topology:
        config.loss.use_supervoxel = False
    if args.supervoxel_weight is not None:
        config.loss.loss_supervoxel_weight = args.supervoxel_weight

    # ----- Forward path -----
    if args.no_memory_attention:
        config.use_memory_attention_path = False
    if args.no_hiera_last_stage:
        config.lora.hiera_last_stage = False

    # ----- Resume -----
    if args.resume:
        config.resume_from = args.resume

    # ----- Echo the resolved config so the SLURM log makes the source obvious -----
    print("\n=== Resolved training config ===")
    print(f"  lora.rank={config.lora.rank}, lora.alpha={config.lora.alpha}, "
          f"lora.dropout={config.lora.dropout}")
    print(f"  lora.hiera_last_stage={config.lora.hiera_last_stage}, "
          f"hiera_rank={config.lora.hiera_rank}, "
          f"hiera_alpha={config.lora.hiera_alpha}")
    print(f"  optimizer.lora_lr={config.optimizer.lora_lr:.1e}, "
          f"base_lr={config.optimizer.base_lr:.1e}, "
          f"weight_decay={config.optimizer.weight_decay}")
    print(f"  data.precompute_features={config.data.precompute_features}, "
          f"clip_stride={config.data.clip_stride}, "
          f"num_frames={config.data.num_frames}")
    if config.data.val_videos:
        print(f"  data.val_videos={config.data.val_videos} (overrides val_fraction)")
    print(f"  loss.use_cbdice={config.loss.use_cbdice}, "
          f"use_supervoxel={config.loss.use_supervoxel}, "
          f"cbdice_weight={config.loss.loss_cbdice_weight}, "
          f"supervoxel_weight={config.loss.loss_supervoxel_weight}")
    print(f"  use_memory_attention_path={config.use_memory_attention_path}")
    print(f"  batch_size={config.batch_size}, num_epochs={config.num_epochs}")
    if config.resume_from:
        print(f"  resume_from={config.resume_from}")
    if config.compile_encoder or config.compile_decoder:
        print(f"  compile_encoder={config.compile_encoder}, "
              f"compile_decoder={config.compile_decoder}")

    train(config)


def main():
    parser = argparse.ArgumentParser(
        description="SAM 2 LoRA+ Fine-Tuning for Neuron Segmentation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- prepare command ---
    p_prepare = subparsers.add_parser(
        "prepare", help="Convert TIF stacks to training format"
    )
    p_prepare.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to JSON manifest listing image/mask TIF pairs",
    )
    p_prepare.add_argument(
        "--output-dir",
        type=str,
        default="training_data",
        help="Output directory",
    )

    # --- train command ---
    p_train = subparsers.add_parser("train", help="Train the model")
    # All numeric/string defaults are None: when omitted, config.py wins.
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--batch-size", type=int, default=None)
    p_train.add_argument("--lora-rank", type=int, default=None)
    p_train.add_argument("--lora-alpha", type=float, default=None)
    p_train.add_argument("--lora-lr", type=float, default=None, help="LoRA A matrix learning rate")
    p_train.add_argument("--lora-b-lr-ratio", type=float, default=None, help="LoRA+ B/A learning rate ratio")
    p_train.add_argument("--base-lr", type=float, default=None, help="Base learning rate for non-LoRA params")
    p_train.add_argument("--warmup-epochs", type=int, default=None)
    p_train.add_argument("--num-frames", type=int, default=None)
    p_train.add_argument("--data-dir", type=str, default=None)
    p_train.add_argument("--checkpoint-dir", type=str, default=None)
    p_train.add_argument("--val-every", type=int, default=None, help="Validate every N epochs")
    p_train.add_argument("--val-samples", type=int, default=None, help="Max validation batches (0 = all)")
    p_train.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    p_train.add_argument(
        "--use-imagenet-stats",
        action="store_true",
        help="Use ImageNet normalization instead of dataset-specific stats (default: auto-load from manifest)",
    )
    p_train.add_argument(
        "--no-cbdice",
        action="store_true",
        help="Disable cbDice loss to save memory on MPS/CPU",
    )
    p_train.add_argument(
        "--supervoxel-weight",
        type=float,
        default=None,
        help="Weight for supervoxel connectivity loss",
    )
    p_train.add_argument(
        "--no-supervoxel",
        action="store_true",
        help="Disable supervoxel loss",
    )
    p_train.add_argument(
        "--no-topology",
        action="store_true",
        help="Disable BOTH cbDice and supervoxel losses (faster training on CPU/MPS)",
    )
    # Performance optimization arguments
    p_train.add_argument(
        "--clip-stride",
        type=int,
        default=None,
        help="Stride between clip start frames (1=all clips, 4=every 4th)",
    )
    p_train.add_argument(
        "--no-precompute",
        action="store_true",
        help="Disable feature precomputation (slower but enables full pixel-level augmentation)",
    )
    p_train.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable in-memory frame caching (saves RAM but slower disk I/O)",
    )
    p_train.add_argument(
        "--val-videos",
        type=str,
        default=None,
        help=(
            "Comma-separated video names to use as the val split, overriding "
            "val_fraction and the seed-based shuffle. Example: "
            "--val-videos A51_C1,A52_C2,A31_C2"
        ),
    )
    p_train.add_argument(
        "--compile-encoder",
        action="store_true",
        help="torch.compile the image encoder (one-time 30s-2min compile cost)",
    )
    p_train.add_argument(
        "--compile-decoder",
        action="store_true",
        help="torch.compile the sam_mask_decoder (one-time 30s-2min compile cost)",
    )
    p_train.add_argument(
        "--no-memory-attention",
        action="store_true",
        help=(
            "Disable SAM 2's full memory_attention path; fall back to the "
            "simplified previous-mask-as-prompt propagation. Useful for "
            "A/B comparison only — the memory path is the recommended default."
        ),
    )
    p_train.add_argument(
        "--no-hiera-last-stage",
        action="store_true",
        help=(
            "Disable LoRA on the Hiera encoder's last stage. By default we "
            "adapt the last 2 trunk blocks (encoder visual features); use this "
            "for A/B comparison against the encoder-frozen baseline."
        ),
    )
    p_train.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Resume training from a train_state.pt checkpoint. Restores "
            "model + optimiser + scheduler + counters and continues from the "
            "saved epoch. Used to chain SLURM jobs together when the 4-hour "
            "queue limit would otherwise cut training off mid-schedule. "
            "Example: --resume checkpoints/train_state.pt"
        ),
    )

    args = parser.parse_args()

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "train":
        cmd_train(args)


if __name__ == "__main__":
    main()
