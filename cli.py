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
    """Run the training loop."""
    from src.config import (
        TrainingConfig, LoRAConfig, DataConfig, OptimizerConfig, AugmentationConfig, LossConfig,
    )
    from src.trainer import train

    # If --use-dataset-stats is passed, leave normalize_mean/std as None
    # so that train.py auto-loads from manifest.json.
    # If --imagenet-stats is passed, use ImageNet stats explicitly.
    aug_config = AugmentationConfig()
    if args.use_imagenet_stats:
        aug_config.normalize_mean = [0.485, 0.456, 0.406]
        aug_config.normalize_std = [0.229, 0.224, 0.225]
    # else: normalize_mean/std remain None → auto-loaded from manifest

    # Configure loss (disable cbDice/supervoxel if requested to save memory/time)
    # --no-topology disables BOTH topology losses at once
    loss_config = LossConfig(
        use_cbdice=not (args.no_cbdice or args.no_topology),
        use_supervoxel=not (args.no_supervoxel or args.no_topology),
        loss_supervoxel_weight=args.supervoxel_weight,
    )

    config = TrainingConfig(
        num_epochs=args.epochs,
        lora=LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha),
        data=DataConfig(
            training_data_dir=args.data_dir,
            num_frames=args.num_frames,
            clip_stride=args.clip_stride,
            cache_in_memory=not args.no_cache,
            precompute_features=not args.no_precompute,
        ),
        optimizer=OptimizerConfig(
            lora_lr=args.lora_lr,
            lora_b_lr_ratio=args.lora_b_lr_ratio,
            base_lr=args.base_lr,
            warmup_epochs=args.warmup_epochs,
        ),
        augmentation=aug_config,
        loss=loss_config,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        val_every_n_epochs=args.val_every,
        val_max_batches=args.val_samples,
        early_stopping_patience=args.patience,
    )

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
    p_train.add_argument("--epochs", type=int, default=60)
    p_train.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2, was 1)")
    p_train.add_argument("--lora-rank", type=int, default=8)
    p_train.add_argument("--lora-alpha", type=float, default=16.0)
    p_train.add_argument("--lora-lr", type=float, default=1e-4, help="LoRA A matrix learning rate (default: 1e-4)")
    p_train.add_argument("--lora-b-lr-ratio", type=float, default=16.0, help="LoRA+ B/A learning rate ratio (default: 16.0)")
    p_train.add_argument("--base-lr", type=float, default=1e-5, help="Base learning rate for non-LoRA params")
    p_train.add_argument("--warmup-epochs", type=int, default=1, help="Number of warmup epochs (default: 1)")
    p_train.add_argument("--num-frames", type=int, default=8)
    p_train.add_argument("--data-dir", type=str, default="training_data")
    p_train.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p_train.add_argument("--val-every", type=int, default=2, help="Validate every N epochs (default: 2, was 5)")
    p_train.add_argument("--val-samples", type=int, default=0, help="Max validation batches (0 = all, >0 = faster validation)")
    p_train.add_argument("--patience", type=int, default=10, help="Early stopping patience (validations without improvement)")
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
        default=1.0,
        help="Weight for supervoxel connectivity loss (default: 1.0)",
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
        default=4,
        help="Stride between clip start frames (1=all clips, 4=every 4th, default: 4)",
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

    args = parser.parse_args()

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "train":
        cmd_train(args)


if __name__ == "__main__":
    main()
