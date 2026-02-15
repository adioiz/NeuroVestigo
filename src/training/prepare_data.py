"""
Convert TIF image + mask stacks into the frame-per-file format needed for training.

Output structure:
    training_data/
        images/
            A31_C2/00000.png, 00001.png, ...
            A42_C1/00000.png, 00001.png, ...
        masks/
            A31_C2/00000.png, 00001.png, ...
            A42_C1/00000.png, 00001.png, ...

Images: 0-255 RGB PNG (lossless; grayscale duplicated to 3 channels, per-frame min-max normalized).
Masks:  8-bit PNG where pixel value = label ID (0 = background).

Also computes per-dataset normalization statistics (mean/std) and saves them
alongside the manifest, so training can use dataset-specific normalization
rather than ImageNet defaults.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tifffile import imread


def prepare_single_video(image_tif_path: str, mask_tif_path: str, output_dir: str, video_name: str):
    """
    Convert a single image+mask TIF pair to per-frame PNG files.

    Args:
        image_tif_path: Path to the source TIF image stack.
        mask_tif_path: Path to the corresponding mask TIF stack.
        output_dir: Root output directory (e.g., 'training_data').
        video_name: Name for this video (used as subdirectory name).

    Returns:
        Dict with TIF metadata
    """
    output_dir = Path(output_dir)
    img_dir = output_dir / "images" / video_name
    mask_dir = output_dir / "masks" / video_name
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Load TIF stacks
    imgs = imread(str(image_tif_path))
    masks = imread(str(mask_tif_path))

    if imgs.ndim == 2:
        imgs = imgs[np.newaxis, ...]
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]

    T = imgs.shape[0]
    assert masks.shape[0] == T, (
        f"Frame count mismatch: images={imgs.shape[0]}, masks={masks.shape[0]}"
    )
    assert imgs.shape[1:] == masks.shape[1:], (
        f"Spatial shape mismatch: images={imgs.shape[1:]}, masks={masks.shape[1:]}"
    )

    num_frames_with_mask = 0
    # Accumulators for computing dataset-level mean/std
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0

    for i in range(T):
        # --- Image ---
        frame = imgs[i].copy().astype(np.float32)
        # Per-frame min-max normalization to 0-255
        fmin, fmax = frame.min(), frame.max()
        if fmax > fmin:
            frame = (frame - fmin) / (fmax - fmin) * 255.0
        frame = frame.astype(np.uint8)

        # Accumulate stats on normalized [0,1] values
        frame_01 = frame.astype(np.float64) / 255.0
        pixel_sum += frame_01.sum()
        pixel_sq_sum += (frame_01 ** 2).sum()
        pixel_count += frame_01.size

        # Grayscale to RGB
        frame_rgb = np.stack([frame, frame, frame], axis=-1)
        # Save as PNG
        Image.fromarray(frame_rgb).save(img_dir / f"{i:05d}.png")

        # --- Mask ---
        mask = masks[i].copy().astype(np.int32)

        mask = np.clip(mask, 0, 255).astype(np.uint8)

        if mask.max() > 0:
            num_frames_with_mask += 1

        Image.fromarray(mask, mode="L").save(mask_dir / f"{i:05d}.png")

    # Per-video statistics
    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)

    print(
        f"  {video_name}: {T} frames, {num_frames_with_mask} with masks, "
        f"mean={mean:.4f}, std={std:.4f}"
    )
    return {
        "video_name": video_name,
        "num_frames": T,
        "frames_with_mask": num_frames_with_mask,
        "frames_without_mask": T - num_frames_with_mask,
        "height": imgs.shape[1],
        "width": imgs.shape[2],
        "pixel_mean": float(mean),
        "pixel_std": float(std),
        "pixel_count": int(pixel_count),
    }


def prepare_dataset(manifest: list, output_dir: str = "training_data"):
    """
    Convert all TIFs listed in the manifest.

    Args:
        manifest: List of dicts with keys:
            - 'image_tif': path to source image TIF
            - 'mask_tif': path to mask TIF
            - 'name': video name
        output_dir: Root output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for entry in manifest:
        image_path = entry["image_tif"]
        mask_path = entry["mask_tif"]
        name = entry.get(
            "name", Path(mask_path).stem.replace("_masks", "").replace(" ", "_")
        )

        print(f"Processing {name}...")
        info = prepare_single_video(
            image_tif_path=image_path,
            mask_tif_path=mask_path,
            output_dir=str(output_dir),
            video_name=name)
        results.append(info)

    # Compute aggregate dataset normalization statistics
    # Weighted average across videos by pixel count
    total_pixels = sum(r["pixel_count"] for r in results)
    dataset_mean = sum(r["pixel_mean"] * r["pixel_count"] for r in results) / total_pixels
    # Combined variance: Var = E[X^2] - (E[X])^2
    dataset_sq_mean = sum(
        (r["pixel_std"] ** 2 + r["pixel_mean"] ** 2) * r["pixel_count"]
        for r in results
    ) / total_pixels
    dataset_std = float(np.sqrt(dataset_sq_mean - dataset_mean ** 2))
    dataset_mean = float(dataset_mean)

    # Since images are grayscale duplicated to 3 channels, all channels have same stats
    norm_stats = {
        "dataset_mean": [dataset_mean, dataset_mean, dataset_mean],
        "dataset_std": [dataset_std, dataset_std, dataset_std],
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
    }

    total_with_mask = sum(r["frames_with_mask"] for r in results)
    total_without_mask = sum(r["frames_without_mask"] for r in results)
    total_frames = total_with_mask + total_without_mask

    # Save manifest with metadata
    output_data = {
        "videos": results,
        "normalization": norm_stats,
        "summary": {
            "total_videos": len(results),
            "total_frames": total_frames,
            "frames_with_mask": total_with_mask,
            "frames_without_mask": total_without_mask,
        },
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved manifest to {manifest_path}")
    print(f"Total: {len(results)} videos, {total_frames} frames")
    print(f"  Frames with mask: {total_with_mask}, without: {total_without_mask}")
    print(f"\nDataset normalization stats (0-1 scale):")
    print(f"  mean = {dataset_mean:.4f}, std = {dataset_std:.4f}")
    print(f"  (ImageNet: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])")
    print(f"\nThese stats are saved in manifest.json under 'normalization'.")
    print(f"train.py will auto-load these stats (pass --use-imagenet-stats to override).")


def main():
    parser = argparse.ArgumentParser(
        description="Convert TIF stacks to training data format"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to JSON manifest file listing image/mask TIF pairs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training_data",
        help="Output directory for converted data",
    )
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    prepare_dataset(manifest, args.output_dir)

if __name__ == "__main__":
    main()
