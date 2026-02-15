"""
Microscopy-specific augmentations for neuron time-lapse training data.

All spatial transforms are applied consistently across frames in a clip.
Intensity transforms can optionally vary slightly per frame.
"""

import random
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter, map_coordinates


def elastic_transform(image: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
    """
    Apply elastic deformation to an image.

    Args:
        image: Input image as numpy array [H, W] or [H, W, C]
        alpha: Intensity of deformation
        sigma: Smoothness of deformation field

    Returns:
        Deformed image
    """
    shape = image.shape[:2]

    # Generate random displacement fields
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

    # Create coordinate grids
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = [np.clip(y + dy, 0, shape[0] - 1), np.clip(x + dx, 0, shape[1] - 1)]

    if image.ndim == 3:
        # Handle multi-channel images
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = map_coordinates(image[:, :, c], indices, order=1, mode='reflect')
        return result
    else:
        return map_coordinates(image, indices, order=1, mode='reflect')


def elastic_transform_mask(mask: np.ndarray, alpha: float, sigma: float, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Apply elastic deformation to a mask using pre-computed displacement fields."""
    shape = mask.shape[:2]
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = [np.clip(y + dy, 0, shape[0] - 1), np.clip(x + dx, 0, shape[1] - 1)]
    return map_coordinates(mask.astype(float), indices, order=0, mode='reflect').astype(mask.dtype)


class VideoAugmentation:
    """
    Augmentation pipeline for video clips (list of PIL images + masks).

    All spatial transforms use the same random parameters across all frames.
    Intensity transforms use the same parameters (microscopy images are grayscale
    duplicated to RGB, so per-frame color jitter is not meaningful).

    Enhanced with stronger augmentations to prevent overfitting:
    - Scale/zoom augmentation
    - Gaussian blur
    - Gaussian noise
    - Elastic deformations (important for neuron morphology)
    """

    def __init__(
        self,
        image_size: int = 1024,
        horizontal_flip_p: float = 0.5,
        vertical_flip_p: float = 0.5,
        rotation_degrees: float = 30.0,  # Increased from 25
        shear: float = 15.0,  # Reduced slightly to compensate for other transforms
        brightness: float = 0.2,  # Increased from 0.15
        contrast: float = 0.2,  # Increased from 0.15
        # New augmentation parameters
        scale_limit: Tuple[float, float] = (0.85, 1.15),  # Random zoom
        gaussian_blur_p: float = 0.3,
        gaussian_blur_sigma: Tuple[float, float] = (0.1, 2.0),
        gaussian_noise_p: float = 0.2,
        gaussian_noise_std: Tuple[float, float] = (0.01, 0.05),
        elastic_transform_p: float = 0.3,
        elastic_alpha: float = 50.0,  # Deformation intensity
        elastic_sigma: float = 5.0,  # Deformation smoothness
        normalize_mean: List[float] = None,
        normalize_std: List[float] = None,
    ):
        self.image_size = image_size
        self.horizontal_flip_p = horizontal_flip_p
        self.vertical_flip_p = vertical_flip_p
        self.rotation_degrees = rotation_degrees
        self.shear = shear
        self.brightness = brightness
        self.contrast = contrast
        # New augmentation settings
        self.scale_limit = scale_limit
        self.gaussian_blur_p = gaussian_blur_p
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.gaussian_noise_p = gaussian_noise_p
        self.gaussian_noise_std = gaussian_noise_std
        self.elastic_transform_p = elastic_transform_p
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        # Use dataset-specific stats if provided, else fall back to ImageNet
        self.normalize_mean = normalize_mean or [0.485, 0.456, 0.406]
        self.normalize_std = normalize_std or [0.229, 0.224, 0.225]

    def __call__(
        self,
        frames: List[torch.Tensor],
        masks: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Apply augmentations to a clip.

        Args:
            frames: List of [C, H, W] float tensors (0-1 range).
            masks: List of [H, W] int/bool tensors.

        Returns:
            Augmented (frames, masks) lists.
        """
        # Sample random parameters once for the whole clip
        do_hflip = random.random() < self.horizontal_flip_p
        do_vflip = random.random() < self.vertical_flip_p
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        scale = random.uniform(*self.scale_limit)
        shear_x = random.uniform(-self.shear, self.shear)
        shear_y = random.uniform(-self.shear, self.shear)
        brightness_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        contrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)

        # New augmentation decisions (consistent across clip)
        do_blur = random.random() < self.gaussian_blur_p
        blur_sigma = random.uniform(*self.gaussian_blur_sigma) if do_blur else 0
        do_noise = random.random() < self.gaussian_noise_p
        noise_std = random.uniform(*self.gaussian_noise_std) if do_noise else 0
        do_elastic = random.random() < self.elastic_transform_p

        # Pre-compute elastic displacement field (same for all frames)
        if do_elastic and len(frames) > 0:
            # Use original frame size for displacement field
            h, w = frames[0].shape[1], frames[0].shape[2]
            dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), self.elastic_sigma) * self.elastic_alpha
            dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), self.elastic_sigma) * self.elastic_alpha
        else:
            dx, dy = None, None

        aug_frames = []
        aug_masks = []

        for frame, mask in zip(frames, masks):
            # Ensure mask has channel dim for transforms
            mask_3d = mask.unsqueeze(0).float()

            # --- Elastic transform (before other spatial transforms) ---
            if do_elastic and dx is not None:
                # Convert to numpy for elastic transform
                frame_np = frame.permute(1, 2, 0).numpy()  # [H, W, C]
                mask_np = mask.numpy()  # [H, W]

                # Apply elastic transform
                frame_np = elastic_transform(frame_np, self.elastic_alpha, self.elastic_sigma)
                mask_np = elastic_transform_mask(mask_np, self.elastic_alpha, self.elastic_sigma, dx, dy)

                # Convert back to tensor
                frame = torch.from_numpy(frame_np).permute(2, 0, 1).float()
                mask_3d = torch.from_numpy(mask_np).unsqueeze(0).float()

            # --- Spatial transforms (consistent) ---
            if do_hflip:
                frame = TF.hflip(frame)
                mask_3d = TF.hflip(mask_3d)

            if do_vflip:
                frame = TF.vflip(frame)
                mask_3d = TF.vflip(mask_3d)

            # Affine (rotation + shear + scale)
            frame = TF.affine(
                frame,
                angle=angle,
                translate=[0, 0],
                scale=scale,  # Now includes random scale
                shear=[shear_x, shear_y],
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0,
            )
            mask_3d = TF.affine(
                mask_3d,
                angle=angle,
                translate=[0, 0],
                scale=scale,
                shear=[shear_x, shear_y],
                interpolation=TF.InterpolationMode.NEAREST,
                fill=0,
            )

            # Resize to target size
            frame = TF.resize(
                frame,
                [self.image_size, self.image_size],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            mask_3d = TF.resize(
                mask_3d,
                [self.image_size, self.image_size],
                interpolation=TF.InterpolationMode.NEAREST,
            )

            # --- Intensity transforms (consistent across frames) ---
            frame = TF.adjust_brightness(frame, brightness_factor)
            frame = TF.adjust_contrast(frame, contrast_factor)

            # Gaussian blur
            if do_blur and blur_sigma > 0:
                frame = TF.gaussian_blur(frame, kernel_size=int(blur_sigma * 4) * 2 + 1, sigma=blur_sigma)

            # Gaussian noise
            if do_noise and noise_std > 0:
                noise = torch.randn_like(frame) * noise_std
                frame = frame + noise

            frame = torch.clamp(frame, 0.0, 1.0)

            # --- Normalize (dataset-specific stats or ImageNet fallback) ---
            frame = TF.normalize(frame, self.normalize_mean, self.normalize_std)

            # Remove channel dim from mask
            mask_out = mask_3d.squeeze(0).long()

            aug_frames.append(frame)
            aug_masks.append(mask_out)

        return aug_frames, aug_masks


class VideoValidationTransform:
    """Validation-only transform: resize + normalize, no augmentation."""

    def __init__(
        self,
        image_size: int = 1024,
        normalize_mean: List[float] = None,
        normalize_std: List[float] = None,
    ):
        self.image_size = image_size
        # Use dataset-specific stats if provided, else fall back to ImageNet
        self.normalize_mean = normalize_mean or [0.485, 0.456, 0.406]
        self.normalize_std = normalize_std or [0.229, 0.224, 0.225]

    def __call__(
        self,
        frames: List[torch.Tensor],
        masks: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        aug_frames = []
        aug_masks = []

        for frame, mask in zip(frames, masks):
            mask_3d = mask.unsqueeze(0).float()

            frame = TF.resize(
                frame,
                [self.image_size, self.image_size],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            mask_3d = TF.resize(
                mask_3d,
                [self.image_size, self.image_size],
                interpolation=TF.InterpolationMode.NEAREST,
            )

            frame = TF.normalize(frame, self.normalize_mean, self.normalize_std)
            mask_out = mask_3d.squeeze(0).long()

            aug_frames.append(frame)
            aug_masks.append(mask_out)

        return aug_frames, aug_masks
