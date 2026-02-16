"""
PyTorch Dataset for neuron time-lapse video segmentation training.

Loads pre-converted PNG frames and PNG masks, samples random subsequences,
and returns data in the format expected by SAM2Train.forward().
"""

import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data_utils import (
    BatchedVideoDatapoint,
    Frame,
    Object,
    VideoDatapoint,
    collate_fn,
)


class NeuronVideoDataset(Dataset):
    """
    Dataset that yields video clips for SAM 2 training.

    Each item is a random subsequence of `num_frames` consecutive frames
    from a labeled neuron time-lapse video.
    """

    def __init__(self, data_dir: str, video_names: List[str], num_frames: int = 8, max_objects: int = 3, transform: Optional[Callable] = None):
        """
        Args:
            data_dir: Root directory containing images/ and masks/ subdirs.
            video_names: List of video names (subdirectory names) to include.
            num_frames: Number of frames per training clip.
            max_objects: Maximum number of objects per clip.
            transform: Augmentation transform (receives lists of tensors).
        """
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.max_objects = max_objects
        self.transform = transform


        # Build index of all videos and their frame counts
        self.videos = []
        for name in video_names:
            img_dir = self.data_dir / "images" / name
            mask_dir = self.data_dir / "masks" / name
            if not img_dir.exists() or not mask_dir.exists():
                print(f"Warning: Skipping {name} (directory not found)")
                continue

            frame_files = sorted(img_dir.glob("*.png"))
            n_frames = len(frame_files)
            if n_frames < num_frames:
                print(
                    f"Warning: {name} has {n_frames} frames, need {num_frames}. Skipping."
                )
                continue

            self.videos.append(
                {
                    "name": name,
                    "num_frames": n_frames,
                    "img_dir": img_dir,
                    "mask_dir": mask_dir,
                }
            )

        if not self.videos:
            raise ValueError(f"No valid videos found in {data_dir}")

        # Compute total number of possible clips
        # Each video contributes (num_total_frames - num_frames + 1) clips
        self.clips = []
        for vid_idx, vid in enumerate(self.videos):
            n_clips = vid["num_frames"] - num_frames + 1
            for start in range(n_clips):
                self.clips.append((vid_idx, start))

        print(
            f"NeuronVideoDataset: {len(self.videos)} videos, "
            f"{len(self.clips)} possible clips of {num_frames} frames"
        )

    def __len__(self) -> int:
        return len(self.clips)

    def _load_frame(self, img_dir: Path, idx: int) -> torch.Tensor:
        """Load a PNG frame as [C, H, W] float tensor in [0, 1]."""
        path = img_dir / f"{idx:05d}.png"
        img = Image.open(path).convert("RGB")
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    def _load_mask(self, mask_dir: Path, idx: int) -> torch.Tensor:
        """Load a PNG mask as [H, W] integer tensor."""
        path = mask_dir / f"{idx:05d}.png"
        mask = Image.open(path)
        return torch.from_numpy(np.array(mask)).long()

    def __getitem__(self, index: int) -> VideoDatapoint:
        vid_idx, start_frame = self.clips[index]
        vid = self.videos[vid_idx]

        # Load frames and masks for this clip
        frames_list = []
        masks_list = []
        for t in range(self.num_frames):
            frame_idx = start_frame + t
            frame = self._load_frame(vid["img_dir"], frame_idx)
            mask = self._load_mask(vid["mask_dir"], frame_idx)
            frames_list.append(frame)
            masks_list.append(mask)

        # Apply augmentations
        if self.transform is not None:
            frames_list, masks_list = self.transform(frames_list, masks_list)

        # Determine object IDs present in this clip
        all_labels = set()
        for mask in masks_list:
            labels = torch.unique(mask)
            for l in labels:
                l_val = l.item()
                if l_val > 0:  # Skip background
                    all_labels.add(l_val)

        # Limit to max_objects
        obj_ids = sorted(all_labels)[: self.max_objects]

        # If no objects found in this clip, create a dummy object with empty masks.
        # This teaches the model that not every frame contains a neuron, preventing
        # hallucination on background-only regions (pred_obj_scores handles this).
        if not obj_ids:
            obj_ids = [1]

        # Build per-frame objects
        video_frames = []
        for t in range(self.num_frames):
            frame_tensor = frames_list[t]  # [C, H, W]
            mask_tensor = masks_list[t]  # [H, W]

            objects = []
            for obj_id in obj_ids:
                binary_mask = (mask_tensor == obj_id)
                objects.append(
                    Object(
                        object_id=obj_id,
                        frame_index=t,
                        segment=binary_mask,  # [H, W] bool
                    )
                )

            video_frames.append(Frame(data=frame_tensor, objects=objects))

        H, W = masks_list[0].shape[-2:]
        return VideoDatapoint(
            frames=video_frames,
            video_id=vid_idx,
            size=(H, W),
        )


def neuron_collate_fn(batch: List[VideoDatapoint]) -> BatchedVideoDatapoint:
    """Collate function wrapping Meta's collate_fn."""
    return collate_fn(batch, dict_key="neuron_train")


def get_video_names(data_dir: str) -> List[str]:
    """List all available video names in the data directory."""
    data_dir = Path(data_dir)
    img_dir = data_dir / "images"
    if not img_dir.exists():
        return []
    return sorted([d.name for d in img_dir.iterdir() if d.is_dir()])


def train_val_split(video_names: List[str], val_fraction: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Split video names into train and validation sets."""
    rng = random.Random(seed)
    names = list(video_names)
    rng.shuffle(names)

    n_val = max(1, int(len(names) * val_fraction))
    val_names = names[:n_val]
    train_names = names[n_val:]

    return train_names, val_names
