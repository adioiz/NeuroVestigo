"""Data containers for video segmentation training.

These replace the Meta-internal training.utils.data_utils module
which is not included in the pip-installable SAM2 package.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class Object:
    """A single object (neuron) in a frame.

    Attributes:
        object_id: Unique identifier for this object across the video.
        frame_index: Which frame this object appears in.
        segment: Binary mask [H, W] indicating which pixels belong to this object.
    """
    object_id: int
    frame_index: int
    segment: torch.Tensor  # [H, W] boolean mask


@dataclass
class Frame:
    """A single video frame with its objects.

    Attributes:
        data: The frame image as [C, H, W] normalized tensor in [0, 1].
        objects: List of Object instances present in this frame.
    """
    data: torch.Tensor  # [C, H, W] normalized image
    objects: List[Object] = field(default_factory=list)


@dataclass
class VideoDatapoint:
    """A video clip for training.

    Represents a contiguous sequence of frames from a single video,
    typically 4-8 frames for SAM 2 training.

    Attributes:
        frames: List of Frame objects in temporal order.
        video_id: Identifier for which video this clip comes from.
        size: Spatial dimensions (H, W) of the frames.
    """
    frames: List[Frame]
    video_id: int
    size: Tuple[int, int]  # (H, W)


@dataclass
class BatchedVideoDatapoint:
    """Batched video data ready for model input.

    After collation, all video clips are stacked into tensors.

    Attributes:
        images: Stacked frame tensors [B, T, C, H, W].
        masks: Stacked segmentation masks [B, T, N_obj, H, W].
        obj_ids: List of object ID lists, one per batch item.
        video_ids: List of video identifiers.
    """
    images: torch.Tensor      # [B, T, C, H, W]
    masks: torch.Tensor       # [B, T, N_obj, H, W]
    obj_ids: List[List[int]]  # Object IDs per batch item
    video_ids: List[int]

    def to(self, device, non_blocking: bool = False):
        """Move tensors to specified device.

        Args:
            device: Target device (e.g. 'cuda').
            non_blocking: If True and source tensors are pinned, the H2D copy
                returns immediately and runs concurrently with subsequent GPU
                work. Pair with DataLoader(pin_memory=True).
        """
        return BatchedVideoDatapoint(
            images=self.images.to(device, non_blocking=non_blocking),
            masks=self.masks.to(device, non_blocking=non_blocking),
            obj_ids=self.obj_ids,
            video_ids=self.video_ids,
        )


def collate_fn(batch: List[VideoDatapoint], dict_key: str = None) -> BatchedVideoDatapoint:
    """Collate video datapoints into a batch.

    Takes a list of VideoDatapoint objects and stacks them into
    batched tensors suitable for model input.

    Args:
        batch: List of VideoDatapoint objects.
        dict_key: Ignored, kept for API compatibility.

    Returns:
        BatchedVideoDatapoint with stacked tensors.
    """
    B = len(batch)
    T = len(batch[0].frames)
    H, W = batch[0].size

    # Find max objects across batch (for padding)
    max_objs = max(len(vd.frames[0].objects) for vd in batch)
    max_objs = max(max_objs, 1)  # At least 1 object slot

    # Get channel count from first frame
    C = batch[0].frames[0].data.shape[0]

    # Allocate output tensors
    images = torch.zeros(B, T, C, H, W)
    masks = torch.zeros(B, T, max_objs, H, W)
    obj_ids = []
    video_ids = []

    for b, vd in enumerate(batch):
        video_ids.append(vd.video_id)
        batch_obj_ids = []

        for t, frame in enumerate(vd.frames):
            images[b, t] = frame.data

            for o, obj in enumerate(frame.objects):
                if o < max_objs:
                    masks[b, t, o] = obj.segment.float()
                    if t == 0:
                        batch_obj_ids.append(obj.object_id)

        obj_ids.append(batch_obj_ids)

    return BatchedVideoDatapoint(
        images=images,
        masks=masks,
        obj_ids=obj_ids,
        video_ids=video_ids,
    )
