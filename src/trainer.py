"""
Standalone training script for SAM 2 LoRA fine-tuning on neuron data.

This bypasses Meta's complex distributed Trainer and uses a simple PyTorch
training loop suitable for single-GPU training.

Key differences from Meta's internal training:
- Uses standard build_sam2() instead of SAM2Train (which isn't in pip package)
- Custom forward pass that handles video frames and mask propagation
- Local loss function implementation (loss_fns.py)

Performance optimizations:
- Feature precomputation: frozen image encoder output cached once, skipped
  during training (~10x speedup)
- In-memory caching: all frames/masks loaded into RAM at init
- Clip stride: skip overlapping clips to reduce steps/epoch
- Feature-level flip augmentation: flip cached feature maps instead of pixels
"""

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from sam2.build_sam import build_sam2, _hf_download

from src.config import TrainingConfig
from src.lora import (
    apply_lora,
    get_trainable_parameters,
    save_lora_weights,
    save_train_state,
    load_train_state,
    SAM2_LORA_TARGETS,
    HIERA_LAST_STAGE_LORA_TARGETS,
    hiera_trailing_block_targets,
)
from src.neuron_dataset import (
    NeuronVideoDataset,
    neuron_collate_fn,
    get_video_names,
    train_val_split,
)
from src.augmentations import VideoAugmentation, VideoValidationTransform
from src.evaluate import validate_epoch
from src.loss_fns import MultiStepMultiMasksAndIous
from src.data_utils import BatchedVideoDatapoint


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Feature precomputation (10x speedup for frozen image encoder)
# ---------------------------------------------------------------------------

def precompute_all_features(
    model: "SAM2TrainingWrapper",
    train_dataset: NeuronVideoDataset,
    val_dataset: NeuronVideoDataset,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    image_size: int,
    norm_mean: List[float],
    norm_std: List[float],
) -> Tuple[Dict, List[torch.Tensor]]:
    """Precompute frozen image encoder features for all frames.

    Since the image encoder is frozen (no LoRA on Conv2d layers), its output
    is deterministic for each input frame. Computing it once and caching
    eliminates ~60-70% of per-step compute.

    Args:
        model: SAM2TrainingWrapper (uses model.model for the inner SAM2).
        train_dataset: Training dataset (to iterate over videos).
        val_dataset: Validation dataset (to iterate over videos).
        device: Target device for encoder forward pass.
        use_amp: Whether to use automatic mixed precision.
        amp_dtype: AMP data type (bfloat16 on CUDA, float32 otherwise).
        image_size: Image size (1024 for SAM2).
        norm_mean: Normalization mean (per-channel).
        norm_std: Normalization std (per-channel).

    Returns:
        feature_cache: Dict mapping (video_name, frame_idx) to list of 3 FPN
                       feature tensors in float16 on CPU.
        pos_enc: List of 3 positional encoding tensors on CPU (spatial-only,
                 same for all frames — computed once).
    """
    sam2 = model.model  # Inner SAM2Base model
    was_training = sam2.training
    sam2.eval()

    feature_cache: Dict[Tuple[str, int], List[torch.Tensor]] = {}
    pos_enc: Optional[List[torch.Tensor]] = None

    # Collect all unique videos from both datasets
    all_videos = []
    seen_videos = set()
    for dataset in [train_dataset, val_dataset]:
        for vid in dataset.videos:
            if vid["name"] not in seen_videos:
                all_videos.append((vid, dataset))
                seen_videos.add(vid["name"])

    total_frames = sum(vid["num_frames"] for vid, _ in all_videos)
    processed = 0
    start_time = time.time()

    print(f"  {len(all_videos)} videos, {total_frames} frames total")

    with torch.no_grad():
        for vid, dataset in all_videos:
            for idx in range(vid["num_frames"]):
                # Load raw frame (from cache or disk)
                frame = dataset._load_frame(vid["name"], vid["img_dir"], idx)

                # Apply same preprocessing as VideoValidationTransform:
                # resize to image_size and normalize
                frame = TF.resize(
                    frame,
                    [image_size, image_size],
                    interpolation=TF.InterpolationMode.BILINEAR,
                    antialias=True,
                )
                frame = TF.normalize(frame, norm_mean, norm_std)

                frame_batch = frame.unsqueeze(0).to(device)
                if device == "cuda":
                    frame_batch = frame_batch.contiguous(memory_format=torch.channels_last)

                with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_amp):
                    backbone_out = sam2.forward_image(frame_batch)

                key = (vid["name"], idx)
                # Cache backbone_fpn as float16 on CPU (saves ~50% memory)
                feature_cache[key] = [
                    f.squeeze(0).cpu().half()
                    for f in backbone_out["backbone_fpn"]
                ]

                # Store positional encodings once (they're purely spatial,
                # independent of image content — same for all frames)
                if pos_enc is None:
                    pos_enc = [
                        pe.squeeze(0).cpu()
                        for pe in backbone_out["vision_pos_enc"]
                    ]

                processed += 1
                if processed % 100 == 0 or processed == total_frames:
                    print(f"  Processed {processed}/{total_frames} frames...")

    elapsed = time.time() - start_time
    total_mb = sum(
        sum(f.nelement() * f.element_size() for f in feats)
        for feats in feature_cache.values()
    ) / 1e6
    print(f"  Feature precomputation done in {elapsed:.1f}s ({total_mb:.0f} MB cached)")

    # Clean up GPU memory
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    if was_training:
        sam2.train()

    return feature_cache, pos_enc


def assemble_cached_backbone(
    feature_cache: Dict[Tuple[str, int], List[torch.Tensor]],
    pos_enc: List[torch.Tensor],
    frame_id_lists: List[List[Tuple[str, int]]],
    device: str,
    do_hflip: bool = False,
    do_vflip: bool = False,
) -> Dict[str, list]:
    """Assemble backbone_out dict from cached per-frame features.

    Reconstructs the same dict that model.forward_image() would return,
    but from pre-cached features instead of running the encoder.

    Optionally applies flip augmentation at the feature level (valid because
    the frozen encoder has no learned spatial asymmetry).

    Args:
        feature_cache: Dict mapping (video_name, frame_idx) to list of 3
                       FPN feature tensors (float16, CPU).
        pos_enc: List of 3 positional encoding tensors (CPU).
        frame_id_lists: List (one per batch item) of lists of
                        (video_name, frame_idx) tuples.
        device: Target device.
        do_hflip: Apply horizontal flip to feature maps.
        do_vflip: Apply vertical flip to feature maps.

    Returns:
        backbone_out dict with 'backbone_fpn' and 'vision_pos_enc', compatible
        with model._prepare_backbone_features().
    """
    # Flatten frame_id_lists to match [B*T] ordering:
    # [vid0_f0, vid0_f1, ..., vid0_fT-1, vid1_f0, ..., vid1_fT-1]
    flat_ids = []
    for frame_ids in frame_id_lists:
        flat_ids.extend(frame_ids)

    num_levels = len(feature_cache[flat_ids[0]])
    backbone_fpn = []
    for level in range(num_levels):
        level_feats = [feature_cache[fid][level] for fid in flat_ids]
        stacked = torch.stack(level_feats, dim=0).to(device).float()

        # Feature-level flip augmentation
        if do_hflip:
            stacked = stacked.flip(-1)
        if do_vflip:
            stacked = stacked.flip(-2)

        backbone_fpn.append(stacked)

    # Expand pos_enc to match batch size (PE is spatial-only, same for all)
    BT = len(flat_ids)
    vision_pos_enc = [
        pe.unsqueeze(0).expand(BT, -1, -1, -1).to(device)
        for pe in pos_enc
    ]

    return {
        "backbone_fpn": backbone_fpn,
        "vision_pos_enc": vision_pos_enc,
    }


# ---------------------------------------------------------------------------
# SAM2 Training Wrapper
# ---------------------------------------------------------------------------

class SAM2TrainingWrapper(nn.Module):
    """Wrapper that adapts SAM2Base for training with video clips.

    SAM2 models from the pip package don't have a training-specific forward.
    This wrapper:
    1. Takes batched video data (images + masks)
    2. Processes frames through the image encoder (or uses precomputed features)
    3. Simulates mask propagation using the memory mechanism
    4. Returns predicted masks for loss computation
    """

    def __init__(
        self,
        sam2_model: nn.Module,
        num_frames: int = 8,
        use_memory_path: bool = True,
        prompt_dropout: float = 0.0,
        prompt_dropout_warmup_epochs: int = 0,
        use_prompt_distill: bool = False,
        distill_frames_per_clip: int = 4,
        distill_start_epoch: int = 0,
    ):
        super().__init__()
        self.model = sam2_model
        self.num_frames = num_frames
        self.use_memory_path = use_memory_path
        # Frame-0 prompt dropout + prompt-pathway distillation. Both are
        # off by default; with both off _forward_memory is bit-identical to
        # the version that trained every shipped checkpoint.
        self.prompt_dropout = prompt_dropout
        self.prompt_dropout_warmup_epochs = prompt_dropout_warmup_epochs
        self.use_prompt_distill = use_prompt_distill
        self.distill_frames_per_clip = distill_frames_per_clip
        self.distill_start_epoch = distill_start_epoch
        self.current_epoch = 0

        # When the memory path is active, force the SAM decoder to be exercised
        # on the conditioning frame too. By default SAM 2 has
        # `use_mask_input_as_output_without_sam=True`, which on frame 0 routes
        # the GT mask straight to output (logits = 20*mask - 10) and only calls
        # the SAM decoder to extract an obj_ptr. Setting it to False makes the
        # decoder produce frame 0's mask from features + the GT-as-dense-prompt,
        # so the loss on frame 0 contributes a real gradient to the LoRA-adapted
        # decoder. (The memory_encoder still ingests the predicted mask either
        # way, so memory bank quality is comparable.)
        if use_memory_path:
            self.model.use_mask_input_as_output_without_sam = False

            # CRITICAL: disable the obj-score nuke. Inside
            # SAM2Base._forward_sam_heads (lines 359-368), when
            # pred_obj_scores=True, the predicted mask logits are replaced
            # with NO_OBJ_SCORE (a large negative constant) wherever the
            # obj_score head outputs <= 0. This is an inference-time
            # consistency rule: "if the model says no object, force the
            # mask to empty."
            #
            # During training the obj_score head is UNTRAINED in our setup
            # (we don't surface pred_obj_scores to the loss in either path).
            # So obj_score outputs ~random near zero -> every sample where
            # it lands <0 has its mask nuked -> Dice loss = 1.0,
            # supervoxel = ~30, and the torch.where has no gradient through
            # the boolean condition so the mask decoder receives no signal.
            # The model can NEVER learn from this state.
            #
            # Setting pred_obj_scores=False skips both the nuke and the
            # obj_ptr-with-no_obj_ptr soft mixing (lines 393-403). The
            # latter is fine to skip during training — obj_ptr is learned
            # directly via gradient through the memory bank.
            #
            # The simple path didn't hit this because it bypassed
            # _forward_sam_heads entirely and called sam_mask_decoder
            # directly.
            self.model.pred_obj_scores = False

    def set_epoch(self, epoch: int) -> None:
        """Training loop calls this once per epoch so the prompt-dropout
        curriculum can ramp. Without it dropout stays at its epoch-0 value."""
        self.current_epoch = epoch

    def _prompt_dropout_p(self) -> float:
        """Current frame-0 prompt-dropout probability, ramped linearly over
        `prompt_dropout_warmup_epochs` and then held at `prompt_dropout`."""
        if self.prompt_dropout <= 0.0 or not self.training:
            return 0.0
        w = self.prompt_dropout_warmup_epochs
        if w <= 0:
            return self.prompt_dropout
        return self.prompt_dropout * min(1.0, self.current_epoch / w)

    def _distill_frames(self, T: int, frame0_prompt_dropped: bool = False) -> List[int]:
        """Which frames get a teacher this step.

        FRAME 1 IS ALWAYS INCLUDED. It is the defect: frame 0 has the dense
        prompt and scores 0.947, frame 1 is the first frame without one and
        drops to ~0.83. Sampling it only k/T of the time spent most of the
        budget on frames 2..15, which sit at the same level as frame 1 and so
        carry the same signal more weakly. Frame 1 is where the step happens.

        Frame 0 joins the pool only when its prompt was dropped, because that
        is the only time it is in the no-prompt regime -- otherwise student and
        teacher both see the GT mask there and the term is ~0, i.e. a wasted
        decoder pass.

        The remaining `distill_frames_per_clip` come at random from frames 2+,
        which spreads coverage over every index across a run.

        NOT ACTIVE BEFORE `distill_start_epoch`. The teacher shares weights
        with the student, so early on it is a moving target: measured over the
        first run, the prompted pathway gained +0.072 IoU across epochs 2-16
        and only +0.002 from epoch 30 onward. Distilling during the first
        window chases a target that outruns the student (teacher +0.077 vs
        student +0.024 over that run, with the gap WIDENING 0.039 -> 0.092).
        Waiting until the teacher is stationary is what a fixed teacher would
        buy, without a second copy of the weights.
        """
        if not (self.use_prompt_distill and self.training):
            return []
        # current_epoch is the loop's 0-indexed counter, but the log prints
        # epoch+1 and distill_start_epoch is stated in those printed terms --
        # 40 means the epoch shown as "Epoch 40/110". Without the +1 it would
        # fire one epoch late, which is harmless but makes the log lie.
        if (self.current_epoch + 1) < self.distill_start_epoch:
            return []

        mandatory = [1] if T > 1 else [0]
        if frame0_prompt_dropped and T > 1:
            mandatory.insert(0, 0)

        # Frames 2+ only. Frame 0 is never a random candidate: unless its
        # prompt was dropped (handled above) student and teacher both see the
        # GT mask there, so the term is ~0 and the decoder pass is wasted.
        pool = [t for t in range(2, T) if t not in mandatory]
        k = self.distill_frames_per_clip
        if k <= 0 or k >= len(pool):
            extra = pool
        else:
            extra = random.sample(pool, k)
        return sorted(mandatory + extra)

    @torch.no_grad()
    def _teacher_mask(self, vf_bn, fs, gt_t):
        """The model's own frame-0 conditioning regime, applied at frame t.

        Raw (memory-free) features plus the GT mask as a dense prompt -- exactly
        what track_step does on the initial conditioning frame, which is the
        setup that measures 0.9466 IoU / 0.080 Betti-0 error. Only the prompt
        encoder and mask decoder run here: the image encoder's output is reused
        and memory_attention is skipped entirely.

        Returns high-res mask LOGITS [BN, 1, H, W], no gradient.
        """
        BN = vf_bn[-1].shape[1]
        pix_feat = vf_bn[-1].permute(1, 2, 0).view(BN, self.model.hidden_dim, *fs[-1])
        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *sz)
            for x, sz in zip(vf_bn[:-1], fs[:-1])
        ] or None
        out = self.model._forward_sam_heads(
            backbone_features=pix_feat,
            point_inputs=None,
            mask_inputs=gt_t,
            high_res_features=high_res_features,
            multimask_output=False,
        )
        return out[4]  # high_res_masks

    def forward(
        self,
        batch: BatchedVideoDatapoint,
        precomputed_backbone_out: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training. Dispatches to the memory or simple path."""
        if self.use_memory_path:
            return self._forward_memory(batch, precomputed_backbone_out)
        return self._forward_simple(batch, precomputed_backbone_out)

    # =====================================================================
    # Path B: full memory_attention forward (SAM 2's track_step pipeline)
    # =====================================================================

    def _forward_memory(
        self,
        batch: BatchedVideoDatapoint,
        precomputed_backbone_out: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Use SAM 2's track_step API for temporal propagation.

        On each frame t, track_step
          (a) conditions the current frame's features on past memory entries
              via memory_attention (LoRA adapters there now receive gradients),
          (b) runs the SAM decoder (with the GT mask as a dense prompt when
              t == 0, or with no prompt for t >= 1 — memory provides context),
          (c) encodes the prediction back into the memory bank for future frames.

        Multi-object handling: we batch the per-clip features along the object
        dim so a single track_step call processes batch B*N_obj. Each
        (clip, object) entry of the batched tensors keeps its own memory in
        the shared output_dict.
        """
        images = batch.images  # [B, T, C, H, W]
        gt_masks = batch.masks  # [B, T, N_obj, H, W]
        B, T, C, H, W = images.shape
        N_obj = gt_masks.shape[2]
        BN = B * N_obj

        # 1. Image encoder forward (one pass over B*T frames).
        # precomputed_backbone_out is supported and assumed to have the same
        # B*T leading dim as a fresh forward_image() call would produce.
        if precomputed_backbone_out is not None:
            backbone_out_all = precomputed_backbone_out
        else:
            images_flat = images.view(B * T, C, H, W)
            if images_flat.is_cuda:
                images_flat = images_flat.contiguous(memory_format=torch.channels_last)
            with torch.set_grad_enabled(self.training):
                backbone_out_all = self.model.forward_image(images_flat)

        # 2. GT mask for frame 0, in [B*N_obj, 1, H, W] format expected by
        # _forward_sam_heads (asserted at sam2_base.py:324).
        frame0_masks = gt_masks[:, 0].reshape(BN, 1, H, W).float()

        # 3. Single output_dict shared across all objects: each (clip, object)
        # slice in the BN batch dim maintains its own memory bank entries.
        output_dict: Dict[str, Dict[int, Dict]] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        per_frame_preds = []

        # Option B: withhold frame 0's dense prompt on a fraction of clips, so
        # the conditioning frame is sometimes segmented from features alone --
        # the regime every other frame is always in. Decided once per clip, not
        # per frame: frame 0 is the only prompted frame.
        # Guarded so the RNG is not touched when dropout is off: an
        # unconditional random() call consumes a draw from the global stream
        # and would shift every downstream augmentation sample, silently
        # breaking reproducibility against every previous run.
        _p_drop = self._prompt_dropout_p()
        drop_frame0_prompt = _p_drop > 0.0 and random.random() < _p_drop

        # Option A: frames that additionally get a teacher pass this step.
        distill_ts = self._distill_frames(T, frame0_prompt_dropped=drop_frame0_prompt)
        teacher_preds: List[torch.Tensor] = []

        for t in range(T):
            # Slice frame-t features out of the B*T-batched backbone_out.
            # Layout is [clip0_t0, clip0_t1, ..., clip0_t(T-1), clip1_t0, ...]
            # so frame t across clips is at indices [t, T+t, 2T+t, ...] == [t::T].
            backbone_t = {
                "backbone_fpn": [
                    feat[t::T] for feat in backbone_out_all["backbone_fpn"]
                ],
                "vision_pos_enc": [
                    pe[t::T] for pe in backbone_out_all["vision_pos_enc"]
                ],
            }
            _, vf, vpe, fs = self.model._prepare_backbone_features(backbone_t)

            # vf and vpe come out as lists of [HW, B, C] (one per FPN level).
            # Expand the per-clip batch dim to per-(clip, object) via
            # repeat_interleave along dim=1. Each (clip, object) entry will
            # share the same image features (correct: same frame, different
            # mask prompt) but get its own memory bank slot.
            vf_bn = [x.repeat_interleave(N_obj, dim=1) for x in vf]
            vpe_bn = [x.repeat_interleave(N_obj, dim=1) for x in vpe]

            mask_input = None if (t > 0 or drop_frame0_prompt) else frame0_masks

            # Teacher: same features, but handed frame t's GT as a dense prompt.
            # Must run BEFORE track_step, which appends frame t to output_dict.
            if t in distill_ts:
                teacher_preds.append(
                    self._teacher_mask(vf_bn, fs, gt_masks[:, t].reshape(BN, 1, H, W).float())
                )

            out_t = self.model.track_step(
                frame_idx=t,
                is_init_cond_frame=(t == 0),
                current_vision_feats=vf_bn,
                current_vision_pos_embeds=vpe_bn,
                feat_sizes=fs,
                point_inputs=None,
                mask_inputs=mask_input,
                output_dict=output_dict,
                num_frames=T,
                run_mem_encoder=True,
            )

            # Store the LIVE tensor dict — gradients must flow back through
            # the memory bank into prior frames' decoder + memory_encoder.
            storage_key = (
                "cond_frame_outputs" if t == 0 else "non_cond_frame_outputs"
            )
            output_dict[storage_key][t] = out_t

            # pred_masks_high_res is [BN, 1, H, W]
            per_frame_preds.append(out_t["pred_masks_high_res"])

        # Stack T frames: [BN, T, 1, H, W] -> squeeze -> [BN, T, H, W]
        preds = torch.stack(per_frame_preds, dim=1).squeeze(2)
        # Unflatten (clip, object) and reorder to [B, T, N_obj, H, W]
        pred_masks = (
            preds.view(B, N_obj, T, H, W)
                 .permute(0, 2, 1, 3, 4)
                 .contiguous()
        )

        out: Dict[str, torch.Tensor] = {"pred_masks": pred_masks}

        # Student/teacher pair for the distillation term. Both are logits in
        # [B, Td, N_obj, H, W]; the teacher carries no gradient, so the loss
        # pulls the student toward it and never the reverse.
        if teacher_preds:
            idx = torch.as_tensor(distill_ts, device=pred_masks.device)
            teacher = (
                torch.stack(teacher_preds, dim=1).squeeze(2)
                     .view(B, N_obj, len(distill_ts), H, W)
                     .permute(0, 2, 1, 3, 4)
                     .contiguous()
            )
            out["distill_student"] = pred_masks.index_select(1, idx)
            out["distill_teacher"] = teacher.detach()

        return out

    # =====================================================================
    # Legacy "simple" forward (interpolated-mask propagation). Kept as a
    # fallback / A-B comparison via config.use_memory_attention_path=False.
    # =====================================================================

    def _forward_simple(
        self,
        batch: BatchedVideoDatapoint,
        precomputed_backbone_out: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Original simplified propagation: feed previous predicted mask as a
        dense prompt to the SAM decoder each step. Does NOT use memory_attention,
        so the LoRA adapters wrapped around memory_attention layers stay inactive.
        """
        images = batch.images  # [B, T, C, H, W]
        gt_masks = batch.masks  # [B, T, N_obj, H, W]

        B, T, C, H, W = images.shape
        N_obj = gt_masks.shape[2]

        # Encode all frames through image encoder (or use precomputed features)
        if precomputed_backbone_out is not None:
            backbone_out = precomputed_backbone_out
        else:
            # Reshape to [B*T, C, H, W] for batch processing.
            # Convert to channels-last so cuDNN picks the fast tensor-core
            # conv kernels in the Hiera encoder. Weights were already
            # converted in build_training_model().
            images_flat = images.view(B * T, C, H, W)
            if images_flat.is_cuda:
                images_flat = images_flat.contiguous(memory_format=torch.channels_last)
            with torch.set_grad_enabled(self.training):
                backbone_out = self.model.forward_image(images_flat)

        # Get the image features (handle different SAM2 versions)
        if "backbone_fpn" in backbone_out:
            # SAM 2.1 format - _prepare_backbone_features returns features
            # in format that may vary depending on model configuration
            _, vision_feats, vision_pos_embeds, feat_sizes = self.model._prepare_backbone_features(backbone_out)

            # Store all feature levels for high-res features
            self._current_vision_feats = vision_feats
            self._current_feat_sizes = feat_sizes

            # vision_feats is a list of tensors at different resolutions
            # Take the highest resolution (last element) for main features
            feat = vision_feats[-1]

            # Handle different possible shapes from SAM2:
            # - [B*T, C, H, W] - standard 4D format
            # - [B*T, C, H*W] - flattened spatial (common in transformers)
            # - [HW, B*T, C] - transformer sequence format (HW tokens, batch, channels)
            if feat.dim() == 3:
                # 3D tensor - need to determine format and reshape
                dim0, dim1, dim2 = feat.shape

                # Check for transformer format: [HW, BT, C] where HW=4096, BT=4, C=256
                # Total elements = 4194304 = 4096 * 4 * 256
                if dim0 == 4096 and dim2 == 256:
                    # Format: [H*W, B*T, C] -> [B*T, C, H, W]
                    HW, BT, C_feat = feat.shape
                    H_feat = W_feat = int(HW ** 0.5)  # 64
                    feat = feat.permute(1, 2, 0).view(BT, C_feat, H_feat, W_feat)
                elif dim1 == 256:
                    # Format: [BT, C, H*W] -> [BT, C, H, W]
                    BT, C_feat, HW = feat.shape
                    H_feat = W_feat = int(HW ** 0.5)
                    feat = feat.view(BT, C_feat, H_feat, W_feat)
                elif dim2 == 256:
                    # Format: [BT, H*W, C] -> [BT, C, H, W]
                    BT, HW, C_feat = feat.shape
                    H_feat = W_feat = int(HW ** 0.5)
                    feat = feat.permute(0, 2, 1).view(BT, C_feat, H_feat, W_feat)
                else:
                    raise ValueError(f"Unknown 3D feature format: {feat.shape}")
            # Now feat should be [B*T, C, H, W]
        else:
            # Fallback: direct feature access
            feat = backbone_out["vision_features"]
            self._current_vision_feats = None

        # Reshape features back to video format
        _, C_feat, H_feat, W_feat = feat.shape
        feat = feat.view(B, T, C_feat, H_feat, W_feat)

        # For each object, propagate masks through time
        # We use the ground truth mask from frame 0 as the prompt
        pred_masks_all = []

        for obj_idx in range(N_obj):
            # Get GT mask for this object at frame 0 as prompt
            prompt_mask = gt_masks[:, 0, obj_idx]  # [B, H, W]

            # Collect predictions for all frames
            obj_pred_masks = []

            for t in range(T):
                # Get features for this frame
                frame_feat = feat[:, t]  # [B, C_feat, H_feat, W_feat]

                if t == 0:
                    # First frame: use the GT mask as prompt
                    # Resize mask to feature resolution for conditioning
                    mask_input = F.interpolate(
                        prompt_mask.unsqueeze(1).float(),
                        size=(H_feat, W_feat),
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    # Subsequent frames: use previous prediction as memory
                    # (simplified - in full SAM2 this uses memory attention)
                    mask_input = F.interpolate(
                        obj_pred_masks[-1].unsqueeze(1),
                        size=(H_feat, W_feat),
                        mode="bilinear",
                        align_corners=False,
                    )

                # Predict mask for this frame
                # We use the mask decoder with dense embeddings
                pred_mask = self._decode_mask(frame_feat, mask_input)
                obj_pred_masks.append(pred_mask)

            # Stack predictions for this object: [B, T, H, W]
            obj_pred_masks = torch.stack(obj_pred_masks, dim=1)
            pred_masks_all.append(obj_pred_masks)

        # Stack all objects: [B, T, N_obj, H, W]
        pred_masks = torch.stack(pred_masks_all, dim=2)

        return {"pred_masks": pred_masks}

    def _decode_mask(
        self, image_features: torch.Tensor, mask_input: torch.Tensor, batch_offset: int = 0
    ) -> torch.Tensor:
        """Decode a mask prediction from image features and mask prompt.

        Args:
            image_features: [B, C, H_feat, W_feat] image encoder output
            mask_input: [B, 1, H_feat, W_feat] mask prompt (GT or previous pred)
            batch_offset: Offset into stored FPN features for this frame

        Returns:
            pred_mask: [B, H, W] predicted mask logits at original resolution
        """
        B, C, H_feat, W_feat = image_features.shape

        # Get high-res features from stored FPN outputs
        # vision_feats[0] and [1] are higher resolution features needed by mask decoder
        high_res_features = None
        if self._current_vision_feats is not None and len(self._current_vision_feats) >= 2:
            # Extract the two highest resolution feature maps
            # vision_feats[0]: highest res (e.g., 256x256), vision_feats[1]: mid res (e.g., 128x128)
            feat_s0 = self._current_vision_feats[0]  # [HW, BT, C] format
            feat_s1 = self._current_vision_feats[1]  # [HW, BT, C] format

            # Get the correct batch slice and reshape to [B, C, H, W]
            # For now, extract features for batch_offset
            if feat_s0.dim() == 3:
                # [HW, BT, C] -> need to extract per-frame and reshape
                HW0, BT, C0 = feat_s0.shape
                H0 = W0 = int(HW0 ** 0.5)
                feat_s0_reshaped = feat_s0.permute(1, 2, 0)[:B].reshape(B, C0, H0, W0)

                HW1, _, C1 = feat_s1.shape
                H1 = W1 = int(HW1 ** 0.5)
                feat_s1_reshaped = feat_s1.permute(1, 2, 0)[:B].reshape(B, C1, H1, W1)

                high_res_features = (feat_s0_reshaped, feat_s1_reshaped)

        # Use the model's mask decoder
        sparse_embeddings = self.model.sam_prompt_encoder.not_a_point_embed.weight
        sparse_embeddings = sparse_embeddings.unsqueeze(0).expand(B, -1, -1)

        # Encode mask_input as dense prompt embeddings through SAM2's mask downscaling
        # mask_input is [B, 1, H_feat, W_feat] - resize to 256x256 as SAM2 expects
        mask_for_encoder = F.interpolate(
            mask_input,
            size=(256, 256),  # SAM2 expects 4x the 64x64 embedding resolution
            mode="bilinear",
            align_corners=False,
        )
        # Encode through the prompt encoder's convolutional downscaling network
        dense_prompt = self.model.sam_prompt_encoder.mask_downscaling(mask_for_encoder)

        # Call mask decoder with high_res_features
        # SAM2 mask decoder returns: (masks, iou_predictions, mask_tokens_out, object_score_logits)
        decoder_output = self.model.sam_mask_decoder(
            image_embeddings=image_features,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_prompt,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        low_res_masks = decoder_output[0]  # First element is the masks

        # Upscale to original resolution
        # low_res_masks: [B, 1, H_low, W_low]
        high_res_masks = F.interpolate(
            low_res_masks,
            size=(1024, 1024),  # SAM 2 default size
            mode="bilinear",
            align_corners=False,
        )

        return high_res_masks.squeeze(1)  # [B, H, W]


def build_training_model(config: TrainingConfig, device: str):
    """
    Build SAM2 model with LoRA adapters for training.

    Uses the standard pip-installable SAM2 model wrapped for training,
    instead of the internal SAM2Train class.

    Returns:
        model: SAM2TrainingWrapper with LoRA applied
        loss_fn: MultiStepMultiMasksAndIous loss function
    """
    # Download checkpoint
    config_name, ckpt_path = _hf_download(config.sam2_model_id)

    # Build the standard SAM2 model
    model = build_sam2(
        config_file=config_name,
        ckpt_path=ckpt_path,
        device="cpu",  # Load on CPU first
        apply_postprocessing=False,
    )

    # Freeze image encoder if configured
    if config.freeze_image_encoder:
        for p in model.image_encoder.parameters():
            p.requires_grad = False

    # Apply LoRA to target modules
    print("\nApplying LoRA adapters...")
    wrapped = apply_lora(
        model,
        target_patterns=SAM2_LORA_TARGETS,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
    )
    print(f"  Wrapped {len(wrapped)} Linear layers with LoRA (rank={config.lora.rank})")
    for name in sorted(wrapped):
        print(f"    - {name}")

    # Optional second LoRA pass on the Hiera encoder's last stage. Image
    # encoder params are still frozen (set above); LoRALinear's A/B matrices
    # are NEW parameters that default to requires_grad=True, so the
    # adapters train while the underlying Hiera weights stay frozen. The
    # gradient flows correctly into A and B via the LoRA forward path.
    if config.lora.hiera_last_stage:
        # Derive the block indices from the trunk that was actually loaded
        # rather than hardcoding them, so this survives a backbone swap.
        trunk = model.image_encoder.trunk
        n_blocks = len(trunk.blocks)
        n_adapt = min(config.lora.hiera_blocks, n_blocks)
        hiera_targets = hiera_trailing_block_targets(n_blocks, n_adapt)

        wrapped_hiera = apply_lora(
            model,
            target_patterns=hiera_targets,
            rank=config.lora.hiera_rank,
            alpha=config.lora.hiera_alpha,
            dropout=config.lora.dropout,
        )
        first = n_blocks - n_adapt
        print(
            f"  Wrapped {len(wrapped_hiera)} Hiera Linear layers with LoRA "
            f"(rank={config.lora.hiera_rank}, blocks {first}-{n_blocks - 1} "
            f"of {n_blocks})"
        )
        # Hiera matches 4 Linears per block (attn.qkv, attn.proj, mlp.layers.0,
        # mlp.layers.1). Anything else means the regex silently missed -- which
        # would leave the encoder unadapted while the run looks completely
        # normal, and cost 11 hours to discover. Fail loudly instead.
        expected = 4 * n_adapt
        if len(wrapped_hiera) != expected:
            raise RuntimeError(
                f"Hiera LoRA matched {len(wrapped_hiera)} Linear layers, "
                f"expected {expected} (4 per block x {n_adapt} blocks). The "
                f"target patterns do not fit this trunk: {hiera_targets}"
            )
        for name in sorted(wrapped_hiera):
            print(f"    - {name}")

    # Unfreeze small learnable embeddings
    unfrozen_embeddings = []
    for name, param in model.named_parameters():
        if any(
            key in name
            for key in [
                "maskmem_tpos_enc",
                "no_mem_embed",
                "no_mem_pos_enc",
                "no_obj_ptr",
                "obj_ptr_proj",
                "obj_ptr_tpos_proj",
            ]
        ):
            param.requires_grad = True
            unfrozen_embeddings.append(name)

    if unfrozen_embeddings:
        print(f"\n  Also unfroze {len(unfrozen_embeddings)} embedding parameters:")
        for name in unfrozen_embeddings:
            print(f"    - {name}")

    # Wrap for training
    training_model = SAM2TrainingWrapper(
        model,
        num_frames=config.data.num_frames,
        use_memory_path=config.use_memory_attention_path,
        prompt_dropout=config.prompt_dropout,
        prompt_dropout_warmup_epochs=config.prompt_dropout_warmup_epochs,
        use_prompt_distill=config.loss.use_prompt_distill,
        distill_frames_per_clip=config.loss.distill_frames_per_clip,
        distill_start_epoch=config.loss.distill_start_epoch,
    )
    if config.prompt_dropout > 0:
        print(f"  Frame-0 prompt dropout: p={config.prompt_dropout} "
              f"(ramped over {config.prompt_dropout_warmup_epochs} epochs)")
    if config.loss.use_prompt_distill:
        k = config.loss.distill_frames_per_clip
        print(f"  Prompt-pathway distillation (privileged teacher): frame 1 + "
              f"{'all others' if k <= 0 else f'{k} random'}, from epoch "
              f"{config.loss.distill_start_epoch}, "
              f"dice={config.loss.distill_dice_weight} "
              f"cbdice={config.loss.distill_cbdice_weight}")
    if config.use_memory_attention_path:
        print("  Forward pass: full memory_attention path (track_step API)")
    else:
        print("  Forward pass: simplified propagation (legacy)")

    # Count parameters
    total_params = sum(p.numel() for p in training_model.parameters())
    trainable_params = sum(p.numel() for p in training_model.parameters() if p.requires_grad)
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    # Build loss function with topology-preserving losses
    loss_fn = MultiStepMultiMasksAndIous(
        weight_dict=config.loss_weight_dict,
        focal_alpha=config.loss.focal_alpha,
        focal_gamma=config.loss.focal_gamma,
        pred_obj_scores=True,
        use_cbdice=config.loss.use_cbdice,
        cbdice_num_iter=config.loss.cbdice_num_iter,
        use_supervoxel=config.loss.use_supervoxel,
        topology_resolution=config.loss.topology_resolution,
    )
    if config.loss.use_cbdice:
        print(f"  Using cbDice loss for skeleton topology (weight={config.loss.loss_cbdice_weight})")
    if config.loss.use_supervoxel:
        print(f"  Using supervoxel loss for connectivity (weight={config.loss.loss_supervoxel_weight})")

    training_model = training_model.to(device)

    # Channels-last memory format on CUDA. .to(memory_format=...) only
    # touches 4D Conv weights — Linear / LayerNorm / Embedding stay as-is.
    # Pair with channels_last input in the wrapper forward (below) so
    # cuDNN picks the fast tensor-core conv kernels on Hopper/Blackwell.
    # Numerically equivalent to bf16 precision — different reduction
    # order in cuDNN may shift values by ≤1 ulp but no systematic bias.
    if device == "cuda":
        training_model = training_model.to(memory_format=torch.channels_last)

    # torch.compile (opt-in). Applied AFTER LoRA is wrapped + after .to(device)
    # so the adapter modules and target dtype are baked into the traced graph.
    # We compile the inner SAM2 submodules (not the wrapper) because the
    # wrapper has Python loops + shape-dependent control flow that cause
    # graph breaks or recompiles. SAM2's own sam2_video_predictor.py:992,1006
    # uses this exact pattern.
    if device == "cuda" and config.compile_encoder:
        print("  Compiling image_encoder (first call will take 30s-2min)...")
        training_model.model.image_encoder = torch.compile(
            training_model.model.image_encoder,
            mode="default",
            dynamic=False,
        )
    if device == "cuda" and config.compile_decoder:
        print("  Compiling sam_mask_decoder (first call will take 30s-2min)...")
        training_model.model.sam_mask_decoder = torch.compile(
            training_model.model.sam_mask_decoder,
            mode="default",
            dynamic=False,
        )

    return training_model, loss_fn


def build_optimizer(model: nn.Module, config: TrainingConfig):
    """Build AdamW optimizer with LoRA+ asymmetric A/B learning rates.

    LoRA+ (Hayou et al., ICML 2024) uses different learning rates for A and B:
    - A (down-projection): standard lora_lr
    - B (up-projection): lora_lr × lora_b_lr_ratio (default 16×)

    B starts at zeros and needs a higher LR to overcome gradient imbalance.
    """
    param_groups = get_trainable_parameters(model)

    lora_b_lr = config.optimizer.lora_lr * config.optimizer.lora_b_lr_ratio

    # Use the fused AdamW kernel on CUDA. Numerically equivalent to the
    # default `foreach` impl, but packs the entire optimizer step (m/v
    # update, weight update, weight decay) into a single CUDA kernel
    # launch — much faster when there are many small parameter tensors,
    # which is exactly LoRA (rank 8 × 68 layers = 136 tiny tensors).
    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(
        [
            {
                "params": param_groups["lora_a"],
                "lr": config.optimizer.lora_lr,
                "weight_decay": config.optimizer.weight_decay,
            },
            {
                "params": param_groups["lora_b"],
                "lr": lora_b_lr,
                "weight_decay": config.optimizer.weight_decay,
            },
            {
                "params": param_groups["other"],
                "lr": config.optimizer.base_lr,
                "weight_decay": 0.0,  # No weight decay for embeddings
            },
        ],
        fused=use_fused,
    )

    print(f"  LoRA+ optimizer: A lr={config.optimizer.lora_lr:.1e}, "
          f"B lr={lora_b_lr:.1e} (ratio={config.optimizer.lora_b_lr_ratio}), "
          f"embeddings lr={config.optimizer.base_lr:.1e}, "
          f"fused={use_fused}")
    return optimizer


def build_scheduler(optimizer, config: TrainingConfig, steps_per_epoch: int):
    """Build LR scheduler with warmup + cosine annealing with minimum LR floor."""
    total_steps = config.num_epochs * steps_per_epoch
    warmup_steps = config.optimizer.warmup_epochs * steps_per_epoch
    min_lr_ratio = config.optimizer.min_lr_ratio  # Never let LR drop below this

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup from min_lr_ratio to 1.0
            warmup_progress = step / max(warmup_steps, 1)
            return min_lr_ratio + (1.0 - min_lr_ratio) * warmup_progress
        if config.optimizer.use_cosine_schedule:
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            # Cosine annealing with minimum floor
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr_ratio, cosine_decay)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _load_normalization_stats(config: TrainingConfig):
    """
    Resolve normalization stats for training.

    If config.augmentation.normalize_mean/std are already set, use them.
    Otherwise, try to load dataset-specific stats from stats.json.
    Falls back to ImageNet stats if stats file is not found.
    """
    mean = config.augmentation.normalize_mean
    std = config.augmentation.normalize_std

    if mean is not None and std is not None:
        return mean, std

    # Try loading from stats.json
    stats_path = Path(config.data.training_data_dir) / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        norm = stats.get("normalization", {})
        ds_mean = norm.get("dataset_mean")
        ds_std = norm.get("dataset_std")
        if ds_mean and ds_std:
            print(f"  Using dataset normalization stats from {stats_path}")
            print(f"    mean={ds_mean}, std={ds_std}")
            return ds_mean, ds_std

    # Fall back to ImageNet
    fallback_mean = [0.485, 0.456, 0.406]
    fallback_std = [0.229, 0.224, 0.225]
    print(f"  No dataset stats found, falling back to ImageNet normalization")
    print(f"    mean={fallback_mean}, std={fallback_std}")
    return fallback_mean, fallback_std


def train(config: TrainingConfig = None):
    """Main training function."""
    if config is None:
        config = TrainingConfig()

    device = detect_device()
    print(f"Using device: {device}")

    # Enable cuDNN's FlashAttention-3 backend for SDPA. SAM2 calls
    # F.scaled_dot_product_attention internally; PyTorch's dispatcher then
    # picks the fastest available kernel based on these enable flags. On
    # B200 (Blackwell, compute 10.0) the cuDNN backend = FA-3, ~10-20%
    # faster than the FA-2 fallback. Default is on in PyTorch 2.8 but we
    # set it explicitly so older versions don't silently regress to FA-2.
    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)        # FA-2 fallback
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        # Math kernel stays as last-resort fallback (correctness for odd shapes).
        active = []
        if torch.backends.cuda.cudnn_sdp_enabled():
            active.append("cuDNN(FA-3)")
        if torch.backends.cuda.flash_sdp_enabled():
            active.append("flash(FA-2)")
        if torch.backends.cuda.mem_efficient_sdp_enabled():
            active.append("mem-efficient")
        if torch.backends.cuda.math_sdp_enabled():
            active.append("math")
        print(f"SDPA backends enabled (PyTorch picks best at runtime): {', '.join(active)}")

    # Seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # --- Data ---
    print("\n=== Data Setup ===")
    all_videos = get_video_names(config.data.training_data_dir)
    if not all_videos:
        raise RuntimeError(
            f"No training videos found in {config.data.training_data_dir}/images/. "
            f"Run prepare_data.py first."
        )

    # Remove the held-out TEST set from the pool BEFORE any split, so it can
    # never leak into train or val (even under the random split).
    test_set = set(config.data.test_videos or [])
    if test_set:
        leaked_to_val = test_set & set(config.data.val_videos or [])
        if leaked_to_val:
            raise ValueError(
                f"Test videos requested as validation: {sorted(leaked_to_val)}. "
                f"Remove them from config.data.test_videos to use them."
            )
        excluded = sorted(v for v in all_videos if v in test_set)
        all_videos = [v for v in all_videos if v not in test_set]
        print(f"Excluding {len(excluded)} held-out TEST video(s) from train/val: {excluded}")

    if config.data.val_videos:
        # Explicit override — pin val to these videos, train on the rest.
        all_set = set(all_videos)
        requested = list(config.data.val_videos)
        missing = [v for v in requested if v not in all_set]
        if missing:
            raise ValueError(
                f"--val-videos requested {missing}, but they are not in the "
                f"dataset. Available: {sorted(all_videos)}"
            )
        val_videos = requested
        train_videos = [v for v in all_videos if v not in set(val_videos)]
        print(f"Using explicit val split (val_fraction ignored)")
    else:
        train_videos, val_videos = train_val_split(
            all_videos, val_fraction=config.data.val_fraction, seed=config.seed
        )
    print(f"Training videos ({len(train_videos)}): {train_videos}")
    print(f"Validation videos ({len(val_videos)}): {val_videos}")

    # Resolve normalization stats (dataset-specific or ImageNet fallback)
    norm_mean, norm_std = _load_normalization_stats(config)

    # When precomputing features, use validation transform for training data
    # (no pixel-level augmentation — augmentation happens at feature level via flips)
    if config.data.precompute_features:
        train_transform = VideoValidationTransform(
            image_size=config.data.image_size,
            normalize_mean=norm_mean,
            normalize_std=norm_std,
        )
        print("  Using precomputed features → train augmentation at feature level (flips only)")
    else:
        train_transform = VideoAugmentation(
            image_size=config.data.image_size,
            horizontal_flip_p=config.augmentation.horizontal_flip_p,
            vertical_flip_p=config.augmentation.vertical_flip_p,
            rotation_degrees=config.augmentation.rotation_degrees,
            shear=config.augmentation.shear,
            brightness=config.augmentation.brightness,
            contrast=config.augmentation.contrast,
            # Previously omitted, so these silently ran at VideoAugmentation's
            # own defaults and never appeared in the config banner.
            scale_limit=config.augmentation.scale_limit,
            gaussian_blur_p=config.augmentation.gaussian_blur_p,
            gaussian_blur_sigma=config.augmentation.gaussian_blur_sigma,
            gaussian_noise_p=config.augmentation.gaussian_noise_p,
            gaussian_noise_std=config.augmentation.gaussian_noise_std,
            elastic_transform_p=config.augmentation.elastic_transform_p,
            elastic_alpha=config.augmentation.elastic_alpha,
            elastic_sigma=config.augmentation.elastic_sigma,
            normalize_mean=norm_mean,
            normalize_std=norm_std,
        )

    val_transform = VideoValidationTransform(
        image_size=config.data.image_size,
        normalize_mean=norm_mean,
        normalize_std=norm_std,
    )

    # Datasets with performance options
    train_dataset = NeuronVideoDataset(
        data_dir=config.data.training_data_dir,
        video_names=train_videos,
        num_frames=config.data.num_frames,
        max_objects=config.data.max_objects,
        transform=train_transform,
        clip_stride=config.data.clip_stride,
        cache_in_memory=config.data.cache_in_memory,
    )
    # Validation clip length is pinned separately so that changing num_frames
    # does not silently change what val_iou measures. per_frame_iou is reported
    # as t0..t{val_frames-1}, and the aggregate is an average over that chain,
    # so a longer chain lowers val_iou mechanically.
    val_frames = config.data.val_num_frames or config.data.num_frames
    if val_frames != config.data.num_frames:
        print(f"Validation clip length pinned to {val_frames} frames "
              f"(training clips are {config.data.num_frames}) — val_iou stays "
              f"comparable to runs at num_frames={val_frames}")
    val_dataset = NeuronVideoDataset(
        data_dir=config.data.training_data_dir,
        video_names=val_videos,
        num_frames=val_frames,
        max_objects=config.data.max_objects,
        transform=val_transform,
        clip_stride=config.data.clip_stride,
        cache_in_memory=config.data.cache_in_memory,
    )

    # When data is cached in memory, workers just add pickle overhead
    effective_num_workers = 0 if config.data.cache_in_memory else config.num_workers

    # Always pin memory on CUDA — even when frames are RAM-cached, the cached
    # tensors are unpinned, so the H2D copy blocks the GPU. Pinning the
    # collated batch is cheap and lets us use non_blocking=True at the .to()
    # call to overlap copy with GPU compute.
    use_pin_memory = (device == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=effective_num_workers,
        collate_fn=neuron_collate_fn,
        pin_memory=use_pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        collate_fn=neuron_collate_fn,
        pin_memory=use_pin_memory,
    )

    # --- Model ---
    print("\n=== Model Setup ===")
    model, loss_fn = build_training_model(config, device)

    # --- Feature Precomputation ---
    feature_cache = None
    pos_enc = None

    # AMP setup (needed for both precomputation and training)
    use_amp = config.use_amp and device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    if config.data.precompute_features:
        print("\n=== Precomputing Image Encoder Features ===")
        feature_cache, pos_enc = precompute_all_features(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            image_size=config.data.image_size,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

    # --- Optimizer & Scheduler ---
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))

    # --- AMP ---
    # No GradScaler: we use bfloat16, which has fp32 exponent range and
    # therefore no underflow risk. GradScaler is only useful for fp16.
    # Keeping it would add per-step unscale/step/update syncs for no benefit.

    # --- Training loop ---
    precompute_label = " (precomputed features)" if feature_cache else ""
    print(f"\n=== Training ({config.num_epochs} epochs, "
          f"{len(train_loader)} steps/epoch{precompute_label}) ===")
    if feature_cache:
        print("  Using precomputed features (encoder skipped during training)")

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_iou = 0.0
    epochs_without_improvement = 0
    training_log = []
    start_epoch = 0

    # Resume from a saved training-state checkpoint if requested.
    # The saved file is at <checkpoint_dir>/train_state.pt (written at the
    # end of each epoch). Restoring it brings back the model weights,
    # optimiser moments, scheduler step position, best-IoU tracking, and
    # the early-stopping counter — so a fresh SLURM job continues exactly
    # where the previous one left off.
    if config.resume_from:
        resume_path = Path(config.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(
                f"--resume points to {resume_path}, but no file exists there."
            )
        print(f"\n=== Resuming from {resume_path} ===")
        resume = load_train_state(
            str(resume_path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        start_epoch = resume["start_epoch"]
        best_val_iou = resume["best_val_iou"]
        epochs_without_improvement = resume["epochs_without_improvement"]
        prev_total_epochs = resume["num_epochs"]
        prev_steps_per_epoch = resume["steps_per_epoch"]
        print(
            f"  Resumed at epoch {start_epoch}/{config.num_epochs}, "
            f"best_val_iou={best_val_iou:.4f}, "
            f"epochs_without_improvement={epochs_without_improvement}"
        )
        # Warn if the schedule parameters differ from the saved run. The
        # cosine LR schedule depends on (num_epochs × steps_per_epoch), so
        # changing either after resume will produce LR values that don't
        # match what the scheduler "thinks" it is at — usually harmless
        # for short extensions, but worth flagging.
        if prev_total_epochs and prev_total_epochs != config.num_epochs:
            print(
                f"  [warn] saved checkpoint targeted num_epochs="
                f"{prev_total_epochs}, this run uses {config.num_epochs}. "
                f"LR schedule curve will shift."
            )
        if prev_steps_per_epoch and prev_steps_per_epoch != len(train_loader):
            print(
                f"  [warn] saved checkpoint had steps_per_epoch="
                f"{prev_steps_per_epoch}, this run has {len(train_loader)}. "
                f"Dataset likely changed."
            )

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        # Drives the prompt-dropout curriculum and the distillation gate.
        # Called unconditionally on purpose: with a hasattr() guard, a model
        # that somehow lost set_epoch would silently pin current_epoch at 0,
        # which switches distillation off forever AND holds prompt dropout at
        # p=0 -- and because loss_distill_* is legitimately 0.000 before
        # distill_start_epoch, the log would look completely normal while both
        # features were dead. Fail loudly instead.
        model.set_epoch(epoch)
        if (epoch + 1) == model.distill_start_epoch and model.use_prompt_distill:
            print(f"  [distill] activating at epoch {epoch+1} "
                  f"(teacher is stationary by now); loss_distill_* should "
                  f"become non-zero from this epoch on")
        # Loss tensors stay on-device — no .item() inside the step loop.
        # We only sync at logging boundaries (every log_every_n_steps and at
        # end of epoch). Each .item() is a cudaStreamSynchronize, so per-step
        # syncs were the dominant CPU stall on B200.
        epoch_losses: List[torch.Tensor] = []
        epoch_components: Dict[str, List[torch.Tensor]] = {}
        step_times = []
        epoch_start = time.time()

        for step, (batch, frame_id_lists) in enumerate(train_loader):
            step_start = time.time()
            batch = batch.to(device, non_blocking=True)

            with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_amp):
                if feature_cache is not None:
                    # Feature-level flip augmentation (replaces pixel-level augmentation)
                    do_hflip = random.random() < config.augmentation.horizontal_flip_p
                    do_vflip = random.random() < config.augmentation.vertical_flip_p

                    backbone_out = assemble_cached_backbone(
                        feature_cache, pos_enc, frame_id_lists, device,
                        do_hflip=do_hflip, do_vflip=do_vflip,
                    )

                    # Flip masks to match flipped features
                    if do_hflip:
                        batch.masks = batch.masks.flip(-1)
                    if do_vflip:
                        batch.masks = batch.masks.flip(-2)

                    # Forward with precomputed features (skips image encoder)
                    outputs = model(batch, precomputed_backbone_out=backbone_out)
                else:
                    # Standard forward (runs image encoder every step)
                    outputs = model(batch)

                # Compute loss
                targets = batch.masks.float()
                losses = loss_fn(outputs, targets)
                loss = losses["core_loss"]

            # Per-component loss tracking (Hypothesis A: irreducible floor).
            # Store *detached tensors* — no sync. We aggregate at log boundaries.
            for k, v in losses.items():
                if k == "core_loss":
                    continue
                if isinstance(v, torch.Tensor):
                    epoch_components.setdefault(k, []).append(v.detach())

            # Backward — bf16 doesn't need GradScaler (no underflow risk like
            # fp16). Plain backward avoids the per-step unscale/step/update
            # syncs that GradScaler.step performs internally.
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.optimizer.max_grad_norm
            )
            optimizer.step()

            scheduler.step()
            epoch_losses.append(loss.detach())
            step_times.append(time.time() - step_start)

            # Memory cleanup for MPS to prevent OOM
            del outputs, losses, loss
            if device == "mps":
                torch.mps.empty_cache()

            # Logging — one sync per log_every_n_steps instead of per step.
            if (step + 1) % config.log_every_n_steps == 0:
                recent = epoch_losses[-config.log_every_n_steps:]
                avg_loss = float(torch.stack(recent).mean().item())
                avg_step_time = np.mean(step_times[-config.log_every_n_steps:])
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch+1}/{config.num_epochs}, "
                    f"Step {step+1}/{len(train_loader)}, "
                    f"Loss: {avg_loss:.4f}, "
                    f"LR: {lr:.2e}, "
                    f"{avg_step_time:.1f}s/step"
                )

        epoch_time = time.time() - epoch_start
        # Now sync once per metric: the loss/components lists hold detached
        # GPU tensors collected during the epoch. Stack + reduce + .item()
        # gives one cudaStreamSynchronize per epoch, not per step.
        if epoch_losses:
            avg_epoch_loss = float(torch.stack(epoch_losses).mean().item())
        else:
            avg_epoch_loss = 0.0
        avg_step_time_epoch = np.mean(step_times) if step_times else 0
        train_components = {
            k: float(torch.stack(v).mean().item())
            for k, v in epoch_components.items()
            if v
        }
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_epoch_loss,
            "time_seconds": epoch_time,
            "sec_per_step": avg_step_time_epoch,
            "train_loss_components": train_components,
        }

        print(
            f"Epoch {epoch+1}/{config.num_epochs}: "
            f"train_loss={avg_epoch_loss:.4f}, "
            f"time={epoch_time:.1f}s, "
            f"{avg_step_time_epoch:.1f}s/step"
        )
        if train_components:
            comp_str = ", ".join(
                f"{k.replace('loss_','')}={v:.3f}"
                for k, v in train_components.items()
            )
            print(f"  train_loss_components: {comp_str}")

        # --- Validation ---
        if (epoch + 1) % config.val_every_n_epochs == 0 and len(val_dataset) > 0:
            print("  Validating...")
            sample_dir = checkpoint_dir / "val_samples" / f"epoch_{epoch+1:03d}"
            val_metrics = validate_epoch(
                model, val_loader, loss_fn, device,
                max_batches=config.val_max_batches,
                feature_cache=feature_cache,
                pos_enc=pos_enc,
                sample_dump_dir=sample_dir,
                sample_dump_tag=f"e{epoch+1:03d}",
            )
            log_entry.update(val_metrics)
            print(
                f"  val_loss={val_metrics['val_loss']:.4f}, "
                f"val_iou={val_metrics['val_iou']:.4f}, "
                f"val_dice={val_metrics['val_dice']:.4f}"
            )
            # Topology quality (lower = better): mean |Δβ0| (component breaks/
            # merges) and |Δβ1| (hole errors) vs. GT — what IoU/Dice miss.
            b0, b1 = val_metrics.get("val_betti0"), val_metrics.get("val_betti1")
            if b0 is not None and not np.isnan(b0):
                print(f"  val_betti_err: b0={b0:.3f}, b1={b1:.3f}")
            # Hypothesis A: per-component val loss (compare floors vs. train).
            if val_metrics.get("loss_components"):
                comp_str = ", ".join(
                    f"{k.replace('loss_','')}={v:.3f}"
                    for k, v in val_metrics["loss_components"].items()
                )
                print(f"  val_loss_components: {comp_str}")
            # Hypothesis B: chain error along the propagation.
            pf_iou = val_metrics.get("per_frame_iou", [])
            if pf_iou:
                fmt = ", ".join(
                    f"t{t}={v:.3f}" if not np.isnan(v) else f"t{t}=nan"
                    for t, v in enumerate(pf_iou)
                )
                print(f"  per_frame_iou: {fmt}")
            # Hypothesis C: variance across val videos.
            pv_iou = val_metrics.get("per_video_iou", {})
            if pv_iou:
                fmt = ", ".join(f"{n}={v:.3f}" for n, v in pv_iou.items())
                print(f"  per_video_iou: {fmt}")

            # Save best checkpoint and track early stopping
            if val_metrics["val_iou"] > best_val_iou:
                best_val_iou = val_metrics["val_iou"]
                epochs_without_improvement = 0
                save_lora_weights(model, checkpoint_dir / "lora_weights.pt")
                print(f"  New best IoU: {best_val_iou:.4f} - saved checkpoint")
            else:
                epochs_without_improvement += 1
                print(f"  No improvement for {epochs_without_improvement} validation(s)")

        training_log.append(log_entry)

        # Save training log
        with open(checkpoint_dir / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2, default=str)

        # Save full training state at the end of every epoch so the run can
        # resume from a fresh SLURM job (4-hour queue limit). This includes
        # optimiser moments + scheduler step + early-stop counter, which
        # lora_weights.pt does NOT — that one is inference-only.
        save_train_state(
            str(checkpoint_dir / "train_state.pt"),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_iou=best_val_iou,
            epochs_without_improvement=epochs_without_improvement,
            num_epochs=config.num_epochs,
            steps_per_epoch=len(train_loader),
        )

        # Early stopping check
        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"No improvement in val IoU for {config.early_stopping_patience} validation checks")
            break

    # Save final checkpoint if not saving best only
    if not config.save_best_only:
        save_lora_weights(model, checkpoint_dir / "lora_weights_final.pt")

    # Save config
    import dataclasses

    with open(checkpoint_dir / "training_config.json", "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)

    print(f"\nTraining complete! Best val IoU: {best_val_iou:.4f}")
    print(f"Checkpoint saved to: {checkpoint_dir / 'lora_weights.pt'}")

    return best_val_iou
