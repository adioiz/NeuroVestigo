"""
sam3_backbone.py -- train the SAM 3 tracker with the same dual-rank LoRA+
recipe used for SAM 2.1.

WHICH SAM 3: build_sam3_video_model() loads facebook/sam3/sam3.pt -- SAM 3,
not 3.1 (model_builder.py:797, download_ckpt_from_hf(version="sam3")). The
3.1 release ships only sam3.1_multiplex.pt, whose tracker is the multiplex
architecture and does not fit Sam3TrackerPredictor (maskmem_tpos_enc and 25
other keys change shape). So every row produced through this path -- the
mask-prompt baseline and any LoRA run -- is SAM 3, and must be labelled so.

WHY THIS CAN WORK AT ALL
------------------------
SAM 3's own paper: "The tracker inherits the SAM 2 transformer encoder-decoder
architecture." So sam_mask_decoder / memory_attention exist there too, and the
LoRA recipe transfers in principle.

WHY IT IS NOT A DROP-IN
-----------------------
Two structural differences, both handled here:

1. The vision encoder is SHARED with the detector, so build_tracker leaves
   `backbone=None` and the caller attaches it -- exactly as Meta's own
   sam3_for_sam2_video_task_example.ipynb (cell 11) does:

       sam3_model = build_sam3_video_model()
       predictor  = sam3_model.tracker
       predictor.backbone = sam3_model.detector.backbone

2. SAM 2 exposes `model.forward_image` and `model.image_encoder`; the SAM 3
   tracker calls `self.backbone.forward_image`. SAM2TrainingWrapper reaches for
   ten SAM 2 attributes (see REQUIRED_API), so a thin delegating adapter
   supplies the aliases and passes everything else straight through.

LoRA targets are DISCOVERED from the real module names rather than copied from
the SAM 2 regexes. If SAM 3 renamed a submodule, a hardcoded pattern would
silently match nothing and "train" zero adapters while the loss slowly went
nowhere -- a failure that looks exactly like a bad hyperparameter.

    python benchmark/sam3_lora_preflight.py     # check BEFORE a long run
"""
from __future__ import annotations

import re
from typing import List, Tuple

import torch
import torch.nn as nn

def _shadowing_entries(path_entries):
    """sys.path entries that would make `import sam3` a NAMESPACE package.

    The SAM 3 clone lives at <repo>/sam3/, and the real package is one level
    down at <repo>/sam3/sam3/. Any sys.path entry whose `sam3/` child has no
    __init__.py -- the repo root, or '' / '.' when the cwd is the repo root --
    is scanned by PathFinder before the editable-install meta-path finder
    gets a turn, so Python builds a namespace package from it. Submodules
    still resolve through the editable finder, which is why
    `from sam3.model_builder import ...` works, but `sam3.__file__` is None
    and the builder's pkg_resources.resource_filename("sam3", <bpe file>)
    dies with

        TypeError: expected str, bytes or os.PathLike object, not NoneType

    That stopped the preflight (it inserted REPO into sys.path), and it is
    what `python cli.py train --backbone sam3` hits too, since running a
    script puts its directory at sys.path[0].
    """
    import os
    bad = []
    for p in path_entries:
        base = p if p not in ("", ".") else os.getcwd()
        cand = os.path.join(base, "sam3")
        if os.path.isdir(cand) and not os.path.exists(
                os.path.join(cand, "__init__.py")):
            bad.append(p)
    return bad


def import_sam3():
    """Import the real sam3 package even when the repo root is on sys.path."""
    import importlib
    import sys

    mod = sys.modules.get("sam3")
    if mod is not None and getattr(mod, "__file__", None):
        return mod                                   # already the real one
    if mod is not None:
        # A namespace shadow is cached; drop it and every submodule imported
        # under it so the re-import starts clean.
        for k in [k for k in sys.modules if k == "sam3" or k.startswith("sam3.")]:
            del sys.modules[k]
    shadow = _shadowing_entries(sys.path)
    saved = list(sys.path)
    try:
        sys.path[:] = [p for p in sys.path if p not in shadow]
        mod = importlib.import_module("sam3")
    finally:
        sys.path[:] = saved
    if not getattr(mod, "__file__", None):
        raise ImportError(
            "sam3 still imports as a namespace package (no __file__). "
            f"Hid shadowing sys.path entries {shadow}; is sam3 installed in "
            "this environment (pip install -e sam3)?")
    if shadow:
        print(f"  [sam3] real package at {mod.__file__} "
              f"(hid {len(shadow)} shadowing sys.path entr"
              f"{'y' if len(shadow) == 1 else 'ies'}: {shadow})")
    return mod


def make_vision_backbone_trainable():
    """Replace SAM 3's inference-only fused MLP with the equivalent unfused op.

    sam3/model/vitdet.py:71 routes every ViT MLP through
    sam3/perflib/fused.py:addmm_act, which detaches the weights, casts to
    bfloat16 and calls aten._addmm_activation -- and raises

        ValueError: Expected grad to be disabled.

    whenever autograd is on. The released backbone therefore cannot appear in
    ANY training forward, frozen or not, and the wrapper's forward_image runs
    under grad. The op computes act(x @ W^T + b), nothing more, so when grad
    is enabled fall back to exactly that through the module's own Linear and
    activation (LoRA-wrapped fc1 included). Inference keeps the fused path.
    Patched on the vitdet module, which is where the name is bound.
    """
    import sam3.model.vitdet as vitdet
    import sam3.perflib.fused as fused
    if getattr(vitdet.addmm_act, "_trainable_patch", False):
        return
    fused_fn = fused.addmm_act

    def addmm_act(activation, linear, mat1):
        # The fused op reads linear.weight / linear.bias directly. A LoRALinear
        # wraps those under .original, so it must take the unfused branch even
        # under no_grad -- otherwise the first validate_epoch, which runs
        # @torch.no_grad(), dies with "'LoRALinear' object has no attribute
        # 'bias'" after epoch 1. Only a plain nn.Linear may use the fused op.
        if not torch.is_grad_enabled() and type(linear) is nn.Linear:
            return fused_fn(activation, linear, mat1)
        act = activation() if isinstance(activation, type) else activation
        return act(linear(mat1))

    addmm_act._trainable_patch = True
    vitdet.addmm_act = addmm_act
    print("  [sam3] vitdet MLP: unfused act(fc1(x)) when grad is enabled "
          "(fused inference path kept)")


# What SAM2TrainingWrapper actually touches on the model.
REQUIRED_API = [
    "forward_image", "_prepare_backbone_features", "sam_prompt_encoder",
    "sam_mask_decoder", "track_step", "_forward_sam_heads", "no_mem_embed",
    "hidden_dim", "pred_obj_scores", "use_mask_input_as_output_without_sam",
]

# Leaf names worth adapting, mirroring SAM2_LORA_TARGETS in lora.py.
_LEAVES = ("q_proj", "k_proj", "v_proj", "out_proj", "linear1", "linear2",
           "qkv", "proj")


# Top-level submodules that carry the adapted attention, as PREFIXES.
# Verified against sam3/model/sam3_tracker_base.py:
#   self.sam_mask_decoder = MaskDecoder(...)      line 192  -- same as SAM 2
#   self.transformer      = transformer           line 89   -- SAM 2 calls this
#                                                              memory_attention
# The layer names underneath are identical to SAM 2's (encoder.py:59-65:
# self_attn / cross_attn_image / linear1 / linear2; sam/transformer.py:212-215:
# q_proj / k_proj / v_proj / out_proj). Only the top-level name differs, and
# an include list of "memory_attention" would have matched nothing there --
# wrapping the decoder only and training with the memory path unadapted while
# looking completely normal.
TRACKER_INCLUDE = ("sam_mask_decoder.", "transformer.", "memory_attention.")


def discover_targets(model: nn.Module,
                     include: Tuple[str, ...] = TRACKER_INCLUDE,
                     verbose: bool = True) -> List[str]:
    """Exact names of the Linear layers to wrap, as escaped regexes.

    Discovery rather than hardcoded patterns: a SAM 2 regex that matches
    nothing under SAM 3 would wrap zero layers and still train happily on an
    empty parameter set. Returning real names makes that impossible -- an empty
    list here is visible immediately. `include` entries are matched as
    PREFIXES of the qualified name, so "transformer." means the tracker's
    memory attention and not the decoder's internal
    sam_mask_decoder.transformer (already covered by its own prefix).
    """
    found = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if not any(name.startswith(k) for k in include):
            continue
        if name.rsplit(".", 1)[-1] in _LEAVES:
            found.append(name)
    if verbose:
        by_group = {}
        for n in found:
            key = next((k for k in include if n.startswith(k)), "other")
            by_group[key] = by_group.get(key, 0) + 1
        print(f"  discovered {len(found)} Linear layers to adapt: {by_group}")
        for n in found[:4]:
            print(f"    e.g. {n}")
        groups_hit = [k for k in include if by_group.get(k)]
        if len(groups_hit) < 2:
            print("  WARNING: only one module group matched -- SAM 2's recipe "
                  "adapts BOTH the decoder and the memory attention")
    return [re.escape(n) for n in found]


def discover_visual_targets(backbone: nn.Module, n_adapt: int,
                            verbose: bool = True) -> List[str]:
    """Last `n_adapt` blocks of the shared vision trunk.

    The SAM 2 recipe adapts Hiera blocks 10-11. SAM 3's visual tower is a
    different family, so find its block list instead of assuming Hiera's shape.
    Returns [] when no block list is found -- the caller then trains the
    decoder/memory side only, which is still a valid experiment.
    """
    # The shared SAM3VLBackbone holds a VISION trunk (vitdet.py:899,
    # self.blocks = nn.ModuleList()) and may hold a TEXT encoder with its own
    # block list. Rank candidates so the vision trunk wins even if the text
    # encoder is registered first; never adapt the text side.
    cands = []
    for name, mod in backbone.named_modules():
        blocks = getattr(mod, "blocks", None)
        if isinstance(blocks, (nn.ModuleList, nn.Sequential)) and len(blocks) > 4:
            low = name.lower()
            if "text" in low or "language" in low or "bpe" in low:
                continue
            score = 2 if any(k in low for k in ("visual", "vision", "trunk")) else 1
            cands.append((score, name, blocks))
    holder, path = None, None
    if cands:
        cands.sort(key=lambda t: -t[0])
        _, name, holder = cands[0]
        path = f"{name}.blocks" if name else "blocks"
    if holder is None:
        if verbose:
            print("  no vision-trunk block list found; skipping visual LoRA")
        return []
    total = len(holder)
    idxs = range(max(0, total - n_adapt), total)
    names = []
    # vitdet blocks: attn.qkv / attn.proj and mlp.fc1 / mlp.fc2 -- the same
    # four per block the Hiera recipe adapts (qkv, proj, mlp.layers.0/1).
    vis_leaves = _LEAVES + ("fc1", "fc2")
    for i in idxs:
        for nm, mod in holder[i].named_modules():
            if isinstance(mod, nn.Linear) and nm.rsplit(".", 1)[-1] in vis_leaves:
                names.append(f"{path}.{i}.{nm}")
    if verbose:
        print(f"  vision trunk '{path}' has {total} blocks; adapting the last "
              f"{len(list(idxs))} -> {len(names)} Linear layers")
    return [re.escape(n) for n in names]


def _prepare_tracker_for_training(tracker: nn.Module) -> None:
    """Supply what Meta's training framework sets and the model does not.

    Every item below was found by reading sam3/model/sam3_tracker_base.py for
    attributes that are READ under `self.training` but never ASSIGNED by the
    constructor. Inference never touches them, which is why the mask-prompt
    run worked and a training forward would not have:

      teacher_force_obj_scores_for_mem   line 333   defined nowhere in sam3
      prob_to_dropout_spatial_mem        line 684   defined nowhere in sam3
      rng                                line 685   only if the prob is > 0

    And the obj-score rule. SAM 2 gates "no object -> replace the mask with
    NO_OBJ_SCORE" on self.pred_obj_scores, which the wrapper sets False because
    an obj-score head that says 'absent' zeroes the gradient to the decoder
    (trainer.py:303-327). SAM 3 applies the rule unconditionally, with one
    alternative branch: in training with teacher forcing it takes object
    presence from `gt_masks` instead of the head (line 333-337). The neuron is
    present in every frame we train on, so an all-ones presence tensor is the
    faithful teacher signal, and it reproduces exactly what pred_obj_scores=
    False achieved under SAM 2. _forward_sam_heads is patched on the INSTANCE
    because track_step calls self._forward_sam_heads on the tracker itself --
    an override on the adapter would only catch the wrapper's direct call.
    Applied in eval too, matching the SAM 2 setup where the flag was permanent.
    """
    import types

    tracker.prob_to_dropout_spatial_mem = 0.0     # no spatial-memory dropout
    tracker.teacher_force_obj_scores_for_mem = True

    orig = tracker._forward_sam_heads

    def _forward_sam_heads(self, backbone_features, point_inputs=None,
                           mask_inputs=None, high_res_features=None,
                           multimask_output=False, gt_masks=None):
        if gt_masks is None:
            gt_masks = torch.ones(backbone_features.shape[0], 1,
                                  device=backbone_features.device)
        was_training = self.training
        self.training = True           # this module's flag only; line 333
        try:
            return orig(backbone_features, point_inputs, mask_inputs,
                        high_res_features, multimask_output, gt_masks)
        finally:
            self.training = was_training

    tracker._forward_sam_heads = types.MethodType(_forward_sam_heads, tracker)


class Sam3TrainingAdapter(nn.Module):
    """SAM 2's model surface over SAM 3's tracker.

    Everything not named here is delegated unchanged, so this stays a thin
    rename layer rather than a reimplementation. Each alias is reported by
    `api_report()` so a missing one is never silently papered over.
    """

    def __init__(self, tracker: nn.Module, backbone: nn.Module):
        super().__init__()
        self.tracker = tracker
        self.tracker.backbone = backbone
        _prepare_tracker_for_training(self.tracker)

        # Two SAM 2 flags the wrapper SETS (trainer.py:301, :327) that SAM 3's
        # tracker does not have. Both are plain attributes here so the wrapper's
        # assignment succeeds, but neither changes SAM 3's behaviour:
        #
        #   use_mask_input_as_output_without_sam -- SAM 3's track_step routes
        #     mask_inputs straight to _use_mask_as_output unconditionally
        #     (sam3_tracker_base.py:961). The wrapper sets this False so the
        #     decoder is exercised on frame 0; under SAM 3 frame 0 therefore
        #     echoes the GT mask and contributes no decoder gradient. Frames
        #     t>=1 go through the memory path as usual.
        #
        #   pred_obj_scores -- SAM 3's _forward_sam_heads applies the
        #     "no object -> empty mask" rule unconditionally (line 339-346).
        #     The wrapper sets this False to disable that rule; here it is
        #     disabled instead by _prepare_tracker_for_training, using SAM 3's
        #     own training-time mechanism (teacher-forced object presence).
        self.use_mask_input_as_output_without_sam = True
        self.pred_obj_scores = True

    # -- the two aliases SAM 3 genuinely renames --------------------------
    @property
    def image_encoder(self):
        # trainer.py freezes model.image_encoder.parameters()
        return self.tracker.backbone

    def __setattr__(self, name, value):
        # config.compile_encoder assigns torch.compile(...) back onto
        # model.image_encoder (trainer.py). A property setter does NOT catch
        # that: nn.Module.__setattr__ registers any Module-typed value straight
        # into _modules without consulting descriptors, which would leave the
        # real backbone untouched and add a dangling second copy of it to the
        # parameter count. Route the name explicitly.
        if name == "image_encoder":
            self.tracker.backbone = value
            return
        if name == "sam_mask_decoder" and "tracker" in self.__dict__.get("_modules", {}):
            # config.compile_decoder assigns torch.compile(...) here too; an
            # assignment on the adapter would leave the tracker's own decoder
            # uncompiled AND make named_modules() list every decoder LoRALinear
            # twice (save_lora_weights would write duplicates).
            self.tracker.sam_mask_decoder = value
            return
        super().__setattr__(name, value)

    def forward_image(self, img_batch):
        # Sam3TrackerBase.forward_image exists (line 442) and already does
        # self.backbone.forward_image(img)["sam2_backbone_out"] plus the
        # conv_s0/conv_s1 projection -- the SAM 2 contract. Fall back only if
        # a build lacks it.
        fn = getattr(self.tracker, "forward_image", None)
        if callable(fn):
            return fn(img_batch)
        return self.tracker.backbone.forward_image(img_batch)["sam2_backbone_out"]

    @property
    def memory_attention(self):
        # SAM 2 name for what SAM 3 calls self.transformer (tracker_base:89).
        return self.tracker.transformer

    def track_step(self, *args, **kwargs):
        """SAM 2's signature, forwarded to SAM 3's.

        SAM 3 adds a required positional `image` (sam3_tracker_base.py:929-
        936) between feat_sizes and point_inputs. The wrapper calls by keyword
        without it. It is only ever consumed by a non-Simple memory encoder
        (line 843, self.maskmem_backbone(image, ...)); build_tracker installs
        SimpleMaskEncoder (model_builder.py:370), whose branch ignores it
        (line 833-836). So None is correct, and the assertion below makes
        that assumption fail loudly rather than silently if a build changes.
        """
        enc = getattr(self.tracker, "maskmem_backbone", None)
        assert enc is None or type(enc).__name__ == "SimpleMaskEncoder", (
            "track_step shim passes image=None, which is only valid for "
            f"SimpleMaskEncoder; this tracker has {type(enc).__name__}")
        kwargs.setdefault("image", None)
        return self.tracker.track_step(*args, **kwargs)

    # -- everything else is the tracker's own -----------------------------
    def __getattr__(self, name):
        # Overriding __getattr__ REPLACES nn.Module's, which is what resolves
        # registered submodules, parameters and buffers out of _modules /
        # _parameters / _buffers. The first version skipped that step, so
        # `self.tracker` itself fell through to delegation and asked the
        # tracker for an attribute named 'tracker':
        #     AttributeError: 'Sam3TrackerPredictor' object has no attribute
        #                     'tracker'
        # Defer to the base class first; delegate only on a genuine miss.
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        tracker = self.__dict__.get("_modules", {}).get("tracker")
        if tracker is None:
            raise AttributeError(name)
        return getattr(tracker, name)

    def forward(self, *a, **k):
        return self.tracker(*a, **k)

    def api_report(self):
        """Which of the ten SAM 2 attributes resolve, and which do not."""
        rows = []
        for name in REQUIRED_API:
            try:
                rows.append((name, getattr(self, name) is not None))
            except AttributeError:
                rows.append((name, False))
        return rows


def build_sam3_lora(config, verbose: bool = True):
    """Meta's cell-11 construction, then our LoRA. Returns (model, info).

    Loads SAM 3 (sam3.pt), not 3.1 -- see the module docstring.
    """
    import_sam3()        # BEFORE the submodule import -- see _shadowing_entries
    make_vision_backbone_trainable()
    from sam3.model_builder import build_sam3_video_model
    from src.lora import apply_lora

    if verbose:
        print("\nBuilding SAM 3 tracker (build_sam3_video_model -> facebook/sam3/sam3.pt)...")
    sam3_model = build_sam3_video_model()
    model = Sam3TrainingAdapter(sam3_model.tracker, sam3_model.detector.backbone)

    if config.freeze_image_encoder:
        for p in model.image_encoder.parameters():
            p.requires_grad = False
        if verbose:
            print("  shared vision backbone frozen")

    # SAM 3's ViT is built with drop_path_rate=0.1 (vitdet.py:776). Under
    # wrapper.train() that stochastic depth fires on a FROZEN backbone every
    # training forward while validation is deterministic -- noise the Hiera
    # recipe (drop_path 0.0) never had. Turn it off so the backbone behaves the
    # same way in both modes, as it does at inference.
    n_dp = 0
    for m in model.image_encoder.modules():
        if type(m).__name__ == "DropPath" and getattr(m, "drop_prob", 0):
            m.drop_prob = 0.0
            n_dp += 1
    if verbose and n_dp:
        print(f"  stochastic depth disabled on {n_dp} DropPath modules")

    # torch.compile routing does not reach through the adapter usefully: the
    # tracker calls backbone.forward_image, and an OptimizedModule only
    # compiles .forward. Say so rather than let the flags silently no-op.
    for flag in ("compile_encoder", "compile_decoder"):
        if getattr(config, flag, False):
            print(f"  NOTE: {flag}=True has no effect with backbone=sam3; disabling")
            setattr(config, flag, False)

    targets = discover_targets(model.tracker, verbose=verbose)
    if not targets:
        raise RuntimeError(
            "no sam_mask_decoder / memory_attention Linear layers found on the "
            "SAM 3 tracker. Training would adapt nothing. Run "
            "benchmark/sam3_lora_preflight.py and read the module dump.")
    wrapped = apply_lora(model.tracker, target_patterns=targets,
                         rank=config.lora.rank, alpha=config.lora.alpha,
                         dropout=config.lora.dropout)
    if verbose:
        print(f"  wrapped {len(wrapped)} layers at rank {config.lora.rank}")

    wrapped_vis = []
    if config.lora.hiera_last_stage:
        vis = discover_visual_targets(model.image_encoder,
                                      config.lora.hiera_blocks, verbose=verbose)
        if vis:
            wrapped_vis = apply_lora(
                model.image_encoder, target_patterns=vis,
                rank=config.lora.hiera_rank or config.lora.rank,
                alpha=config.lora.hiera_alpha or config.lora.alpha,
                dropout=config.lora.dropout)
            if verbose:
                print(f"  wrapped {len(wrapped_vis)} vision layers at rank "
                      f"{config.lora.hiera_rank or config.lora.rank}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    info = {"wrapped": len(wrapped), "wrapped_visual": len(wrapped_vis),
            "trainable": trainable, "total": total}
    if verbose:
        print(f"  trainable {trainable:,} / {total:,} "
              f"({100.0 * trainable / max(total, 1):.1f}%)")
    return model, info
