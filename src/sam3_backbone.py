"""
sam3_backbone.py -- train the SAM 3.1 tracker with the same dual-rank LoRA+
recipe used for SAM 2.1.

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


# What SAM2TrainingWrapper actually touches on the model.
REQUIRED_API = [
    "forward_image", "_prepare_backbone_features", "sam_prompt_encoder",
    "sam_mask_decoder", "track_step", "_forward_sam_heads", "no_mem_embed",
    "hidden_dim", "pred_obj_scores", "use_mask_input_as_output_without_sam",
]

# Leaf names worth adapting, mirroring SAM2_LORA_TARGETS in lora.py.
_LEAVES = ("q_proj", "k_proj", "v_proj", "out_proj", "linear1", "linear2",
           "qkv", "proj")


def discover_targets(model: nn.Module,
                     include: Tuple[str, ...] = ("sam_mask_decoder",
                                                 "memory_attention"),
                     verbose: bool = True) -> List[str]:
    """Exact names of the Linear layers to wrap, as escaped regexes.

    Discovery rather than hardcoded patterns: a SAM 2 regex that matches
    nothing under SAM 3 would wrap zero layers and still train happily on an
    empty parameter set. Returning real names makes that impossible -- an empty
    list here is visible immediately.
    """
    found = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if not any(k in name for k in include):
            continue
        if name.rsplit(".", 1)[-1] in _LEAVES:
            found.append(name)
    if verbose:
        by_group = {}
        for n in found:
            key = next((k for k in include if k in n), "other")
            by_group[key] = by_group.get(key, 0) + 1
        print(f"  discovered {len(found)} Linear layers to adapt: {by_group}")
        for n in found[:6]:
            print(f"    e.g. {n}")
    return [re.escape(n) for n in found]


def discover_visual_targets(backbone: nn.Module, n_adapt: int,
                            verbose: bool = True) -> List[str]:
    """Last `n_adapt` blocks of the shared vision trunk.

    The SAM 2 recipe adapts Hiera blocks 10-11. SAM 3's visual tower is a
    different family, so find its block list instead of assuming Hiera's shape.
    Returns [] when no block list is found -- the caller then trains the
    decoder/memory side only, which is still a valid experiment.
    """
    holder, path = None, None
    for name, mod in backbone.named_modules():
        blocks = getattr(mod, "blocks", None)
        if isinstance(blocks, (nn.ModuleList, nn.Sequential)) and len(blocks) > 4:
            holder, path = blocks, f"{name}.blocks" if name else "blocks"
            break
    if holder is None:
        if verbose:
            print("  no vision-trunk block list found; skipping visual LoRA")
        return []
    total = len(holder)
    idxs = range(max(0, total - n_adapt), total)
    names = []
    for i in idxs:
        for nm, mod in holder[i].named_modules():
            if isinstance(mod, nn.Linear) and nm.rsplit(".", 1)[-1] in _LEAVES:
                names.append(f"{path}.{i}.{nm}")
    if verbose:
        print(f"  vision trunk '{path}' has {total} blocks; adapting the last "
              f"{len(list(idxs))} -> {len(names)} Linear layers")
    return [re.escape(n) for n in names]


class Sam3TrainingAdapter(nn.Module):
    """SAM 2's model surface over SAM 3.1's tracker.

    Everything not named here is delegated unchanged, so this stays a thin
    rename layer rather than a reimplementation. Each alias is reported by
    `api_report()` so a missing one is never silently papered over.
    """

    def __init__(self, tracker: nn.Module, backbone: nn.Module):
        super().__init__()
        self.tracker = tracker
        self.tracker.backbone = backbone

    # -- the two aliases SAM 3 genuinely renames --------------------------
    @property
    def image_encoder(self):
        # trainer.py freezes model.image_encoder.parameters()
        return self.tracker.backbone

    @image_encoder.setter
    def image_encoder(self, value):
        # config.compile_encoder assigns torch.compile(...) back onto this
        # name (trainer.py). Without a setter that raises AttributeError and
        # the run dies after the model is already on the GPU.
        self.tracker.backbone = value

    def forward_image(self, img_batch):
        fn = getattr(self.tracker, "forward_image", None)
        if callable(fn):
            return fn(img_batch)
        return self.tracker.backbone.forward_image(img_batch)

    # -- everything else is the tracker's own -----------------------------
    def __getattr__(self, name):
        # nn.Module.__getattr__ runs first for registered submodules; this is
        # only reached for genuine misses.
        try:
            tracker = object.__getattribute__(self, "_modules")["tracker"]
        except KeyError:
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
    """Meta's cell-11 construction, then our LoRA. Returns (model, info)."""
    import_sam3()        # BEFORE the submodule import -- see _shadowing_entries
    from sam3.model_builder import build_sam3_video_model
    from src.lora import apply_lora

    if verbose:
        print("\nBuilding SAM 3.1 tracker (build_sam3_video_model)...")
    sam3_model = build_sam3_video_model()
    model = Sam3TrainingAdapter(sam3_model.tracker, sam3_model.detector.backbone)

    if config.freeze_image_encoder:
        for p in model.image_encoder.parameters():
            p.requires_grad = False
        if verbose:
            print("  shared vision backbone frozen")

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
