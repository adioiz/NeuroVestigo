"""The adapter against the REAL Sam3TrackerPredictor -- no checkpoint needed.

build_tracker(with_backbone=True) constructs the genuine classes at random
init: Sam3TrackerBase, MaskDecoder, TwoWayTransformer, the memory
TransformerEncoder, SimpleMaskEncoder and the ViT trunk. Attribute names,
signatures and the training-only code paths are therefore the ones the cluster
runs, which is what the stub-based tests could never guarantee. Every one of
the earlier cluster failures would have been caught here first.

Runs anywhere sam3 is importable:
  * cluster (sam3-env, CUDA)     -- as is
  * a CPU box without triton     -- a stand-in for triton's decorators is
                                    injected, and sam3's hardcoded
                                    device="cuda" precomputes are redirected
Skips, not fails, when sam3 is not installed.

    python tests/test_sam3_real_classes.py
"""
import os
import sys
import time
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch  # noqa: E402


def _stub_triton():
    """sam3 imports triton at module level only to DEFINE GPU kernels."""
    try:
        import triton  # noqa: F401
        return
    except ImportError:
        pass
    import torch._dynamo, torch._inductor  # noqa: E401,F401  (probe before stub)
    try:
        import torch._inductor.runtime.hints  # noqa: F401
    except Exception:
        pass
    tri = types.ModuleType("triton")
    tri.jit = lambda fn=None, **kw: (fn if fn is not None else (lambda f: f))
    tri.autotune = lambda *a, **k: (lambda f: f)
    tri.heuristics = lambda *a, **k: (lambda f: f)
    tri.Config = type("Config", (), {"__init__": lambda self, *a, **k: None})
    tri.cdiv = lambda a, b: -(-a // b)

    class _TL(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return lambda *a, **k: None
    tl = _TL("triton.language")
    for t in ("constexpr", "float32", "int32", "int64", "float16", "bfloat16"):
        setattr(tl, t, object)
    tri.language = tl
    sys.modules["triton"], sys.modules["triton.language"] = tri, tl


def _cpu_redirect():
    """sam3 hardcodes device="cuda" in a few precomputes (position_encoding.py:55)."""
    if torch.cuda.is_available():
        return

    def wrap(fn):
        def w(*a, **k):
            d = k.get("device")
            if d is not None and str(d).startswith("cuda"):
                k["device"] = "cpu"
            return fn(*a, **k)
        return w
    for n in ("zeros", "ones", "arange", "empty", "full", "tensor", "linspace", "randn"):
        setattr(torch, n, wrap(getattr(torch, n)))
    torch.Tensor.cuda = lambda self, *a, **k: self
    # _get_tpos_enc does torch.tensor(...).pin_memory().to(device); on a Mac
    # with MPS, pin_memory pins to MPS and the .to("cpu") then refuses.
    torch.Tensor.pin_memory = lambda self, *a, **k: self
    _to = torch.Tensor.to

    def to(self, *a, **k):
        a = tuple("cpu" if str(x).startswith("cuda") else x for x in a)
        if str(k.get("device", "")).startswith("cuda"):
            k["device"] = "cpu"
        return _to(self, *a, **k)
    torch.Tensor.to = to


def main():
    _stub_triton()
    _cpu_redirect()
    try:
        from src.sam3_backbone import import_sam3
        import_sam3()
    except ImportError as e:
        print(f"SKIP: sam3 not importable ({e})")
        return 0
    from sam3.model_builder import build_tracker
    from src.sam3_backbone import (Sam3TrainingAdapter, discover_targets,
                                   discover_visual_targets,
                                   make_vision_backbone_trainable)
    from src.lora import apply_lora
    from src.config import TrainingConfig
    from src.trainer import SAM2TrainingWrapper
    from src.data_utils import BatchedVideoDatapoint

    make_vision_backbone_trainable()
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    tracker = build_tracker(apply_temporal_disambiguation=False, with_backbone=True)
    backbone, tracker.backbone = tracker.backbone, None   # as build_sam3_video_model
    model = Sam3TrainingAdapter(tracker, backbone)
    print(f"built real Sam3TrackerPredictor in {time.time() - t0:.0f}s")

    report = model.api_report()
    missing = [n for n, ok in report if not ok]
    assert not missing, f"wrapper attributes missing on the real tracker: {missing}"
    print(f"PASS: all {len(report)} wrapper attributes resolve")

    cfg = TrainingConfig()
    targets = discover_targets(model.tracker, verbose=False)
    names = [t.replace("\\", "") for t in targets]
    n_dec = sum(n.startswith("sam_mask_decoder.") for n in names)
    n_mem = sum(n.startswith("transformer.") for n in names)
    assert n_dec == 28 and n_mem == 40, (n_dec, n_mem)   # == SAM 2's 68
    wrapped = apply_lora(model.tracker, target_patterns=targets, rank=cfg.lora.rank,
                         alpha=cfg.lora.alpha, dropout=cfg.lora.dropout)
    assert len(wrapped) == 68, len(wrapped)
    print("PASS: 68 tracker layers discovered and wrapped (40 memory + 28 decoder)")

    for p in model.image_encoder.parameters():
        p.requires_grad = False
    vis = discover_visual_targets(model.image_encoder, cfg.lora.hiera_blocks, verbose=False)
    assert len(vis) == 8, len(vis)                        # 2 blocks x (qkv, proj, fc1, fc2)
    apply_lora(model.image_encoder, target_patterns=vis, rank=cfg.lora.hiera_rank,
               alpha=cfg.lora.hiera_alpha, dropout=cfg.lora.dropout)
    lora = [(n, p) for n, p in model.named_parameters()
            if p.requires_grad and (".lora_A" in n or ".lora_B" in n)]
    assert len(lora) == 152, len(lora)                    # == SAM 2's 152
    print("PASS: 8 visual layers wrapped; 152 LoRA tensors, same as the SAM 2 recipe")

    wrapper = SAM2TrainingWrapper(model, num_frames=2, use_memory_path=True,
                                  prompt_dropout=0.0, prompt_dropout_warmup_epochs=0,
                                  use_prompt_distill=False, distill_frames_per_clip=0,
                                  distill_start_epoch=10 ** 9).to(dev)
    wrapper.train()
    sz = int(model.tracker.image_size)
    assert sz == 1008, sz
    images = torch.randn(1, 2, 3, sz, sz, device=dev)
    masks = torch.zeros(1, 2, 1, sz, sz, device=dev)
    masks[..., sz // 3: 2 * sz // 3, sz // 3: 2 * sz // 3] = 1.0
    batch = BatchedVideoDatapoint(images=images, masks=masks, obj_ids=[[1]], video_ids=[0])
    t0 = time.time()
    out = wrapper(batch)
    assert "pred_masks" in out, list(out)
    loss = sum(v.float().mean() for v in out.values() if torch.is_tensor(v) and v.requires_grad)
    loss.backward()
    got = {n for n, p in lora if p.grad is not None and torch.isfinite(p.grad).all()
           and p.grad.abs().sum() > 0}
    # lora_B is zero-initialised, so every lora_A has exactly zero gradient at
    # step 0 -- the informative count is the B matrices.
    b_dec = sum(1 for n in got if ".lora_B" in n and "sam_mask_decoder" in n)
    b_mem = sum(1 for n in got if ".lora_B" in n and ".transformer." in n
                and "sam_mask_decoder" not in n)
    b_vis = sum(1 for n in got if ".lora_B" in n and ".backbone." in n)
    assert (b_dec, b_mem, b_vis) == (28, 40, 8), (b_dec, b_mem, b_vis)
    print(f"PASS: real training forward+backward in {time.time() - t0:.0f}s; "
          f"gradient reached every lora_B (decoder {b_dec}, memory {b_mem}, visual {b_vis})")
    print("ALL REAL-CLASS CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
