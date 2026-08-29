"""Exercise the probe's own code against a stub with SAM 3's EXACT signatures
(as printed by stage A on the cluster). This cannot test SAM 3 -- it tests MY
code for the bug classes that have cost four cluster runs."""
import importlib.util, os, sys, numpy as np, torch
R="/Users/adioiz/Documents/Learnings/Thesis/scripts/Fine-tune-SAM2-model-for-Neurons-segmentation"
os.chdir(R); sys.path.insert(0, R)
spec=importlib.util.spec_from_file_location("probe", "benchmark/probe_sam3_mask.py")
probe=importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)

class StubTracker(torch.nn.Module):
    """Signatures copied verbatim from stage A output."""
    def __init__(self, refuse_mask=False, need_preflight=False):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(4,4))
        self.refuse_mask, self.need_preflight = refuse_mask, need_preflight
        self._pre = False
    def init_state(self, video_height=None, video_width=None, num_frames=None,
                   video_path=None, cached_features=None, offload_video_to_cpu=False,
                   offload_state_to_cpu=False, async_loading_frames=False):
        if video_path is None:
            return {"images": None, "n": num_frames, "H": video_height, "W": video_width}
        n = len([f for f in os.listdir(video_path) if f.endswith(".jpg")])
        if n == 0:
            raise RuntimeError(f"no images found in {video_path}")
        return {"images": list(range(n)), "n": n}
    def add_new_mask(self, inference_state, frame_idx, obj_id, mask,
                     add_mask_to_memory=False):
        if inference_state.get("images") is None:
            raise KeyError("images")
        assert mask.dim() == 2, "mask must be 2-D"        # SAM 3 does this
        if self.refuse_mask:
            raise RuntimeError("mask prompts are not supported on this path")
        self._seeded = (frame_idx, obj_id, bool(add_mask_to_memory))
    def propagate_in_video_preflight(self, inference_state, run_mem_encoder=True):
        self._pre = True
    def propagate_in_video(self, inference_state, start_frame_idx,
                           max_frame_num_to_track, reverse, tqdm_disable=False,
                           obj_ids=None, run_mem_encoder=True, propagate_preflight=False):
        if self.need_preflight and not self._pre:
            raise RuntimeError("preflight not run")
        H, W = 485, 307
        for t in range(inference_state["n"]):
            # bfloat16 on purpose: this is what killed stage E before
            yield t, [1], torch.zeros(1, 1, H, W, dtype=torch.bfloat16)

gt = np.zeros((12, 485, 307), bool); gt[:, 100:140, 100:160] = True
frames = os.path.join(R, "training_data", "images", "A81_C1")

for name, kw in (("happy path", {}),
                 ("needs explicit preflight", {"need_preflight": True}),
                 ("genuinely refuses masks", {"refuse_mask": True})):
    print("\n" + "#"*66); print("# SCENARIO:", name); print("#"*66)
    t = StubTracker(**kw)
    t, _ = probe.prepare(t)
    ok = probe.drive(t, frames, gt)
    print(f">>> drive() returned {ok}")
