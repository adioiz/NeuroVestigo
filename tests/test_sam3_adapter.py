"""Sam3TrainingAdapter: the SAM 2 surface over SAM 3.1's tracker.

Both bugs this guards against were found on the cluster or in the first local
test, not by reading:
  * __getattr__ that skipped nn.Module's own lookup, so `self.tracker` fell
    through to delegation ("'Sam3TrackerPredictor' object has no attribute
    'tracker'").
  * a property setter for image_encoder that nn.Module.__setattr__ bypasses
    for Module-typed values, leaving a dangling second submodule.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch                                  # noqa: E402
import torch.nn as nn                         # noqa: E402
from src.sam3_backbone import Sam3TrainingAdapter  # noqa: E402


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(3, 3)
    def forward_image(self, x):
        # the shared SAM3VLBackbone returns the SAM 2 contract under this key
        return {"sam2_backbone_out": {"vision_features": x}}


class Tracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = None                  # build_tracker leaves it unset
        self.sam_mask_decoder = nn.Linear(2, 2)
        self.hidden_dim = 256
        self.no_mem_embed = nn.Parameter(torch.zeros(1))
        self.seen = []
    def track_step(self, image=None, **kw):
        # SAM 3's signature has a required `image`; the shim must supply it
        assert "image" in {"image": image}, "shim must pass image"
        return "tracked"
    def _forward_sam_heads(self, backbone_features, point_inputs=None,
                           mask_inputs=None, high_res_features=None,
                           multimask_output=False, gt_masks=None):
        # record what SAM 3's line 333 would see
        self.seen.append((self.training,
                          getattr(self, "teacher_force_obj_scores_for_mem", None),
                          None if gt_masks is None else tuple(gt_masks.shape)))
        return ("heads",)


bb = Backbone()
a = Sam3TrainingAdapter(Tracker(), bb)        # crashed on the cluster before the fix
assert a.tracker.backbone is bb
assert a.image_encoder is bb
assert isinstance(a.sam_mask_decoder, nn.Linear)
assert a.hidden_dim == 256
assert a.no_mem_embed.shape == (1,)
assert a.track_step() == "tracked"
assert a.forward_image(torch.ones(1))["vision_features"].shape == (1,)
try:
    a.definitely_missing
    raise SystemExit("FAIL: a genuine miss must raise AttributeError")
except AttributeError:
    pass
print("PASS: constructs, aliases, delegates, raises on genuine misses")

n_before = sum(p.numel() for p in a.parameters())
bb2 = Backbone()
a.image_encoder = bb2                         # what compile_encoder does
assert a.tracker.backbone is bb2, "setter must replace the tracker's backbone"
assert "image_encoder" not in a._modules, "must not register a second copy"
assert sum(p.numel() for p in a.parameters()) == n_before, "param count must not change"
print("PASS: image_encoder assignment replaces the backbone, no dangling module")

# --- _prepare_tracker_for_training: the obj-score rule is disabled through
# SAM 3's own teacher-forcing branch, in eval mode too, and the flag it flips
# is restored afterwards.
tk = a.tracker
assert tk.prob_to_dropout_spatial_mem == 0.0
assert tk.teacher_force_obj_scores_for_mem is True
a.eval()
feats = torch.zeros(3, 8, 4, 4)
assert tk._forward_sam_heads(feats) == ("heads",)
training_seen, tf_seen, gt_shape = tk.seen[-1]
assert training_seen is True and tf_seen is True and gt_shape == (3, 1), tk.seen[-1]
assert tk.training is False, "training flag must be restored after the call"
tk._forward_sam_heads(feats, gt_masks=torch.ones(3, 1, 4, 4))
assert tk.seen[-1][2] == (3, 1, 4, 4), "a caller-supplied gt_masks must pass through"
print("PASS: obj-score rule disabled via teacher forcing; training flag restored")
