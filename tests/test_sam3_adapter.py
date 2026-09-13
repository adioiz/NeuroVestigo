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
        return {"vision_features": x}


class Tracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = None                  # build_tracker leaves it unset
        self.sam_mask_decoder = nn.Linear(2, 2)
        self.hidden_dim = 256
        self.no_mem_embed = nn.Parameter(torch.zeros(1))
    def track_step(self):
        return "tracked"


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
