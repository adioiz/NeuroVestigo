"""repair_backbone: does it read the right attribute name and attach a real one?"""
import sys, types, inspect, re

torch = types.ModuleType("torch")
class _M:
    def __init__(self): self._modules = {}
    def named_children(self): return list(self._modules.items())
    def add(self, n, m): self._modules[n] = m; setattr(self, n, m)
torch.nn = types.SimpleNamespace(Module=_M)
torch.Tensor = type("T", (), {}); torch.cuda = types.SimpleNamespace(is_available=lambda: False)
torch.backends = types.SimpleNamespace(cuda=types.SimpleNamespace())
torch.autocast = lambda *a, **k: None
sys.modules["torch"] = torch
sys.modules["PIL"] = types.ModuleType("PIL"); sys.modules["PIL"].Image = types.SimpleNamespace()

src = open("benchmark/run_sam3_mask.py").read()
body = "def repair_backbone" + src.split("def repair_backbone")[1].split("\n# ---")[0]
ns = {"inspect": inspect, "print": print}
exec(compile(body, "s", "exec"), ns)

class Tracker(_M):
    """Mimics the real class: calls self.image_encoder.forward_image, and the
    multiplex build leaves that slot None."""
    def __init__(self):
        super().__init__()
        self.image_encoder = None
    def _get_image_feature(self):
        return self.image_encoder.forward_image(1)

class Encoder(_M):
    def forward_image(self, x): return "feats"

vp = _M(); det = _M(); enc = Encoder()
vp.add("detector", det); det.add("backbone", enc)
tk = Tracker()
assert ns["repair_backbone"](vp, tk) is True
assert tk.image_encoder is enc, tk.image_encoder
assert tk._get_image_feature() == "feats"
print("PASS: found 'image_encoder' from source, attached the real encoder")

# Nothing None -> must not touch anything
tk2 = Tracker(); tk2.image_encoder = enc
assert ns["repair_backbone"](vp, tk2) is False
print("PASS: no-op when the slot is already populated")

# No provider anywhere -> report, do not attach junk
tk3 = Tracker()
assert ns["repair_backbone"](_M(), tk3) is False
assert tk3.image_encoder is None
print("PASS: refuses when the predictor has no forward_image provider")
