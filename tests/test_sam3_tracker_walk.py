"""Does the fixed walk reach a tracker nested inside nn.Modules?

Stubs torch/PIL/sam3 so this runs with no GPU and no 3.5GB checkpoint. The
point is the traversal predicate, which is pure Python.
"""
import sys, types

# --- stub torch: nn.Module must be CALLABLE, which is the whole bug ---------
torch = types.ModuleType("torch")
class _Module:
    def __init__(self): self._children = {}
    def __call__(self, *a, **k): raise AssertionError("never called")
    def named_children(self): return list(self._children.items())
    def add_module(self, n, m): self._children[n] = m; setattr(self, n, m)
torch.nn = types.SimpleNamespace(Module=_Module)
torch.Tensor = type("Tensor", (), {})
torch.autocast = lambda *a, **k: None
torch.cuda = types.SimpleNamespace(is_available=lambda: False)
torch.backends = types.SimpleNamespace(cuda=types.SimpleNamespace())
sys.modules["torch"] = torch
sys.modules["PIL"] = types.ModuleType("PIL")
sys.modules["PIL"].Image = types.SimpleNamespace(open=None)

# --- the shape the real log showed: predictor -> model -> ... -> tracker ----
class Tracker(_Module):
    def add_new_masks(self, **k): return "seeded"
class Middle(_Module): pass
class Predictor(_Module): pass

def build_predictor_tree():
    vp, mid, trk = Predictor(), Middle(), Tracker()
    vp.add_module("model", mid)
    mid.add_module("tracker", trk)
    vp.junk = "a string"          # leaves the walk must skip, not crash on
    vp.count = 7
    return vp, trk

sam3 = types.ModuleType("sam3"); mb = types.ModuleType("sam3.model_builder")
_TREE = {}
mb.build_sam3_predictor = lambda **k: _TREE["vp"]
mb.build_tracker = lambda atd, with_backbone=False: (_ for _ in ()).throw(
    RuntimeError("fresh-init path must NOT be reached first"))
sam3.model_builder = mb
sys.modules["sam3"] = sam3; sys.modules["sam3.model_builder"] = mb

sys.path.insert(0, "benchmark")
import probe_sam3_mask as P

vp, trk = build_predictor_tree(); _TREE["vp"] = vp
got = P.build({"sam3.model_builder.build_tracker": mb.build_tracker}, {}, None)

assert got is trk, f"expected the nested tracker, got {got!r}"
print("PASS: nested tracker found through two nn.Module hops")

# --- the old predicate, to show it could not have worked -------------------
def old_walk(o, depth=0, seen=None):
    seen = seen if seen is not None else set()
    if depth > 3 or id(o) in seen: return None
    seen.add(id(o))
    for a in dir(o):
        if a.startswith("__"): continue
        try: v = getattr(o, a)
        except BaseException: continue
        if hasattr(v, "add_new_mask") or hasattr(v, "add_new_masks"): return v
        if hasattr(v, "__dict__") and not callable(v):
            r = old_walk(v, depth + 1, seen)
            if r is not None: return r
    return None

vp2, trk2 = build_predictor_tree()
assert old_walk(vp2) is None, "old walk should have failed here"
print("PASS: old `not callable(v)` guard finds nothing on the same tree")


# ---------------------------------------------------------------------------
# The three Stage A classes disagree about init_state. Picking the first match
# would often hand stage C an object that cannot be given frames at all.
# ---------------------------------------------------------------------------
class MultiplexDemo(_Module):                      # no video_path
    def add_new_masks(self, **k): return "seeded"
    def init_state(self, video_height, video_width, num_frames,
                   cached_features=None): pass
class RealTracker(_Module):                        # has video_path
    def add_new_mask(self, **k): return "seeded"
    def init_state(self, video_height=None, video_width=None, num_frames=None,
                   video_path=None): pass

vp3 = Predictor(); mid3 = Middle()
vp3.add_module("model", mid3)
mid3.add_module("a_multiplex", MultiplexDemo())    # found FIRST alphabetically
mid3.add_module("z_tracker", RealTracker())
_TREE["vp"] = vp3
got3 = P.build({"sam3.model_builder.build_tracker": mb.build_tracker}, {}, None)
assert isinstance(got3, RealTracker), (
    f"picked {type(got3).__name__}; must prefer the one taking video_path")
print("PASS: prefers the feedable tracker over an earlier unfeedable match")

# Only an unfeedable candidate exists -> still returned, with a warning, so the
# run is not aborted on a guess about what stage C can do.
vp4 = Predictor(); mid4 = Middle()
vp4.add_module("model", mid4); mid4.add_module("only", MultiplexDemo())
_TREE["vp"] = vp4
got4 = P.build({"sam3.model_builder.build_tracker": mb.build_tracker}, {}, None)
assert isinstance(got4, MultiplexDemo), "should fall back, not return None"
print("PASS: falls back to an unfeedable candidate rather than returning None")


# ---------------------------------------------------------------------------
# session_state: the tracker's own init_state cannot load video, so the state
# must come from the predictor's session interface (run_sam3.py:283).
# ---------------------------------------------------------------------------
class SessionPredictor(_Module):
    def __init__(self):
        super().__init__()
        self.model = _Module()
        self.model.init_state = lambda **k: None
        self._sessions = {}
    def handle_request(self, req):
        assert req["type"] == "start_session"
        assert req["resource_path"] == "/frames", req
        self._sessions["s1"] = {"images": ["frame"] * 12, "num_frames": 12}
        return {"session_id": "s1"}

st, sid = P.session_state(SessionPredictor(), "/frames")
assert sid == "s1", sid
assert isinstance(st, dict) and "images" in st, st
print("PASS: session_state finds a state that holds images")

# A predictor that hands back a session id but hides the state must not be
# mistaken for success -- returning None keeps stage D marked INCONCLUSIVE.
class Hidden(SessionPredictor):
    def handle_request(self, req):
        return {"session_id": "nowhere"}
st2, _ = P.session_state(Hidden(), "/frames")
assert st2 is None, st2
print("PASS: unlocatable state returns None rather than a false positive")
