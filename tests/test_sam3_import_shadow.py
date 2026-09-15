"""import_sam3 must return the REAL package when the repo root shadows it.

Reproduces the cluster failure: <repo>/sam3/ is the clone (no __init__.py),
<repo>/sam3/sam3/ is the package, and the package is reachable only through a
meta-path finder (how `pip install -e` exposes it). With the repo root on
sys.path, PathFinder builds a namespace `sam3` with __file__ = None and
pkg_resources.resource_filename("sam3", ...) dies.
"""
import importlib.abc
import importlib.util
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sam3_backbone import _shadowing_entries, import_sam3  # noqa: E402

root = tempfile.mkdtemp(prefix="fakerepo_")
clone = os.path.join(root, "sam3")               # the clone: NO __init__.py
pkg = os.path.join(clone, "sam3")                # the real package
os.makedirs(pkg)
open(os.path.join(pkg, "__init__.py"), "w").write("MARK = 'real'\n")
open(os.path.join(pkg, "model_builder.py"), "w").write("def build(): return 'ok'\n")


class EditableFinder(importlib.abc.MetaPathFinder):
    """What setuptools' editable install registers: resolves sam3 and its
    submodules from the package dir without putting it on sys.path."""
    def find_spec(self, name, path=None, target=None):
        if name == "sam3":
            return importlib.util.spec_from_file_location(
                name, os.path.join(pkg, "__init__.py"),
                submodule_search_locations=[pkg])
        if name.startswith("sam3."):
            f = os.path.join(pkg, name.split(".", 1)[1].replace(".", "/") + ".py")
            if os.path.exists(f):
                return importlib.util.spec_from_file_location(name, f)
        return None


sys.meta_path.append(EditableFinder())           # after PathFinder, as in real life
for k in [k for k in sys.modules if k == "sam3" or k.startswith("sam3.")]:
    del sys.modules[k]

# --- reproduce the bug ------------------------------------------------------
sys.path.insert(0, root)                          # what the preflight / cli.py do
import sam3                                       # noqa: E402
assert getattr(sam3, "__file__", None) is None, "expected the namespace shadow"
from sam3.model_builder import build              # noqa: E402  (submodule still works!)
assert build() == "ok"
print("REPRODUCED: sam3.__file__ is None while sam3.model_builder imports fine")
assert _shadowing_entries(sys.path) == [root], _shadowing_entries(sys.path)
print("PASS: the shadowing entry is identified")

# --- the fix ----------------------------------------------------------------
real = import_sam3()
assert real.__file__ and real.__file__.endswith(os.path.join("sam3", "sam3", "__init__.py")), real.__file__
assert real.MARK == "real"
assert sys.modules["sam3"] is real
assert root in sys.path, "sys.path must be restored afterwards"
from sam3.model_builder import build as build2    # noqa: E402
assert build2() == "ok"
print("PASS: import_sam3 returns the real package and restores sys.path")

# --- idempotent when already correct ----------------------------------------
assert import_sam3() is real
print("PASS: no-op when the real package is already loaded")
