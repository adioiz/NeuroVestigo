#!/usr/bin/env python3
"""
data.py -- manifest-driven access to the neuron sequences for the benchmark.

Ground-truth masks live in the neuron-labeling repo; the raw image stacks live
on an external SSD. Both are resolved through `data_manifest.json`, so this is
the single source of truth for "which sequences exist and where".

    from benchmark.data import Dataset
    ds = Dataset()
    for name in ds.split("val"):
        gt = ds.load_gt(name)          # (T, H, W) int32 label stack
        img = ds.load_image(name)      # (T, H, W) or None if the SSD isn't mounted
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "data_manifest.json")
# The split is read from the SHIPPED model's own config. This was
# gpu-checkpoints/ until that directory was removed, at which point
# val_videos silently became empty -- TEST_VIDEOS below is hardcoded, so
# test scoring kept working and nothing announced the loss.
CONFIG = os.path.join(REPO, "checkpoints", "training_config.json")

# True held-out TEST set: videos whose masks exist in neurons-labeling but were
# NEVER prepared into training_data, so the shipped (25-video) model never saw them.
# A71_C1 / A81_C1 are entirely new series (cleanest); A12_C3_100k / A12_C1_100k_2 are
# new fields from the same 100k batch (mild same-sample caveat).
# NOTE: these must be `prepare`d into training_data/{images,masks}/<name>/ before use.
# (The earlier `_100k` test set was WRONG: the shipped model DID train on those.)
TEST_VIDEOS = ["A71_C1", "A81_C1", "A12_C3_100k", "A12_C1_100k_2"]


def config_hint() -> str:
    """Why the val split might be empty, said once, in the place it is raised."""
    return (f"no data.val_videos in {os.path.relpath(CONFIG, REPO)} "
            f"(it is written by training and may be null when the run used "
            f"val_fraction instead of an explicit list)")


class Dataset:
    def __init__(self, manifest: str = MANIFEST, config: str = CONFIG):
        with open(manifest) as f:
            entries = json.load(f)
        self.entries = {e["name"]: e for e in entries}
        self.val_videos: list[str] = []
        if os.path.exists(config):
            with open(config) as f:
                # `or []`, not a .get default: the key is PRESENT-but-null in a
                # config written by a run that used val_fraction instead of an
                # explicit list, so the default never fires and val_videos would
                # become None -- split("val") then died with
                # "TypeError: 'NoneType' object is not iterable".
                self.val_videos = json.load(f)["data"].get("val_videos") or []

    # -- listing -----------------------------------------------------------
    def all(self) -> list[str]:
        return sorted(self.entries)

    def split(self, which: str = "val") -> list[str]:
        """'test' -> true held-out set (post-training-manifest _100k batch);
        'val' -> model-selection videos; 'train' -> the 15 actually trained on;
        'all' -> everything."""
        if which == "all":
            return self.all()
        test = [n for n in TEST_VIDEOS if n in self.entries]
        val = [n for n in self.val_videos if n in self.entries]
        if which == "test":
            return test
        if which == "val":
            if not val:
                raise RuntimeError(
                    f"the 'val' split is empty: {config_hint()}\n"
                    "Pass --videos explicitly, or point CONFIG at a "
                    "training_config.json whose data.val_videos is set."
                )
            return val
        if which == "train":
            return [n for n in self.all() if n not in set(val) | set(test)]
        raise ValueError(f"unknown split '{which}'")

    # -- loading -----------------------------------------------------------
    def gt_path(self, name: str) -> str:
        """Absolute path to the GT mask TIF from the manifest.

        May not exist off this Mac -- the manifest stores absolute paths into
        an external SSD and a sibling checkout. Use load_gt(), which falls back
        to the prepared PNGs. Kept for callers that genuinely want the TIF.
        """
        return self.entries[name]["mask_tif"]

    def prepared_mask_dir(self, name: str) -> str:
        return os.path.join(REPO, "training_data", "masks", name)

    def image_path(self, name: str) -> str:
        return self.entries[name]["image_tif"]

    def load_gt(self, name: str) -> np.ndarray:
        """GT stack [T,H,W].

        Prefers the manifest TIF, but falls back to the prepared per-frame PNGs
        in training_data/masks/<name>/. The manifest holds ABSOLUTE paths into
        /Volumes/Adi-SSD and a sibling neurons-labeling checkout, so on the
        cluster the TIF is simply absent -- while training_data/ is rsynced
        there and is what run_sam3.py itself reads. Without this, scoring only
        ever worked on the laptop.
        """
        import tifffile
        tif = self.gt_path(name)
        if os.path.exists(tif):
            m = tifffile.imread(tif)
            if m.ndim == 2:
                m = m[None]
            return m.astype(np.int32)

        d = self.prepared_mask_dir(name)
        files = sorted(glob.glob(os.path.join(d, "*.png")))
        if not files:
            raise FileNotFoundError(
                f"no ground truth for {name!r}: manifest TIF {tif!r} is absent "
                f"and no prepared PNGs in {os.path.relpath(d, REPO)}/. "
                f"Run `python cli.py prepare` for this recording, or score on a "
                f"machine where the manifest paths resolve."
            )
        from PIL import Image
        return np.stack([np.array(Image.open(f)) for f in files]).astype(np.int32)

    def load_image(self, name: str) -> np.ndarray | None:
        """Raw image stack, or None if the SSD path is not currently mounted."""
        import tifffile
        p = self.image_path(name)
        if not os.path.exists(p):
            return None
        img = tifffile.imread(p)
        return img if img.ndim == 3 else img[None]

    def available(self, name: str) -> dict[str, bool]:
        return {"gt": os.path.exists(self.gt_path(name)),
                "image": os.path.exists(self.image_path(name))}


if __name__ == "__main__":
    ds = Dataset()
    print(f"manifest: {len(ds.all())} sequences")
    print(f"val split ({len(ds.split('val'))}): {ds.split('val')}")
    print(f"train split ({len(ds.split('train'))} sequences)")
    print("\nper-sequence availability (val):")
    for n in ds.split("val"):
        a = ds.available(n)
        gt = ds.load_gt(n) if a["gt"] else None
        shape = gt.shape if gt is not None else "-"
        labels = np.unique(gt).tolist() if gt is not None else "-"
        print(f"  {n:<12} gt={a['gt']!s:<5} img={a['image']!s:<5} "
              f"shape={shape} labels={labels}")
