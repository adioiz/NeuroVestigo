"""
The cross-validation folds (src/folds.py): the rules the test sets were chosen
under, the agreement of every place that names the paper's split, and the
--fold path through cli.py into the config the trainer receives.
"""
import dataclasses
import io
import os
import sys
from contextlib import redirect_stdout

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src import folds  # noqa: E402


def test_rules_hold():
    folds.check_folds()


def test_every_fold_is_23_5_4_and_tests_are_disjoint():
    names = folds.manifest_names()
    assert len(names) == 32
    seen = set()
    for f, test in folds.FOLDS.items():
        assert len(folds.train_videos(f, names)) == 23
        assert len(folds.VAL) == 5 and not set(test) & set(folds.VAL)
        assert not set(test) & seen
        seen |= set(test)
    assert len(seen) == 12


def test_fold_1_is_the_split_every_other_module_names():
    from src.config import TrainingConfig
    from benchmark.data import TEST_VIDEOS, Dataset
    assert TrainingConfig().data.test_videos == folds.FOLDS[1]
    assert TEST_VIDEOS == folds.FOLDS[1]
    assert Dataset().split("test") == Dataset(fold=1).split("test")
    assert Dataset().split("val") == Dataset(fold=1).split("val") == folds.VAL


@pytest.mark.parametrize("bad, message", [
    ({2: ["A51_C1", "A61_C1", "B11_C1_100k", "A31_C1"]}, "validation"),
    ({2: ["A71_C1", "A61_C1", "B11_C1_100k", "B12_C1_100k"]}, "share"),
    ({2: ["A51_C1", "A61_C1", "B11_C1_100k", "A32_C1_1"]}, "frames"),
    ({2: ["A51_C1", "A61_C1", "B11_C1_100k"]}, "4 distinct"),
])
def test_rules_reject(monkeypatch, bad, message):
    monkeypatch.setattr(folds, "FOLDS", {**folds.FOLDS, **bad})
    with pytest.raises(ValueError, match=message):
        folds.check_folds()


def test_cli_fold_sets_test_and_val_and_nothing_else(monkeypatch):
    import src.trainer
    import cli
    seen = {}
    monkeypatch.setattr(src.trainer, "train", lambda c: seen.setdefault("c", c))

    def resolve(*argv):
        seen.clear()
        monkeypatch.setattr(sys, "argv", ["cli.py", "train", *argv])
        with redirect_stdout(io.StringIO()):
            cli.main()
        return dataclasses.asdict(seen["c"])

    base = resolve("--val-videos", ",".join(folds.VAL))
    for f in (1, 2, 3):
        got = resolve("--fold", str(f))
        assert got["data"]["test_videos"] == folds.FOLDS[f]
        assert got["data"]["val_videos"] == folds.VAL
        diff = {k for k in got if got[k] != base[k]}
        diff |= {f"data.{k}" for k in got["data"] if got["data"][k] != base["data"][k]}
        assert diff <= {"data", "data.test_videos"}, diff


def test_cli_refuses_fold_with_val_videos(monkeypatch):
    import cli
    monkeypatch.setattr(sys, "argv", ["cli.py", "train", "--fold", "2",
                                      "--val-videos", "A31_C1"])
    with pytest.raises(SystemExit, match="drop --val-videos"):
        cli.main()
