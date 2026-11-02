"""Reproducibility plumbing: the knobs that make a run re-runnable.

Everything asserted here failed silently before: the seed was a module constant
(un-overridable), the patch size never left the dataset class's default, and
results from different runs shared one set of filenames. Each check below pins
one of those, so a future edit that re-breaks them fails the suite instead of
quietly producing a table whose numbers cannot be rebuilt.

Run from repository root:
    python -m pytest train/tests/test_reproducibility.py
    python -m train.tests.run_tests
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "./train")

from datasets import DEFAULT_CROP, get_dataloader  # noqa: E402
from runenv import env_snapshot, git_state  # noqa: E402


def _load_train_script():
    """Import ``train/train.py`` as a module, whatever the entry point.

    ``import train`` resolves to the *package* when the suite is started with
    ``python -m train.tests.run_tests`` and to the *script* when pytest has
    ``./train`` on sys.path -- so the plain import passed one way and failed the
    other. Loading by file path removes the ambiguity.
    """
    path = pathlib.Path(__file__).resolve().parents[1] / "train.py"
    spec = importlib.util.spec_from_file_location("_pixelforge_train_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train_mod = _load_train_script()


def _write(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr.astype("uint8")).save(path)


def _sr_tree(n: int = 4, size: int = 48, seed: int = 0) -> str:
    """A tiny DIV2K-like tree: two splits, ``n`` images each."""
    base = tempfile.mkdtemp()
    rng = np.random.default_rng(seed)
    for split in ("train", "val"):
        for i in range(n):
            _write(f"{base}/div2k/{split}/{i:03d}.png",
                   rng.random((size, size, 3)) * 255)
    return base


# --------------------------------------------------------------------------- #
# 1. --crop_size reaches the dataset (it used to stop at get_dataloader).
# --------------------------------------------------------------------------- #
def test_dataloader_forwards_crop_size():
    base = _sr_tree()
    loader = get_dataloader("sr", base, "train", batch_size=2, scale=2,
                            num_workers=0, seed=1, crop_size=32)
    assert loader.dataset.crop_size == 32, \
        f"--crop_size=32 did not reach the dataset (got {loader.dataset.crop_size})"


def test_dataloader_crop_default_per_task():
    """No --crop_size means the task's own constraint-satisfying default."""
    base = _sr_tree()
    sr = get_dataloader("sr", base, "train", batch_size=1, scale=2,
                        num_workers=0, seed=1)
    assert sr.dataset.crop_size == DEFAULT_CROP["sr"], \
        "SR default crop must come from DEFAULT_CROP"

    lol = tempfile.mkdtemp()
    rng = np.random.default_rng(1)
    for split in ("train", "val"):
        for sub in ("low", "high"):
            for i in range(2):
                _write(f"{lol}/lol/{split}/{sub}/{i:03d}.png",
                       rng.random((64, 64, 3)) * (40 if sub == "low" else 220))
    ll = get_dataloader("lowlight", lol, "train", batch_size=1, num_workers=0,
                        seed=1)
    assert ll.dataset.crop_size == DEFAULT_CROP["lowlight"], \
        "low-light default crop must come from DEFAULT_CROP"


def test_crop_default_has_one_source():
    """train.py must not restate 96/128; it must read datasets.DEFAULT_CROP."""
    assert train_mod.TASK_CROP_DEFAULT is DEFAULT_CROP, \
        "TASK_CROP_DEFAULT diverged from datasets.DEFAULT_CROP"


# --------------------------------------------------------------------------- #
# 2. Same seed => same batches, including the augmentation RNG.
# --------------------------------------------------------------------------- #
def test_same_seed_gives_identical_batches():
    base = _sr_tree(n=4, size=48)
    a = get_dataloader("sr", base, "train", batch_size=2, scale=2,
                       num_workers=2, seed=123, crop_size=32)
    b = get_dataloader("sr", base, "train", batch_size=2, scale=2,
                       num_workers=2, seed=123, crop_size=32)
    for (la, ha), (lb, hb) in zip(a, b):
        assert torch.equal(la, lb) and torch.equal(ha, hb), \
            "two loaders with the same seed produced different batches"


def test_different_seed_changes_batches():
    """Guard against the trivial pass: seeding must actually matter.

    If this fails while the previous test passes, the augmentation is no longer
    random at all -- which would make the seed matrix meaningless in the other
    direction (five identical "seeds").
    """
    base = _sr_tree(n=4, size=48)
    a = get_dataloader("sr", base, "train", batch_size=2, scale=2,
                       num_workers=0, seed=1, crop_size=32)
    b = get_dataloader("sr", base, "train", batch_size=2, scale=2,
                       num_workers=0, seed=2, crop_size=32)
    same = all(torch.equal(x, y) for (xa, ha), (xb, hb) in zip(a, b)
               for x, y in ((xa, xb), (ha, hb)))
    assert not same, "seeds 1 and 2 produced identical batches"


# --------------------------------------------------------------------------- #
# 3. --tag isolates runs; the cross-seed guard reads the seed back.
# --------------------------------------------------------------------------- #
def test_run_dir_honours_tag():
    assert train_mod._run_dir("models", "") == "models"
    assert train_mod._run_dir("models", None) == "models"
    assert train_mod._run_dir("models", "seeds/s43") == os.path.join(
        "models", "seeds", "s43")
    # A stray leading/trailing slash must not create an absolute path.
    assert train_mod._run_dir("results", "/seeds/s43/") == os.path.join(
        "results", "seeds", "s43")


def test_seed_from_log_reads_header():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "log.csv")
    with open(p, "w", encoding="utf-8") as f:
        f.write("# objective=charbonnier\n# seed=43\n1,0.1,20.0,0.5,1.0\n")
    assert train_mod._seed_from_log(p) == 43

    with open(p, "w", encoding="utf-8") as f:  # pre---seed log
        f.write("# objective=charbonnier\n1,0.1,20.0,0.5,1.0\n")
    assert train_mod._seed_from_log(p) is None, \
        "a log without a seed header must read as unknown, not as a mismatch"

    assert train_mod._seed_from_log(os.path.join(d, "missing.csv")) is None


# --------------------------------------------------------------------------- #
# 4. CLI surface exists and stays backward compatible.
# --------------------------------------------------------------------------- #
def test_cli_accepts_seed_crop_tag():
    argv = sys.argv
    try:
        sys.argv = ["train.py", "--task", "sr", "--seed", "7",
                    "--crop_size", "32", "--tag", "seeds/s7"]
        args = train_mod.parse_args()
    finally:
        sys.argv = argv
    assert (args.seed, args.crop_size, args.tag) == (7, 32, "seeds/s7")


def test_cli_defaults_preserve_old_behaviour():
    """Omitting the new flags must reproduce the previous run exactly."""
    argv = sys.argv
    try:
        sys.argv = ["train.py", "--task", "sr"]
        args = train_mod.parse_args()
    finally:
        sys.argv = argv
    assert args.seed == train_mod.DEFAULT_SEED == 42
    assert args.crop_size is None, "crop default must resolve per task, not here"
    assert args.tag == "", "empty tag must mean 'no sub-directory'"


# --------------------------------------------------------------------------- #
# 5. Determinism switches and the environment snapshot.
# --------------------------------------------------------------------------- #
def test_seed_everything_pins_cudnn():
    before = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        train_mod.seed_everything(0)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = before


def test_env_snapshot_has_provenance():
    snap = env_snapshot()
    for key in ("timestamp_utc", "python", "torch", "git", "cudnn_deterministic"):
        assert key in snap, f"environment snapshot is missing {key}"
    git = git_state()
    assert set(git) == {"commit", "short", "branch", "dirty"}
    # Inside a repo checkout the commit must resolve; None here would mean the
    # snapshot silently records nothing.
    assert git["commit"] and len(git["commit"]) >= 7
