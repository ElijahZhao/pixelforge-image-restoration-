"""Environment snapshot: the software stack behind a reported number.

Why this exists
---------------
A PSNR printed in a paper is only reproducible if the reader can rebuild the
stack that produced it. This project has already been bitten by the absence of
that record: three mutually inconsistent torch versions were declared in three
different files (root ``requirements.txt``, ``deploy/streamlit/requirements.txt``,
``requirements.lock.txt``), and nothing in the training log said which one the
numbers came from.

So every run writes ``config_<task>_<model>_scale<n>.json`` next to its CSV,
containing the full argument set plus this snapshot. The JSON is the artifact a
reviewer can diff against their own environment; the CSV stays a plain table.

``scripts/collect_env.py`` reuses this module for a standalone dump (it is the
one that also embeds the full ``pip freeze``).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone


def _repo_root() -> str:
    """Walk up from this file to the directory holding ``.git``.

    Falling back to the CWD would make the git snapshot silently depend on
    where the user happened to run from -- a config file that says
    ``git_commit: null`` because of a path accident is worse than no field.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (here, os.path.dirname(here), os.getcwd()):
        probe = cand
        for _ in range(6):
            if os.path.isdir(os.path.join(probe, ".git")):
                return probe
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
    return here


def _git(*args: str, cwd: str | None = None) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd or _repo_root(),
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_state() -> dict:
    """Commit SHA, branch, and whether the tree was dirty at run time.

    ``dirty`` matters: a result produced from uncommitted code cannot be
    rebuilt from the SHA alone, and a reader who is not told will assume it can.
    """
    root = _repo_root()
    commit = _git("rev-parse", "HEAD", cwd=root)
    porcelain = _git("status", "--porcelain", cwd=root)
    return {
        "commit": commit,
        "short": commit[:7] if commit else None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=root),
        "dirty": bool(porcelain) if porcelain is not None else None,
    }


def pip_freeze() -> list[str]:
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                             capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return []
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def _pkg_version(name: str) -> str | None:
    try:
        mod = __import__(name)
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        return None
    return getattr(mod, "__version__", None)


def env_snapshot(include_packages: bool = False) -> dict:
    """Return a JSON-serialisable description of the current environment.

    Import-light on purpose: ``torch`` is imported lazily so the module stays
    usable from a CPU-only box or from a test that has no torch installed.
    """
    snap: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "git": git_state(),
    }

    try:
        import torch

        snap["torch"] = torch.__version__
        snap["torchvision"] = _pkg_version("torchvision")
        snap["numpy"] = _pkg_version("numpy")
        snap["pillow"] = _pkg_version("PIL")
        snap["cuda_available"] = bool(torch.cuda.is_available())
        snap["cuda_version"] = getattr(torch.version, "cuda", None)
        snap["cudnn_version"] = (torch.backends.cudnn.version()
                                 if torch.backends.cudnn.is_available() else None)
        snap["device"] = (torch.cuda.get_device_name(0)
                          if torch.cuda.is_available() else "cpu")
        snap["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        snap["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    except Exception:  # noqa: BLE001 - torch absent is a legitimate state
        snap["torch"] = None

    if include_packages:
        pkgs = pip_freeze()
        snap["pip_freeze"] = pkgs
        snap["pip_freeze_sha256"] = hashlib.sha256(
            "\n".join(pkgs).encode("utf-8")).hexdigest() if pkgs else None
    else:
        # The per-run config keeps a fingerprint, not the whole list: 15 seed
        # runs x 200 lines of package noise is a lot of file for one digest,
        # and `scripts/collect_env.py` can always expand it.
        pkgs = pip_freeze()
        snap["pip_freeze_sha256"] = hashlib.sha256(
            "\n".join(pkgs).encode("utf-8")).hexdigest() if pkgs else None
        snap["pip_freeze_count"] = len(pkgs)

    return snap


def dumps(obj: dict, indent: int = 2) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=indent, default=str)
