#!/usr/bin/env python3
"""Dump the environment a training/eval run would use, as JSON.

    python scripts/collect_env.py                 # to stdout
    python scripts/collect_env.py -o env.json     # to a file
    python scripts/collect_env.py --freeze        # include full `pip freeze`

This is the standalone companion to the per-run snapshot that
``train/train.py`` writes. Use it before training to record the box, and in the
paper's reproducibility section to show what "our environment" means.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "train"))

from runenv import dumps, env_snapshot  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Record the runtime environment")
    p.add_argument("-o", "--out", help="write JSON here instead of stdout")
    p.add_argument("--freeze", action="store_true",
                   help="include the full `pip freeze` list (large)")
    p.add_argument("--indent", type=int, default=2)
    args = p.parse_args()

    snap = env_snapshot(include_packages=args.freeze)
    text = dumps(snap, indent=args.indent)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
