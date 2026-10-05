"""Run all train/ unit tests without pytest.

Usage:
    python -m train.tests.run_tests        (from repository root)
    python train/tests/run_tests.py        (from repository root)

Tests may raise ``unittest.SkipTest`` to be reported as SKIP rather than FAIL;
used for checks that need a network download (e.g. ImageNet VGG weights), so an
offline run does not look like a real regression.
"""

from __future__ import annotations

import importlib
import sys
import traceback
import unittest


MODULES = ["train.tests.test_models", "train.tests.test_metrics",
           "train.tests.test_correctness"]


def run() -> int:
    failures = []
    skipped = []
    total = 0
    for mod_name in MODULES:
        mod = importlib.import_module(mod_name)
        tests = [v for k, v in mod.__dict__.items() if k.startswith("test_")]
        print(f"\n[{mod_name}]")
        for t in tests:
            total += 1
            try:
                t()
                print(f"  PASS {t.__name__}")
            except unittest.SkipTest as e:
                # Dependency unavailable (typically no network), not a regression.
                print(f"  SKIP {t.__name__}: {e}")
                skipped.append(f"{mod_name}.{t.__name__}")
            except Exception as e:
                print(f"  FAIL {t.__name__}: {e}")
                traceback.print_exc()
                failures.append(f"{mod_name}.{t.__name__}")
    print(f"\n{'='*50}")
    print(f"Result: {total - len(failures) - len(skipped)}/{total} passed"
          + (f", {len(skipped)} skipped" if skipped else ""))
    if skipped:
        print("Skipped (dependency unavailable):")
        for s in skipped:
            print(f"  - {s}")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
