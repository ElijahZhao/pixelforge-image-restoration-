"""Run all train/ unit tests without pytest.

Usage:
    python -m train.tests.run_tests        (from repository root)
    python train/tests/run_tests.py        (from repository root)
"""

from __future__ import annotations

import importlib
import sys
import traceback


MODULES = ["train.tests.test_models", "train.tests.test_metrics"]


def run() -> int:
    failures = []
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
            except Exception as e:
                print(f"  FAIL {t.__name__}: {e}")
                traceback.print_exc()
                failures.append(f"{mod_name}.{t.__name__}")
    print(f"\n{'='*50}")
    print(f"Result: {total - len(failures)}/{total} passed")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
