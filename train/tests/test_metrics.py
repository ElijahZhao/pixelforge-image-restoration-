"""Unit tests for train/metrics.py.

Run from repository root:
    python -m train.tests.run_tests
"""

from __future__ import annotations

import math
import sys
import traceback

import torch

sys.path.insert(0, "./train")
from metrics import psnr, ssim, evaluate_batch


def test_psnr_identical_is_inf():
    x = torch.rand(2, 3, 64, 64)
    val = psnr(x, x).item()
    assert math.isinf(val), f"PSNR of identical images should be inf, got {val}"


def test_psnr_finite_on_noise():
    x = torch.rand(2, 3, 64, 64)
    y = torch.clamp(x + 0.05 * torch.rand_like(x), 0, 1)
    val = psnr(x, y).item()
    assert not math.isinf(val) and val > 20, f"PSNR unexpectedly low: {val}"


def test_ssim_identical_is_one():
    x = torch.rand(2, 3, 64, 64)
    val = ssim(x, x).item()
    assert abs(val - 1.0) < 1e-5, f"SSIM of identical images should be 1, got {val}"


def test_ssim_decreases_with_noise():
    x = torch.rand(2, 3, 64, 64)
    y = torch.clamp(x + 0.2 * torch.rand_like(x), 0, 1)
    val = ssim(x, y).item()
    assert val < 0.98, f"SSIM should drop with noise, got {val}"
    assert val < ssim(x, x).item(), f"SSIM with noise should be lower than identical-image SSIM"


def test_evaluate_batch_keys():
    x = torch.rand(2, 3, 64, 64)
    y = torch.clamp(x + 0.02 * torch.rand_like(x), 0, 1)
    out = evaluate_batch(x, y)
    assert "psnr" in out and "ssim" in out
    assert isinstance(out["psnr"], float) and isinstance(out["ssim"], float)
    assert 0.0 <= out["ssim"] <= 1.0


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            traceback.print_exc()
            failed.append(t.__name__)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        sys.exit(1)
