"""Unit tests for train/models.py.

Run from repository root:
    python -m train.tests.run_tests

Or directly:
    python train/tests/test_models.py
"""

from __future__ import annotations

import sys
import traceback

import torch

sys.path.insert(0, "./train")
from models import SRCNN, SRGenerator, LowLightUNet, build_model


def _assert_shape(model, x, expected, label):
    y = model(x)
    assert list(y.shape) == expected, f"{label}: expected {expected}, got {list(y.shape)}"


def _assert_range(y, label):
    assert y.min() >= -1e-5 and y.max() <= 1.0 + 1e-5, f"{label}: output not in [0, 1]"


def test_srcnn_scale2():
    x = torch.rand(1, 3, 32, 32)
    model = SRCNN(scale=2)
    y = model(x)
    assert list(y.shape) == [1, 3, 64, 64], f"SRCNN scale2 shape wrong: {list(y.shape)}"
    _assert_range(y, "SRCNN")


def test_srcnn_scale4():
    x = torch.rand(1, 3, 32, 32)
    model = SRCNN(scale=4)
    y = model(x)
    assert list(y.shape) == [1, 3, 128, 128], f"SRCNN scale4 shape wrong: {list(y.shape)}"
    _assert_range(y, "SRCNN scale4")


def test_sr_generator_scale2():
    x = torch.rand(1, 3, 32, 32)
    model = SRGenerator(scale=2)
    y = model(x)
    assert list(y.shape) == [1, 3, 64, 64], f"SRGenerator scale2 shape wrong: {list(y.shape)}"
    _assert_range(y, "SRGenerator scale2")


def test_sr_generator_scale4():
    x = torch.rand(1, 3, 32, 32)
    model = SRGenerator(scale=4)
    y = model(x)
    assert list(y.shape) == [1, 3, 128, 128], f"SRGenerator scale4 shape wrong: {list(y.shape)}"
    _assert_range(y, "SRGenerator scale4")


def test_lowlight_unet():
    x = torch.rand(2, 3, 64, 64)
    model = LowLightUNet()
    y = model(x)
    assert list(y.shape) == [2, 3, 64, 64], f"LowLightUNet shape wrong: {list(y.shape)}"
    _assert_range(y, "LowLightUNet")


def test_build_model_factory():
    srcnn = build_model("sr", scale=2, advanced=False)
    srgen = build_model("sr", scale=4, advanced=True)
    ll = build_model("lowlight", scale=1, advanced=False)
    assert isinstance(srcnn, SRCNN)
    assert isinstance(srgen, SRGenerator)
    assert isinstance(ll, LowLightUNet)
    # Advanced SR should be the generator, not SRCNN.
    assert srgen.scale == 4


def test_batch_independence_and_determinism():
    """Real, falsifiable properties (the old version used torch.zeros and only
    asserted y[0] == y[1], which is trivially true for ANY deterministic module
    (even a single Conv2d stub), so it could not catch a real defect).

    We now assert two things that CAN fail:
      1. In eval() the model is deterministic for the same input.
      2. Different inputs produce different outputs (i.e. the input is actually
         used; a model that ignored its input or collapsed to a constant, e.g.
         a dead-ReLU/zeroed-weight bug, would fail this).
    """
    model = SRGenerator(scale=2, num_blocks=4).eval()
    x = torch.rand(2, 3, 16, 16)
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x.clone())
        y_other = model(torch.rand(2, 3, 16, 16))

    # 1. Deterministic in eval mode.
    assert torch.allclose(y1, y2, atol=1e-6), "eval-mode output is not deterministic"

    # 2. The model actually responds to its input: a different input must give a
    #    different output. Guards against constant-output / dead-network bugs.
    assert not torch.allclose(y1, y_other, atol=1e-4), (
        "model produced (nearly) identical outputs for different inputs; "
        "it is not using its input"
    )

    # 3. Batch independence: running the two samples together must equal running
    #    them separately. This is the property the old test *claimed* to check.
    with torch.no_grad():
        separately = torch.cat([model(x[i:i + 1]) for i in range(2)], dim=0)
    assert torch.allclose(y1, separately, atol=1e-5), (
        "batched output differs from per-sample output"
    )


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
