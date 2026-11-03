"""Tests that lock the LR degradation operator.

Three clauses, each of which was previously implicit or duplicated:

1. **training's operator did not change.** `datasets.py` used to inline
   `TF.resize(..., BICUBIC, antialias=True)`; it now calls
   `degradation.build_lr`. If those two ever disagree, every checkpoint in the
   repo was fitted on a distribution the code no longer reproduces -- and the
   symptom would be a quiet 0.1-0.3 dB drift nobody would trace back to here.
2. **the modes are actually different.** A "protocol dimension" whose levels are
   identical is not a dimension. antialias on/off and PIL-vs-float must move the
   pixels by more than float noise, or the P1..P6 matrix would be measuring
   nothing.
3. **batching does not change a pixel.** PIL has no batch API, so the PIL mode
   loops; a bug there (wrong size order, per-batch normalisation) would make the
   number depend on how images were grouped -- the exact failure `metrics.py`
   was fixed for.

Run from repository root:
    python -m train.tests.run_tests
    python -m pytest train/tests/test_degradation.py -q
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest
import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, "./train")
from degradation import (DEFAULT_MODE, LEGACY_EVAL_MODE, MODES,  # noqa: E402
                         PIL_BICUBIC, TV_BICUBIC_AA, TV_BICUBIC_NOAA,
                         build_lr, build_lr_pil, describe, lr_size)
from datasets import SuperResolutionDataset  # noqa: E402


def _img(seed: int = 0, h: int = 64, w: int = 64, batch: int = 1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(batch, 3, h, w, generator=g)


# ------------------------------------------------------- 1. training unchanged


def test_default_mode_is_the_training_operator():
    """DEFAULT_MODE must be the antialiased float bicubic, not a library default."""
    assert DEFAULT_MODE == TV_BICUBIC_AA
    assert LEGACY_EVAL_MODE == PIL_BICUBIC


def test_tv_aa_matches_the_previous_inline_implementation():
    """Bit-for-bit equal to the expression `datasets.py` used to inline.

    This is the load-bearing test of the whole refactor: the training
    distribution must be *identical*, not merely similar.
    """
    hr = _img(seed=7, h=64, w=64)
    lr_size_px = [max(1, hr.shape[-2] // 4), max(1, hr.shape[-1] // 4)]
    legacy = TF.resize(hr[0], lr_size_px,
                       interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
    new = build_lr(hr, 4, mode=TV_BICUBIC_AA)[0]
    assert torch.equal(legacy, new), (
        "training LR changed: the refactor altered the distribution every "
        "checkpoint was fitted on")


def test_sr_dataset_uses_the_shared_operator():
    """The dataset's LR equals build_lr(HR) -- the two cannot drift apart."""
    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, "div2k")
        os.makedirs(os.path.join(root, "val"), exist_ok=True)
        g = torch.Generator().manual_seed(3)
        for i in range(3):
            arr = (torch.rand(96, 96, 3, generator=g) * 255).to(torch.uint8).numpy()
            Image.fromarray(arr).save(os.path.join(root, "val", f"0{i}.png"))
        ds = SuperResolutionDataset(root, split="val", scale=2, crop_size=96,
                                    augment=False)
        lr, hr = ds[0]
        expected = build_lr(hr.unsqueeze(0), 2, mode=DEFAULT_MODE).squeeze(0)
        assert torch.equal(lr, expected)
        assert lr.shape[-2:] == (48, 48) and hr.shape[-2:] == (96, 96)


# ----------------------------------------------------- 2. modes are different


def test_antialias_flag_changes_the_image():
    """The axis that actually moves the number. Measured mean |diff| ~0.03 (x2)."""
    hr = _img(seed=11, h=96, w=96)
    aa = build_lr(hr, 4, mode=TV_BICUBIC_AA)
    noaa = build_lr(hr, 4, mode=TV_BICUBIC_NOAA)
    diff = (aa - noaa).abs()
    assert diff.max().item() > 0.05, (
        f"antialias on/off differ by only {diff.max():.4f} on a 4x downscale; "
        f"the P1..P6 protocol axis would be measuring noise")
    assert diff.mean().item() > 0.01


def test_pil_mode_is_within_quantisation_of_tv_aa():
    """PIL bicubic does NOT disagree with antialiased torchvision bicubic.

    This is the load-bearing *negative* result of the refactor. The repo had two
    LR constructors and the natural assumption is that train/eval were therefore
    mismatched. They were not: PIL scales its support kernel when shrinking, so
    the two agree to within 8-bit quantisation (measured max 0.006-0.014, mean
    0.002 at x2/x4). Writing "the two differ by 3 dB" would be quoting the
    antialias comparison, which is a different flag entirely.

    Pinned as a test so that a future "fix" to either mode cannot silently turn
    a 0.002 discrepancy into a real one and be justified by this module's name.
    """
    hr = _img(seed=13, h=96, w=96)
    pil = build_lr(hr, 4, mode=PIL_BICUBIC)
    tv = build_lr(hr, 4, mode=TV_BICUBIC_AA)
    d = (pil - tv).abs()
    assert d.max().item() < 0.02, (
        f"PIL vs torchvision-AA now differ by {d.max():.4f}; either mode changed "
        f"and the published eval numbers need re-checking")
    assert d.max().item() > 0.0, "identical outputs would mean one mode is a no-op"


def test_pil_mode_is_quantised_to_8bit():
    """PIL goes through uint8; that quantisation is part of the difference."""
    hr = _img(seed=17, h=32, w=32)
    pil = build_lr(hr, 2, mode=PIL_BICUBIC)
    scaled = pil * 255.0
    assert torch.allclose(scaled, scaled.round(), atol=1e-4), (
        "PIL path is not on the 8-bit grid -- it no longer reproduces what "
        "published pipelines actually do")


def test_all_modes_produce_the_right_shape():
    hr = _img(seed=19, h=50, w=70, batch=2)
    for mode in MODES:
        out = build_lr(hr, 4, mode=mode)
        assert out.shape == (2, 3, 12, 17), (mode, tuple(out.shape))


# --------------------------------------------------- 3. batching is irrelevant


def test_batch_invariance():
    """Batch of 4 must equal four batches of 1, for every mode."""
    hr = _img(seed=23, h=64, w=64, batch=4)
    for mode in MODES:
        together = build_lr(hr, 2, mode=mode)
        apart = torch.cat([build_lr(hr[i:i + 1], 2, mode=mode) for i in range(4)])
        assert torch.allclose(together, apart, atol=1e-6), mode


def test_pil_entry_point_matches_tensor_entry_point():
    """`build_lr_pil` (used by eval_baseline) must not be a second operator.

    For the PIL mode the two entry points are the same path and must match
    exactly. For the torchvision modes the PIL entry point returns a PIL image,
    i.e. an 8-bit round trip, so the only allowed difference is quantisation
    (<= 1/255) -- if it were larger, the two entry points would be different
    operators and eval_baseline would be measuring something training never saw.
    """
    hr = _img(seed=29, h=64, w=96)
    pil_in = TF.to_pil_image(hr[0])
    for mode in MODES:
        via_pil = TF.to_tensor(build_lr_pil(pil_in, 2, mode=mode)).unsqueeze(0)
        via_tensor = build_lr(hr, 2, mode=mode)
        d = (via_pil - via_tensor).abs()
        if mode == PIL_BICUBIC:
            assert d.max().item() < 1e-6, f"{mode}: two different PIL paths"
        else:
            # Compare on the mean, not the max: torchvision's `to_pil_image`
            # converts with a truncating cast (`(x*255).to(uint8)`) rather than
            # rounding, so the round trip sits systematically ~1 8-bit step low
            # (measured mean 0.0039 = exactly 1/255). On top of that, bicubic
            # overshoot at saturated pixels cannot be represented in 8-bit at
            # all -- with antialias off the ringing is strong enough to move
            # individual pixels by ~0.17. Both are properties of the container,
            # not of the operator, so the mean is the discriminating statistic.
            assert d.mean().item() <= 2.0 / 255, \
                f"{mode}: mean diff {d.mean():.6f} exceeds the 8-bit round trip"


# -------------------------------------------------------------------- edges


def test_lr_size_floors_at_one_pixel():
    assert lr_size(3, 3, 4) == [1, 1]
    assert lr_size(8, 16, 4) == [2, 4]
    with pytest.raises(ValueError):
        lr_size(8, 8, 0)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown degradation mode"):
        build_lr(_img(), 2, mode="bicubic_whatever")


def test_describe_covers_every_mode_and_names_the_training_one():
    for mode in MODES:
        text = describe(mode)
        assert text and isinstance(text, str)
    assert "training" in describe(DEFAULT_MODE)
    assert "legacy" in describe(LEGACY_EVAL_MODE)


# ------------------------------------------------------- no 8-bit round trip


def _step_edge(h=64, w=64, k=24):
    """A hard step edge: the input that makes bicubic ring the most."""
    hr = torch.zeros(1, 3, h, w)
    hr[:, :, (h - k) // 2:(h + k) // 2, (w - k) // 2:(w + k) // 2] = 1.0
    return hr


def test_float_modes_keep_bicubic_overshoot():
    """`build_lr` must not clamp to [0,1]. Training does not clamp either.

    Bicubic downscaling rings: a hard edge produces values below 0 and above 1.
    Those excursions carry real high-frequency energy; truncating them smooths
    the edge back. The cost is not theoretical -- the v1 protocol kernel passed
    every variant through `TF.to_pil_image` (an 8-bit container that clamps),
    and on these same weights that alone moved the score 0.69 dB on synthetic
    edges and is why its "tv_aa" point reads 25.55 dB where the same operator
    run in float reads 27.51 dB. A future "let's just clamp the input, it's an
    image" edit would silently reintroduce that gap.
    """
    hr = _step_edge()
    # Only the antialiased mode rings at 4x: its low-pass widens the bicubic
    # support, so neighbouring pixels overshoot. `noaa` at 4x degenerates to
    # near point-sampling and lands on exact 0/1, so it is not the case here.
    lr = build_lr(hr, 4, mode=TV_BICUBIC_AA)
    lo, hi = lr.min().item(), lr.max().item()
    assert lo < 0.0 or hi > 1.0, (
        f"{TV_BICUBIC_AA} returned a clamped range [{lo:.4f}, {hi:.4f}]; the "
        f"operator no longer matches the distribution training saw")
    # Quantify: clamping to [0,1] must actually cost something here, so the
    # assertion above is not merely about a handful of invisible pixels.
    clamped = lr.clamp(0.0, 1.0)
    assert (clamped - lr).abs().max().item() > 0.01


def test_pil_mode_is_inherently_8bit_and_that_is_declared():
    """The PIL mode clamps because its container does -- and says so."""
    lr = build_lr(_step_edge(), 4, mode=PIL_BICUBIC)
    assert lr.min().item() >= 0.0 and lr.max().item() <= 1.0
    assert "8-bit" in describe(PIL_BICUBIC)
