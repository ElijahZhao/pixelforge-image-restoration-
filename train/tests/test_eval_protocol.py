"""Golden tests for the literature-comparable protocol (``eval_protocol.py``).

The point of these tests is narrower than ``test_protocol.py``: that file pins
the aggregation policy and the degradation operator, this one pins the
**coordinate-system conversion** (RGB -> Y, border shave) whose numbers get
compared against published tables.

Every assertion below is either
  (a) analytically exact (derived by hand, written in the comment), or
  (b) a bit-exactness check against the kernel implementation that produced the
      published 28.91 / 33.82 dB figures.

Run:
    python -m pytest train/tests/test_eval_protocol.py -q
"""

from __future__ import annotations

import sys

import pytest
import torch

sys.path.insert(0, "./train")
from eval_protocol import evaluate_y, psnr_y, rgb_to_y, shave  # noqa: E402
from metrics import psnr_per_image  # noqa: E402


def _img(seed: int, h: int = 32, w: int = 32) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, h, w, generator=g)


# --------------------------------------------------------------- Y transform


def test_y_primaries_match_bt601_studio_swing():
    """Pure primaries -> the textbook 8-bit Y values 81 / 145 / 41.

    Derivation: Y = 16 + 65.481*R + 128.553*G + 24.966*B  (R,G,B in 0..255)
    red   -> 16 + 65.481  = 81.481  -> /255 = 0.3195
    green -> 16 + 128.553 = 144.553 -> /255 = 0.5669
    blue  -> 16 + 24.966  = 40.966  -> /255 = 0.1607
    """
    px = torch.zeros(1, 3, 1, 1)
    px[0, 0, 0, 0] = 1.0                       # pure red
    assert rgb_to_y(px).item() == pytest.approx(81.481 / 255, abs=1e-7)
    px.zero_(); px[0, 1, 0, 0] = 1.0           # pure green
    assert rgb_to_y(px).item() == pytest.approx(144.553 / 255, abs=1e-7)
    px.zero_(); px[0, 2, 0, 0] = 1.0           # pure blue
    assert rgb_to_y(px).item() == pytest.approx(40.966 / 255, abs=1e-7)


def test_y_of_gray_is_affine_with_weight_sum_219():
    """R=G=B=v -> Y = (16 + 219*v)/255, because 65.481+128.553+24.966 == 219.0.

    This is exactly why a constant RGB offset of d moves Y by 219/255*d, which
    the next test uses to predict PSNR analytically.
    """
    v = torch.tensor([0.0, 0.25, 0.5, 1.0]).view(1, 1, 1, 4).expand(1, 3, 1, 4)
    got = rgb_to_y(v.contiguous())[0, 0, 0]
    want = (16.0 + 219.0 * torch.tensor([0.0, 0.25, 0.5, 1.0])) / 255
    assert torch.allclose(got, want, atol=1e-6)


def test_psnr_y_matches_analytic_for_gray_offset():
    """gray target vs gray target+d (everything else equal).

    Y error is c*d with c = 219/255, so
        PSNR = 10*log10(1 / (c*d)^2)          [max_val = 1 in [0,1] domain]
    which is the same value as the 0-255 domain because both MSE and MAX^2
    scale by 1/255^2.
    """
    base = torch.full((1, 3, 16, 16), 0.5)
    # 2**-7: exactly representable in float32, so the only rounding left is the
    # luma-weight sum itself; tolerance 1e-4 dB is far below any real error.
    d = 2.0 ** -7
    pred = base + d
    c = 219.0 / 255.0
    want = 10 * torch.log10(torch.tensor(1.0 / (c * d) ** 2, dtype=torch.float64))
    got = psnr_y(pred, base, border=0).item()
    assert got == pytest.approx(want.item(), abs=1e-4)


def test_psnr_y_infinite_for_identical_images():
    x = _img(0)
    assert psnr_y(x, x, border=0).item() == float("inf")


# --------------------------------------------------------------------- shave


def test_shave_is_exact_crop_and_noop_at_zero():
    x = _img(1, 24, 28)
    assert torch.equal(shave(x, 0), x)
    assert torch.equal(shave(x, 4), x[..., 4:20, 4:24])
    assert shave(x, 4).shape[-2:] == (16, 20)


def test_shave_removes_border_only_damage():
    """Damage confined to the border must vanish after shave, and only then.

    Unshaved PSNR must be finite; shaved PSNR must be +inf. This is the
    property papers rely on when they say "we shave to avoid boundary effects".
    """
    clean = _img(2, 32, 32)
    dirty = clean.clone()
    dirty[..., :4, :] = 1.0 - dirty[..., :4, :]
    dirty[..., -4:, :] = 1.0 - dirty[..., -4:, :]
    dirty[..., :, :4] = 1.0 - dirty[..., :, :4]
    dirty[..., :, -4:] = 1.0 - dirty[..., :, -4:]

    assert torch.isfinite(psnr_y(dirty, clean, border=0)).all()
    assert (psnr_y(dirty, clean, border=4) == float("inf")).all()


def test_shave_too_large_raises_instead_of_silently_shrinking():
    x = _img(3, 8, 8)
    with pytest.raises(ValueError):
        shave(x, 4)


# --------------------------------------------------- equivalence with kernel
def test_matches_kernel_impl_bit_exact():
    """The published y_shave numbers came from this exact expression.

    Copied verbatim from `run_eval.py` of kernel `pixelforge-eval-protocol`
    (archived in ``archive_private/06-实验脚本与输出/kag-eval-protocol/``).
    If the module ever drifts, 28.91 / 33.82 stop being reproducible from the
    repo -- this test makes that drift a red build instead of a silent one.
    """

    def kernel_rgb_to_y(t):
        r, g, b = t[:, 0:1], t[:, 1:2], t[:, 2:3]
        return 65.481 / 255 * r + 128.553 / 255 * g + 24.966 / 255 * b + 16 / 255

    x = _img(4, 40, 40)
    assert torch.equal(rgb_to_y(x), kernel_rgb_to_y(x))

    # And the full pipeline: shave(scale) on both sides, then per-image PSNR.
    pred = _img(5, 40, 40)
    y_p = shave(kernel_rgb_to_y(pred), 4)
    y_t = shave(kernel_rgb_to_y(x), 4)
    assert torch.equal(psnr_y(pred, x, border=4), psnr_per_image(y_p, y_t))


# ------------------------------------------------------------- batch invariant
def test_y_protocol_is_batch_invariant():
    """Splitting a batch must not change any image's score."""
    a = _img(6, 32, 32)
    b = _img(7, 32, 32)
    both = torch.cat([a, b], 0)
    ref = torch.cat([a, a], 0)  # dummy target with same shape

    full = psnr_y(both, torch.cat([ref[0:1], ref[0:1]], 0).expand(2, -1, -1, -1), 4)
    s1 = psnr_y(both[0:1], ref[0:1], 4)
    s2 = psnr_y(both[1:2], ref[0:1], 4)
    assert full[0].item() == pytest.approx(s1[0].item(), abs=1e-9)
    assert full[1].item() == pytest.approx(s2[0].item(), abs=1e-9)


def test_evaluate_y_reports_protocol_name():
    x = _img(8)
    r = evaluate_y(x, x + 0.01, border=4)
    assert r["protocol"] == "y_shave(border=4)"
    assert r["psnr"] == pytest.approx(sum(r["psnr_per_image"]) / 1, abs=1e-9)
    r0 = evaluate_y(x, x + 0.01, border=0)
    assert r0["protocol"] == "y(border=0)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
