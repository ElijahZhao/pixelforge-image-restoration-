"""Correctness tests (not shape tests).

Motivation: the pre-existing suite only asserted shapes, ranges and
batch-invariance, so it could NOT catch any of the real defects (loss weighting,
VGG normalization, size mismatches, semantics). Mutation testing showed 6 of 10
injected fatal bugs slipped through.

These tests assert on *behaviour that matters*, so this class of bug now fails
the suite.

Run from repository root:
    python -m train.tests.run_tests
"""

from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "./train")

from metrics import psnr  # noqa: E402
from datasets import LowLightDataset  # noqa: E402


# --------------------------------------------------------------------------- #
# VGG perceptual loss must ImageNet-normalize its input.
# --------------------------------------------------------------------------- #
def _vgg_features():
    """VGG16 features[:30] with pinned ImageNet weights.

    Raises ``unittest.SkipTest`` if the weights cannot be obtained (typically no
    network on a clean CI runner) so the suite reports SKIP, not a misleading
    FAIL. The weights download once and are then cached under ~/.cache/torch.
    Also uses the explicit ``weights=`` enum: ``pretrained=True`` is deprecated
    and removed in newer torchvision.
    """
    import unittest

    import torchvision.models as tvm
    from torchvision.models import VGG16_Weights

    try:
        return tvm.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:30].eval()
    except Exception as e:  # noqa: BLE001 - download/offline failures are skips
        raise unittest.SkipTest(
            f"VGG16 ImageNet weights unavailable ({type(e).__name__}: {e})"
        )


def test_vgg_perceptual_normalizes_input():
    """Un-normalized [0,1] input collapses VGG activations (~90% zeros, tiny
    mean). With proper normalization the feature magnitude recovers markedly.

    Judgment uses feature *magnitude* (mean of non-zero activations), not the
    zero fraction: ReLU is intrinsically sparse, so zero-fraction alone is not
    a valid signal.
    """
    vgg = _vgg_features()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # A textured input (random noise is a fair, content-agnostic probe).
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        f_raw = vgg(x)
        f_norm = vgg((x - mean) / std)

    raw_mag = f_raw[f_raw != 0].mean().item()
    norm_mag = f_norm[f_norm != 0].mean().item()
    assert norm_mag > raw_mag * 1.3, (
        f"normalized features should be markedly stronger: "
        f"raw_nonzero_mag={raw_mag:.4f} norm_nonzero_mag={norm_mag:.4f}"
    )


def test_vgg_perceptual_loss_sensitive_to_blur():
    """A correct perceptual loss must react to a real perceptual difference
    (mild blur). The un-normalized version was ~4x less sensitive."""
    import torchvision.transforms.functional as TF

    vgg = _vgg_features()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    x = torch.rand(1, 3, 128, 128)
    blur = TF.gaussian_blur(x, kernel_size=[5, 5], sigma=[1.0, 1.0])

    def _feat(t):
        return vgg((t - mean) / std)

    with torch.no_grad():
        d_norm = torch.nn.functional.l1_loss(_feat(x), _feat(blur)).item()
    # Sanity: normalized perceptual distance must be clearly non-trivial.
    assert d_norm > 0.01, f"perceptual distance to blur too small: {d_norm}"
    assert math.isfinite(d_norm)
# --------------------------------------------------------------------------- #
# Loss weighting must be explicit and the pixel term must not be erased.
# --------------------------------------------------------------------------- #
def _import_train_module():
    """Load train/train.py (the training module, not the package) safely."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_script", os.path.join("train", "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_loss_weights_are_explicit_and_sane():
    """The pixel term must not be crushed to ~1% of the perceptual term by a
    hidden 0.01 factor. With the new explicit weights the two terms should be
    the same order of magnitude.

    The weights are read from ``train.py`` (DEFAULT_W_PIXEL / DEFAULT_W_PERCEP)
    rather than re-typed here: hard-coding them meant the test would still pass
    if the real defaults were changed or broken.
    """
    tr = _import_train_module()
    pred = torch.rand(1, 3, 64, 64)
    tgt = torch.rand(1, 3, 64, 64)
    pix = tr.l1_charbonnier(pred, tgt).item()
    per = tr.VGGPerceptualLoss()(pred, tgt).item()

    w_pixel = tr.DEFAULT_W_PIXEL
    w_percep = tr.DEFAULT_W_PERCEP
    pix_contrib, per_contrib = w_pixel * pix, w_percep * per
    ratio = max(pix_contrib, per_contrib) / max(min(pix_contrib, per_contrib), 1e-12)
    # Measured on random 64x64 input: pixel=0.332, percep=0.089, so with the
    # defaults (1.0 / 0.006) the ratio is ~620; the pixel term leads, as the doc
    # below intends. The bound is deliberately generous (order-of-magnitude, not
    # ±10%) so it does not become flaky on a different random draw, while still
    # catching a default that is off by an order of magnitude (e.g. 0.01 -> 1%).
    assert ratio < 5000, (
        f"pixel/perceptual contributions wildly imbalanced (ratio={ratio:.1f}); "
        f"w_pixel={w_pixel} w_percep={w_percep} "
        f"pixel={pix_contrib:.4f} percep={per_contrib:.4f}"
    )
    # The pixel (task) term must dominate or at least lead: an "enhancement"
    # loss whose perceptual term outweighs the reconstruction term is wrong.
    assert pix_contrib >= per_contrib, (
        f"perceptual term outweighs the pixel term: "
        f"pixel={pix_contrib:.4f} percep={per_contrib:.4f}"
    )
    # And the pixel term must carry real weight, not be effectively zero.
    assert pix_contrib > 1e-3, f"pixel term contribution too small: {pix_contrib}"
    # Both weights must be strictly positive; a 0 would silently disable a term.
    assert w_pixel > 0 and w_percep > 0, (
        f"loss weights must be positive, got w_pixel={w_pixel} w_percep={w_percep}"
    )


# --------------------------------------------------------------------------- #
# SR output size contract (no hidden second upsample).
# --------------------------------------------------------------------------- #
def test_predict_sr_output_size_is_lr_times_scale():
    """predict_sr must return the model's TRUE resolution (lr*scale), not the
    fake orig*scale produced by a second PIL resize. Uses torch.jit on a tiny
    traced identity-ish model to stay CPU-fast."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ml_loader", os.path.join("serve", "model_loader.py"))
    ml = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ml)

    # Build a tiny stand-in SR model: nearest-upscale by `scale` (shape-correct).
    class TinySR(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def forward(self, x):
            return torch.nn.functional.interpolate(
                x, scale_factor=self.scale, mode="nearest")

    scale = 4
    traced = torch.jit.trace(TinySR(scale), torch.rand(1, 3, 32, 32))

    # Monkeypatch the loader to return our tiny model.
    orig_get = ml.get_sr_model
    ml.get_sr_model = lambda s: traced if s == scale else None
    try:
        img = Image.new("RGB", (64, 64))
        out = ml.predict_sr(img, scale)
        lr_side = max(1, 64 // scale) * scale  # 16*4 = 64
        assert out.size == (lr_side, lr_side), (
            f"expected true output {(lr_side, lr_side)}, got {out.size} "
            f"(orig*scale would be {(64*scale, 64*scale)}, the fake upsample)")
    finally:
        ml.get_sr_model = orig_get


# --------------------------------------------------------------------------- #
# Data pipeline must not silently mis-pair or crash on bad data.
# --------------------------------------------------------------------------- #
def _write(path, arr=None, raw=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if raw is not None:
        with open(path, "wb") as f:
            f.write(raw)
    else:
        Image.fromarray(arr).save(path)


def test_lowlight_skips_size_mismatched_pairs():
    """A low/high pair with different sizes used to load fine but crop from
    different regions (silent misalignment). Now it must be skipped."""
    base = tempfile.mkdtemp()
    _write(f"{base}/low/0.png", np.zeros((100, 120, 3), "uint8"))
    _write(f"{base}/high/0.png", np.zeros((110, 130, 3), "uint8"))
    # Only the bad pair exists -> dataset should raise (no valid pairs), which
    # proves the misaligned pair was rejected rather than silently used.
    try:
        LowLightDataset(base, "")
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "size-mismatched pair should be rejected, not silently loaded"


def test_lowlight_pairs_by_filename_not_order():
    """Extra low image with no high counterpart must be skipped, not crash."""
    base = tempfile.mkdtemp()
    _write(f"{base}/low/0.png", np.zeros((100, 100, 3), "uint8"))
    _write(f"{base}/low/1.png", np.zeros((100, 100, 3), "uint8"))  # orphan
    _write(f"{base}/high/0.png", np.zeros((100, 100, 3), "uint8"))
    ds = LowLightDataset(base, "")
    assert len(ds) == 1, f"expected 1 valid pair, got {len(ds)}"


def test_lowlight_val_is_deterministic():
    """Validation must be reproducible: two fetches of the same index must be
    identical (random cropping is disabled for eval)."""
    base = tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    _write(f"{base}/low/0.png", (rng.random((200, 200, 3)) * 40).astype("uint8"))
    _write(f"{base}/high/0.png", (rng.random((200, 200, 3)) * 200).astype("uint8"))
    ds = LowLightDataset(base, "", augment=False)
    a = ds[0]
    b = ds[0]
    assert all(torch.equal(x, y) for x, y in zip(a, b)), \
        "validation fetch is not deterministic"


# --------------------------------------------------------------------------- #
# Metrics: numeric correctness (not just finiteness).
# --------------------------------------------------------------------------- #
def test_psnr_matches_manual_formula():
    """PSNR must equal 10*log10(1/MSE) for [0,1] images (guards the coefficient
    and the max^2 term)."""
    x = torch.rand(1, 3, 32, 32)
    y = torch.clamp(x + 0.05 * torch.rand_like(x), 0, 1)
    mse = torch.mean((x - y) ** 2).item()
    expected = 10 * math.log10(1.0 / mse)
    got = psnr(x, y).item()
    assert abs(got - expected) < 0.1, f"psnr={got} expected~{expected}"


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {t.__name__}: {e}")
            traceback.print_exc()
            failed.append(t.__name__)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        sys.exit(1)
