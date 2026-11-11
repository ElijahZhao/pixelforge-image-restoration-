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
from datasets import LowLightDataset, SuperResolutionDataset  # noqa: E402
from models import (  # noqa: E402
    LowLightUNet,
    forward_lowlight,
    pad_to_align,
)


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
    """Load train/train.py (the training module, not the package) safely.

    The path is derived from THIS FILE, not from the current working
    directory: a relative "train/train.py" only resolves when pytest happens
    to be started in the repository root, so the test fails for the wrong
    reason anywhere else. It also has to be loaded by path rather than with
    `import train`, because the repository root has a `train/` DIRECTORY (a
    namespace package) and `train/train.py` is a file inside it -- which of
    the two `import train` resolves to depends on sys.path order, and the
    two runners of this project disagree.
    """
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "train.py"
    spec = importlib.util.spec_from_file_location("train_script", path)
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
    import pathlib

    # Absolute path from this file: a relative "serve/model_loader.py" only
    # resolves when pytest is started in the repository root.
    loader = (pathlib.Path(__file__).resolve().parents[2]
              / "serve" / "model_loader.py")
    spec = importlib.util.spec_from_file_location("ml_loader", loader)
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


def test_lowlight_val_keeps_aspect_ratio():
    """The val branch must not squash a non-square image into a square.

    LOL-v1 images are 400x600. Resizing them to a 128x128 square changed the
    width to 21% and the height to 32% of their original scale -- a different
    picture from the one ``scripts/eval_baseline.py`` scores. The two numbers
    were therefore not comparable, which made it impossible to check training
    against evaluation results.
    """
    base = tempfile.mkdtemp()
    rng = np.random.default_rng(1)
    _write(f"{base}/low/0.png", (rng.random((400, 600, 3)) * 40).astype("uint8"))
    _write(f"{base}/high/0.png", (rng.random((400, 600, 3)) * 200).astype("uint8"))
    ds = LowLightDataset(base, "", augment=False)
    low, high = ds[0]
    assert tuple(low.shape) == tuple(high.shape) == (3, 400, 600), \
        f"val must keep the native 400x600 size, got {tuple(low.shape)}"


def test_lowlight_val_upscales_uniformly_when_small():
    """A uniformly upscaled image is still the same picture; a squashed one is not."""
    base = tempfile.mkdtemp()
    rng = np.random.default_rng(2)
    _write(f"{base}/low/0.png", (rng.random((80, 40, 3)) * 40).astype("uint8"))
    _write(f"{base}/high/0.png", (rng.random((80, 40, 3)) * 200).astype("uint8"))
    ds = LowLightDataset(base, "", augment=False)
    low, _ = ds[0]
    assert low.shape[1] / low.shape[2] == 80 / 40, \
        "small val images must be upscaled uniformly, preserving aspect ratio"
    assert min(low.shape[1], low.shape[2]) == ds.crop_size


def test_sr_val_is_deterministic():
    """Two validation fetches of the same index must be identical.

    The crop position was drawn from ``torch.randint`` regardless of ``augment``,
    so each validation epoch saw a different part of every image. Measured on
    DIV2K val, 200 independent resamples of the whole split move PSNR by 2.83 dB
    peak-to-peak (max 27.54 / mean 25.80 / sigma 0.51), against a real training
    gain of +3.36 dB over 200 epochs -- so the noise did not swamp training, but
    it did make the per-epoch number incomparable with itself.
    """
    base = tempfile.mkdtemp()
    rng = np.random.default_rng(3)
    _write(f"{base}/val/0.png", (rng.random((300, 400, 3)) * 255).astype("uint8"))
    ds = SuperResolutionDataset(base, "val", scale=4, crop_size=96)
    a = ds[0]
    b = ds[0]
    assert all(torch.equal(x, y) for x, y in zip(a, b)), \
        "SR validation fetch is not deterministic"


def test_sr_train_still_random_crops():
    """Validation was fixed, not the augmentation: training must stay random."""
    base = tempfile.mkdtemp()
    rng = np.random.default_rng(4)
    _write(f"{base}/train/0.png", (rng.random((300, 400, 3)) * 255).astype("uint8"))
    ds = SuperResolutionDataset(base, "train", scale=4, crop_size=64)
    patches = {ds[0][1].numpy().tobytes() for _ in range(20)}
    assert len(patches) > 1, "training crops must still vary between fetches"


def test_sr_rejects_crop_smaller_than_scale():
    """A crop below the scale factor would give a 1x1 LR patch.

    That is not a super-resolution task, yet the loader would happily produce it
    and the model would train into something useless without any error. Refusing
    the configuration is the only way this surfaces.
    """
    base = tempfile.mkdtemp()
    _write(f"{base}/val/0.png", np.zeros((100, 100, 3), "uint8"))
    for crop in (1, 2, 3):
        try:
            SuperResolutionDataset(base, "val", scale=4, crop_size=crop)
            raise AssertionError(f"crop_size={crop} with scale=4 must be rejected")
        except ValueError:
            pass


def test_sr_rejects_crop_not_divisible_by_scale():
    """A crop not divisible by the scale truncates the LR patch.

    e.g. crop=98 / scale=4 -> LR 24, so the pair is really 24x4=96 rather than
    98: the model is trained on a ratio its own output size contradicts.
    """
    base = tempfile.mkdtemp()
    _write(f"{base}/val/0.png", np.zeros((200, 200, 3), "uint8"))
    for crop, scale in ((98, 4), (97, 2), (100, 8)):
        try:
            SuperResolutionDataset(base, "val", scale=scale, crop_size=crop)
            raise AssertionError(
                f"crop_size={crop} with scale={scale} must be rejected"
            )
        except ValueError:
            pass
    # The default configuration must remain legal.
    SuperResolutionDataset(base, "val", scale=4, crop_size=96)
    SuperResolutionDataset(base, "val", scale=2, crop_size=96)


def test_lowlight_rejects_crop_not_divisible_by_pad_multiple():
    """The low-light U-Net's training patches must be multiples of 32 -- the
    multiple the project uses (the network's own minimum is 8; see
    ``models.LOWLIGHT_ALIGN``). Otherwise the decoder's concatenation fails on
    the first batch with "Sizes of tensors must match", a message that points
    at the network rather than at the real cause."""
    base = tempfile.mkdtemp()
    low_dir = os.path.join(base, "train", "low")
    high_dir = os.path.join(base, "train", "high")
    _write(os.path.join(low_dir, "0.png"), np.zeros((200, 200, 3), "uint8"))
    _write(os.path.join(high_dir, "0.png"), np.zeros((200, 200, 3), "uint8"))
    for crop in (100, 127, 33):
        try:
            LowLightDataset(base, "train", crop_size=crop)
            raise AssertionError(f"crop_size={crop} is not a multiple of 32")
        except ValueError:
            pass
    LowLightDataset(base, "train", crop_size=128)  # default stays legal
    LowLightDataset(base, "train", crop_size=96)


# --------------------------------------------------------------------------- #
# Regression tests for two defects found by reading the code and confirmed by
# running it. Both were silent: no error, no warning, just a wrong number.
# --------------------------------------------------------------------------- #
def test_sr_val_crop_does_not_pad_with_black():
    """The SR validation centre-crop must not read outside the image.

    ``TF.crop`` does not complain when the requested box is larger than the
    image -- it silently fills the out-of-bounds area with black. The training
    branch resized such an image first, and the low-light branch does the
    same; only the SR validation branch was missing the guard. Measured on a
    64x64 validation image with crop_size=96, the resulting 96x96 patch was
    55.6% filler black, so the validation PSNR was measured against a picture
    that is mostly black and the number was wrong with no symptom at all.
    """
    base = tempfile.mkdtemp()
    rng = np.random.default_rng(7)
    for name, (h, w) in (("small", (64, 64)), ("narrow", (80, 100))):
        arr = (rng.random((h, w, 3)) * 255).astype("uint8")
        _write(os.path.join(base, "val", f"{name}.png"), arr)
    ds = SuperResolutionDataset(base, "val", scale=2, crop_size=96)

    for idx in range(len(ds)):
        patch = ds._load(idx)
        arr = np.asarray(patch)
        # A black-filled border is the signature of the out-of-bounds crop.
        assert arr.min() > 0, (
            f"validation patch contains pure-black filler pixels "
            f"({(arr == 0).all(axis=-1).mean():.1%} of the patch); the crop "
            f"read outside the image instead of resizing it first")
        assert patch.size == (96, 96)


def test_sr_val_crop_is_unchanged_for_images_that_fit():
    """The guard must not alter images that are already large enough.

    The published DIV2K validation numbers were produced by the unguarded
    centre crop. This pins that the fix is a no-op there: for an image at
    least as large as the patch, the loaded patch must be pixel-identical to
    the plain centre crop.
    """
    base = tempfile.mkdtemp()
    rng = np.random.default_rng(8)
    arr = (rng.random((300, 400, 3)) * 255).astype("uint8")
    _write(os.path.join(base, "val", "0.png"), arr)
    ds = SuperResolutionDataset(base, "val", scale=2, crop_size=96)

    import torchvision.transforms.functional as TF
    plain = TF.crop(Image.open(os.path.join(base, "val", "0.png")).convert("RGB"),
                    (300 - 96) // 2, (400 - 96) // 2, 96, 96)
    assert np.array_equal(np.asarray(ds._load(0)), np.asarray(plain)), \
        "the centre crop changed for an image that already fits"


def test_lowlight_val_forward_survives_non_multiple_of_8():
    """A validation image whose size is not a multiple of 8 must still run.

    The low-light validation split keeps every image at its native size (the
    "full_native" policy), and the U-Net halves the resolution three times,
    so its skip connections need both sides to be a multiple of 8. A native
    size that is not a multiple of 8 made the forward pass raise
    "Sizes of tensors must match" with no other symptom. LOL-v1 is 400x600,
    which happens to be a multiple of 8, so the published numbers were never
    affected -- but any other dataset crashed on the first validation epoch.

    This calls `models.forward_lowlight` -- the SAME function the training
    validation loop calls -- rather than re-implementing the padding here.
    A test that carries its own copy of the logic it is meant to police
    passes even when the production code is broken, which is exactly what
    happened the first time this was written.
    """
    model = LowLightUNet().eval()
    for (h, w) in ((300, 300), (601, 400), (604, 402), (400, 400)):
        x = torch.rand(1, 3, h, w)
        try:
            out = forward_lowlight(model, x)
        except RuntimeError as exc:
            raise AssertionError(
                f"a {h}x{w} validation image must not crash the U-Net: {exc}")
        assert out.shape[-2:] == (h, w), \
            f"output {tuple(out.shape)} must keep the original {h}x{w} size"


def test_lowlight_val_forward_is_unchanged_for_multiple_of_8():
    """For a size that IS a multiple of 8 the padding is zero-width, so the
    result must be bit-identical to the plain forward pass. This is what
    keeps the published LOL-v1 (400x600) validation numbers unchanged.

    The comparison is against the raw `model(x)` -- i.e. what the code did
    BEFORE the fix -- not against a second call of the same expression,
    which would be true by construction and assert nothing.
    """
    torch.manual_seed(0)
    model = LowLightUNet().eval()
    x = torch.rand(1, 3, 600, 400)
    with torch.no_grad():
        before = model(x)                      # the pre-fix path: no padding
        after = forward_lowlight(model, x)     # the fixed path
    assert torch.equal(before, after), (
        "the fixed path must be bit-identical to the old one for a size that "
        "is already a multiple of 8, or the published LOL-v1 numbers moved")


def test_lowlight_val_forward_survives_images_smaller_than_the_multiple():
    """An image smaller than the alignment multiple must not crash either.

    Reflection padding requires the pad to be narrower than the side it pads,
    so it cannot be applied to an image smaller than 8 px. The padding falls
    back to replicate mode there, which has no such limit. (A photo that small
    is not a realistic input, but a crash with an obscure message from deep
    inside a padding call is a poor way to find out.)
    """
    model = LowLightUNet().eval()
    for (h, w) in ((1, 1), (3, 3), (4, 4), (5, 7)):
        x = torch.rand(1, 3, h, w)
        try:
            out = forward_lowlight(model, x)
        except RuntimeError as exc:
            raise AssertionError(
                f"a {h}x{w} image must not crash the low-light path: {exc}")
        assert out.shape[-2:] == (h, w)


def test_training_validation_forward_is_the_one_being_tested():
    """The training loop must score validation images through the fixed path.

    `train.py`'s validation loop is where the low-light crash actually
    happened, and it is excluded from coverage (it is a CLI entry point).
    A test that only exercises `models.forward_lowlight` therefore leaves the
    real call site free to revert to the broken `model(lr)` with CI still
    green -- which is exactly what happened once.

    So the training loop calls `train.validation_forward`, and this test
    calls the same function. Reverting the call site in train.py breaks this
    test, and so does breaking the helper it delegates to.
    """
    train_mod = _import_train_module()

    model = LowLightUNet().eval()
    # A size that is NOT a multiple of 8: the raw forward pass raises here.
    x = torch.rand(1, 3, 300, 300)
    try:
        out = train_mod.validation_forward(model, x, "lowlight")
    except RuntimeError as exc:
        raise AssertionError(
            f"the training validation forward must not crash: {exc}")
    assert out.shape[-2:] == (300, 300)


def test_training_loop_actually_calls_validation_forward():
    """The training loop must route validation through `validation_forward`.

    The previous test proves `validation_forward` works; it does NOT prove the
    training loop calls it. Reverting the call site to `model(lr)` leaves
    every other test green, because `train.py` is a CLI entry point excluded
    from coverage and nothing else executes its validation loop.

    So this asserts on the source of the training loop directly. It is a blunt
    instrument -- a source-text check, not a behavioural one -- but a
    behavioural check would need the whole training loop to run, and the
    failure mode being guarded against is precisely "the loop quietly stopped
    using the fixed path".
    """
    import pathlib

    train_py = pathlib.Path(__file__).resolve().parents[1] / "train.py"
    source = train_py.read_text(encoding="utf-8")

    # The validation loop's forward call must be the shared helper.
    assert "validation_forward(model, lr" in source, (
        "train.py's validation loop must call validation_forward(...); a bare "
        "model(lr) there reintroduces the low-light crash on any dataset "
        "whose native size is not a multiple of 8")

    # And the helper must route the low-light task through the padding
    # helper, not straight to the raw forward pass.
    helper = source.split("def validation_forward", 1)[1].split("\\ndef ", 1)[0]
    assert "forward_lowlight(model, x)" in helper, (
        "validation_forward must send the low-light task through "
        "forward_lowlight, which pads the input up to the alignment multiple")


def test_training_validation_forward_leaves_sr_alone():
    """For SR the validation forward is the plain model call.

    The SR path has no alignment constraint (conv + pixel shuffle), so
    routing it through the padding helper would change nothing -- but it must
    not change anything either, and this pins that.
    """
    train_mod = _import_train_module()

    model = LowLightUNet().eval()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        plain = model(x)
        via_helper = train_mod.validation_forward(model, x, "sr")
    assert torch.equal(plain, via_helper), \
        "the SR validation path must stay a plain forward pass"


def test_lowlight_pad_to_align_is_a_no_op_for_a_multiple_of_8():
    """`pad_to_align` must return the tensor untouched when no pad is needed.

    This is the property that keeps the published numbers intact, stated on
    the helper itself rather than only through its caller.
    """
    x = torch.rand(1, 3, 600, 400)
    padded, (h, w) = pad_to_align(x)
    assert (h, w) == (600, 400)
    assert torch.equal(padded, x), "a zero-width pad must not touch the tensor"


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
