"""Tests that lock the evaluation protocol.

The project reports headline numbers (val PSNR / SSIM), and each of those
numbers is only meaningful together with the protocol that produced it. These
tests pin the three clauses that were previously implicit:

1. metrics are computed **per image** and then averaged, never from a batch-wide
   MSE (which makes the score a function of the batch size);
2. aggregation weights every image equally;
3. the LR degradation operator is fixed to ``antialias=True`` rather than
   inheriting a torchvision default that changed across versions.

Run from repository root:
    python -m train.tests.run_tests
    python -m pytest train/tests/test_protocol.py -q
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, "./train")
from metrics import aggregate, evaluate_batch, psnr, ssim  # noqa: E402
from datasets import SuperResolutionDataset  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. Per-image metrics must not see the batch
# --------------------------------------------------------------------------- #
def _difficulty_mixed_batch(n=6, seed=0):
    """A batch whose images are deliberately of very different quality.

    Uniform-quality batches cannot reveal batch-dependence: every image
    contributes the same MSE, so global and per-image aggregation coincide.
    The bug only shows up when the batch mixes easy and hard images, which is
    exactly what a real validation split does.
    """
    g = torch.Generator().manual_seed(seed)
    pred = [torch.rand(3, 32, 32, generator=g) for _ in range(n)]
    scales = [0.01, 0.05, 0.2, 0.01, 0.3, 0.1]
    tgt = [torch.clamp(p + s * torch.randn(p.shape, generator=g), 0, 1)
           for p, s in zip(pred, scales)]
    return torch.stack(pred), torch.stack(tgt)


def test_psnr_is_batch_invariant():
    pred, tgt = _difficulty_mixed_batch()
    whole = psnr(pred, tgt).item()

    singles = [psnr(pred[i:i + 1], tgt[i:i + 1]).item() for i in range(pred.shape[0])]
    by_image = sum(singles) / len(singles)

    assert abs(whole - by_image) < 1e-4, (
        f"PSNR depends on how images are batched: batch-of-6={whole:.4f}, "
        f"one-at-a-time mean={by_image:.4f}"
    )


def test_ssim_is_batch_invariant():
    pred, tgt = _difficulty_mixed_batch()
    whole = ssim(pred, tgt).item()
    singles = [ssim(pred[i:i + 1], tgt[i:i + 1]).item() for i in range(pred.shape[0])]
    by_image = sum(singles) / len(singles)
    assert abs(whole - by_image) < 1e-5, (
        f"SSIM depends on how images are batched: {whole:.6f} vs {by_image:.6f}"
    )


def test_psnr_is_not_a_global_mse():
    """Guard against a regression to batch-wide MSE.

    A batch-wide MSE is dominated by the hardest image, so it reports a value
    close to that image's PSNR instead of the mean over images. On this mixed
    batch the two must differ by a wide margin; if they agree, the metric has
    silently gone back to being batch-wide.
    """
    pred, tgt = _difficulty_mixed_batch()
    per_image_mean = psnr(pred, tgt).item()
    global_mse = torch.nn.functional.mse_loss(pred, tgt, reduction="mean").item()
    global_psnr = 10.0 * torch.log10(torch.tensor(1.0 / global_mse)).item()
    assert abs(per_image_mean - global_psnr) > 1.0, (
        "per-image mean and global-MSE PSNR are nearly identical on a batch with "
        f"mixed image difficulty ({per_image_mean:.2f} vs {global_psnr:.2f}); the "
        "metric is probably aggregating over the batch again"
    )


def test_evaluate_batch_per_image_lengths():
    pred, tgt = _difficulty_mixed_batch(n=5)
    out = evaluate_batch(pred, tgt, per_image=True)
    assert len(out["psnr_per_image"]) == 5
    assert len(out["ssim_per_image"]) == 5
    assert abs(aggregate(out["psnr_per_image"]) - out["psnr"]) < 1e-6


def test_aggregate_weights_images_equally():
    assert abs(aggregate([10.0, 20.0, 30.0]) - 20.0) < 1e-9
    # A duplicate image must pull the mean exactly as much as a duplicate entry.
    assert abs(aggregate([10.0, 20.0, 20.0]) - (50.0 / 3)) < 1e-9


def test_perfect_image_does_not_poison_the_metric():
    """One pixel-exact image is +inf PSNR; the batch mean must reflect that."""
    pred, tgt = _difficulty_mixed_batch(n=2)
    pred = pred.clone()
    pred[0] = tgt[0]  # image 0 is now perfect
    per = psnr(pred, tgt, reduction="none")
    assert torch.isinf(per[0])
    assert torch.isfinite(per[1])
    assert torch.isinf(per.mean())


# --------------------------------------------------------------------------- #
# 2. The degradation operator is pinned, not inherited
# --------------------------------------------------------------------------- #
def _write_png(path, size=(128, 128), seed=0):
    g = torch.Generator().manual_seed(seed)
    arr = (torch.rand(size[1], size[0], 3, generator=g) * 255).to(torch.uint8)
    Image.fromarray(arr.numpy()).save(path)


def test_sr_lr_uses_antialias_true():
    """The LR patch must equal an explicit antialias=True resize.

    torchvision flipped this default in 0.17; the project does not pin
    torchvision for training, so relying on the default means the degradation
    operator silently depends on which machine produced the numbers. On this
    project's x4 weights the flag alone flips the gain over bicubic from
    +0.57 dB to -1.19 dB.
    """
    with tempfile.TemporaryDirectory() as d:
        val_dir = os.path.join(d, "div2k", "val")
        os.makedirs(val_dir)
        _write_png(os.path.join(val_dir, "0001.png"), (128, 128))

        ds = SuperResolutionDataset(os.path.join(d, "div2k"), "val", scale=4)
        lr, hr = ds[0]

        expect = TF.resize(hr, [24, 24], interpolation=TF.InterpolationMode.BICUBIC,
                           antialias=True)
        no_aa = TF.resize(hr, [24, 24], interpolation=TF.InterpolationMode.BICUBIC,
                          antialias=False)

        assert torch.equal(lr, expect), (
            "LR patch is not the antialias=True result; the degradation operator "
            "is following a library default again"
        )
        assert not torch.equal(lr, no_aa), (
            "antialias=True and antialias=False produced identical LR patches on a "
            "high-frequency test image, so this test can no longer detect a switch"
        )


if __name__ == "__main__":
    failed = []
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {t.__name__}: {e}")
            failed.append(t.__name__)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
