"""The one place that builds a low-resolution image from a high-resolution one.

Why this module exists
----------------------
Until now the repo had **two** LR constructors:

* training, ``datasets.SuperResolutionDataset.__getitem__``::

      TF.resize(hr, size, interpolation=BICUBIC, antialias=True)

* evaluation, ``scripts/eval_baseline.py``::

      hr.resize(size, Image.BICUBIC)          # PIL, no antialias concept

Measuring them first, before assuming they disagreed, is how this module got
its shape -- and the measurement contradicts the obvious guess. On this repo's
images the two agree to within 8-bit quantisation (max |diff| 0.006-0.014,
mean 0.002) at both x2 and x4, because PIL's bicubic already scales its support
kernel when shrinking. **The duplicate code was a maintenance hazard, not a
silent train/eval distribution mismatch.** Anyone writing "the two differ by
3 dB" would be quoting the antialias comparison below.

What actually moves the number is the ``antialias`` flag. With it off, the
same x4 weights go from 22.66 to 19.35 dB on the same images, and the *gain
over bicubic* flips from +0.57 dB to -1.19 dB -- so whether the model "wins"
was being decided by a flag that appeared in no log, no config and no paper
table. That is the degradation-operator axis the paper needs to vary
deliberately (P1..P6 in the plan), and it is impossible to vary while the
operator lives in two copies.

Which mode is which
-------------------
``tv_bicubic_aa``    torchvision BICUBIC + antialias. **Training default**,
                     i.e. the distribution every shipped checkpoint was fitted
                     on. Any comparison that changes this while claiming to
                     compare models is comparing protocols, not models.
``tv_bicubic_noaa``  same kernel, antialias off. The "we did not write down
                     which one we used" case that most papers fall into.
``pil_bicubic``      ``PIL.Image.resize(..., BICUBIC)``. Still the default in
                     ``scripts/eval_baseline.py`` because every number already
                     published was produced that way. It sits within
                     quantisation noise of ``tv_bicubic_aa``, which is worth
                     stating in the paper rather than leaving to a reviewer to
                     discover: the published eval side was never mismatched
                     with training, it was merely undocumented.

``pil_bicubic`` is not merely "another kernel": PIL resizes through an 8-bit
intermediate, so its output is quantised even when the input is float. That
quantisation is part of the residual difference, and it is reported, not hidden.

Contract
--------
``build_lr`` is pure: same input, same mode, same output, bit for bit. Batch
handling never changes a pixel -- ``pil_bicubic`` is applied per image because
PIL has no batch API, and a test pins that a batch of one matches a batch of N.
"""

from __future__ import annotations

import torch
import torchvision.transforms.functional as TF
from PIL import Image

TV_BICUBIC_AA = "tv_bicubic_aa"
TV_BICUBIC_NOAA = "tv_bicubic_noaa"
PIL_BICUBIC = "pil_bicubic"

MODES = (TV_BICUBIC_AA, TV_BICUBIC_NOAA, PIL_BICUBIC)

#: What training uses. Named, not implied: this constant is the reason
#: ``datasets.py`` no longer spells out ``antialias=True`` inline.
DEFAULT_MODE = TV_BICUBIC_AA

#: What ``scripts/eval_baseline.py`` still defaults to, for backward
#: compatibility with already-published numbers.
LEGACY_EVAL_MODE = PIL_BICUBIC


def lr_size(h: int, w: int, scale: int) -> list[int]:
    """Integer LR size, never smaller than 1 px (a 3x3 HR at x4 still works)."""
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    return [max(1, h // scale), max(1, w // scale)]


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"unknown degradation mode {mode!r}; choose from {MODES}")


def _pil_one(img: torch.Tensor, size_wh: tuple[int, int]) -> torch.Tensor:
    """PIL bicubic on a single ``(3,H,W)`` float tensor.

    Goes through ``to_pil_image`` -> ``resize`` -> ``to_tensor``, i.e. through
    8-bit. That is the point: it reproduces what most published pipelines
    actually do, including the quantisation.
    """
    pil = TF.to_pil_image(img.clamp(0.0, 1.0))
    return TF.to_tensor(pil.resize(size_wh, Image.BICUBIC))


def build_lr(hr: torch.Tensor, scale: int, mode: str = DEFAULT_MODE) -> torch.Tensor:
    """Downscale ``hr`` ``(B,3,H,W)`` in [0,1] to ``(B,3,H/scale,W/scale)``.

    ``mode`` selects the degradation operator; see the module docstring for what
    each one means and why the default is the training one.
    """
    _check_mode(mode)
    if hr.dim() != 4:
        raise ValueError(f"expected (B,3,H,W), got {tuple(hr.shape)}")
    size = lr_size(hr.shape[-2], hr.shape[-1], scale)

    if mode == TV_BICUBIC_AA:
        return TF.resize(hr, size, interpolation=TF.InterpolationMode.BICUBIC,
                         antialias=True)
    if mode == TV_BICUBIC_NOAA:
        return TF.resize(hr, size, interpolation=TF.InterpolationMode.BICUBIC,
                         antialias=False)

    # PIL path: per image, then re-stack. Re-stacking with torch.stack keeps the
    # batch dimension explicit so a caller cannot mistake it for a single image.
    size_wh = (size[1], size[0])  # PIL wants (W, H)
    return torch.stack([_pil_one(img, size_wh) for img in hr], dim=0)


def build_lr_pil(img: Image.Image, scale: int,
                 mode: str = DEFAULT_MODE) -> Image.Image:
    """Same operator, but PIL in / PIL out.

    ``scripts/eval_baseline.py`` reads PIL images and hands PIL to the model
    loader, so it needs this entry point rather than a tensor one. The two
    entry points share :func:`build_lr`, so a change in the operator cannot
    reach one side and miss the other.
    """
    _check_mode(mode)
    w, h = img.size
    size = lr_size(h, w, scale)
    if mode == PIL_BICUBIC:
        return img.resize((size[1], size[0]), Image.BICUBIC)
    out = build_lr(TF.to_tensor(img).unsqueeze(0), scale, mode=mode)
    return TF.to_pil_image(out.squeeze(0).clamp(0.0, 1.0))


def describe(mode: str) -> str:
    """One-line description for logs, CSVs and the paper's protocol table."""
    _check_mode(mode)
    return {
        TV_BICUBIC_AA: "bicubic, float, antialias=True (training)",
        TV_BICUBIC_NOAA: "bicubic, float, antialias=False",
        PIL_BICUBIC: "bicubic, PIL, 8-bit intermediate (legacy eval)",
    }[mode]
