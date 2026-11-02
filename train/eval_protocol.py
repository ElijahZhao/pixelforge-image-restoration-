"""Literature-comparable evaluation protocol: Y channel + border shave + full image.

Why this module exists
----------------------
The rest of this repo reports **RGB full-channel, full-image, no shave** numbers
(``metrics.py``).  Nearly every SR paper (SRCNN/SRResNet/EDSR/RCAN, and the NTIRE
challenge protocol) instead reports **PSNR/SSIM on the Y channel of YCbCr with a
border shave of ``scale`` pixels**.  The two coordinate systems differ by a
measured **2.4-2.8 dB** on this repo's own weights, so a number from one cannot
be placed in a table built from the other.

This module implements the literature side of that conversion so that the
comparison is a computed number rather than an extrapolated one.

Definitions (each one is a choice; state it in the paper)
---------------------------------------------------------
* **Y** -- BT.601 studio swing, identical to MATLAB ``rgb2ycbcr``::

      Y = (16 + 65.481 R + 128.553 G + 24.966 B) / 255      R,G,B in [0,1]
      Y in [16/255, 235/255]

  The affine map keeps PSNR numerically identical to the 0-255 domain (both
  MSE and MAX**2 scale by 1/255**2), so values are directly comparable to
  published tables.

* **shave** -- crop ``border`` pixels from each side of *all three* maps before
  measuring.  Default ``border = scale`` (the EDSR convention); the NTIRE
  challenge uses a fixed 4 px, which for x4 is the same thing.

* **SSIM** -- Gaussian window 11x11, sigma 1.5, ``C1=(0.01L)**2``,
  ``C2=(0.03L)**2`` (MATLAB ``ssim`` defaults), computed on the Y map only.
  Implemented in ``metrics.ssim_per_image`` and reused here.

* **aggregation** -- per-image first, equal weight per image (``metrics.aggregate``).

Provenance
----------
The transform below is **byte-for-byte the same expression** that produced the
published y_shave figures (28.91 / 33.82 dB) in the cross-check kernel
`pixelforge-eval-protocol`; ``tests/test_eval_protocol.py::test_matches_kernel_impl``
pins that equivalence so the two can never drift apart silently.
"""

from __future__ import annotations

import torch

from metrics import aggregate, psnr_per_image, ssim_per_image

# BT.601 studio-swing luma weights (MATLAB rgb2ycbcr). Sum = 219.0 exactly.
_Y_R, _Y_G, _Y_B = 65.481, 128.553, 24.966
_Y_OFFSET = 16.0


def rgb_to_y(t: torch.Tensor) -> torch.Tensor:
    """RGB ``(B,3,H,W)`` in [0,1] -> Y ``(B,1,H,W)`` in [16/255, 235/255].

    Identical coefficients and operation order as the kernel implementation that
    produced the published y_shave numbers, so results are bit-comparable.
    """
    if t.dim() != 4 or t.shape[1] != 3:
        raise ValueError(f"expected (B,3,H,W) RGB, got {tuple(t.shape)}")
    r, g, b = t[:, 0:1], t[:, 1:2], t[:, 2:3]
    return _Y_R / 255 * r + _Y_G / 255 * g + _Y_B / 255 * b + _Y_OFFSET / 255


def shave(t: torch.Tensor, border: int) -> torch.Tensor:
    """Crop ``border`` pixels from every spatial side (no-op for border=0)."""
    if border < 0:
        raise ValueError(f"border must be >= 0, got {border}")
    if border == 0:
        return t
    h, w = t.shape[-2], t.shape[-1]
    if 2 * border >= min(h, w):
        raise ValueError(f"border={border} too large for spatial size {h}x{w}")
    return t[..., border:h - border, border:w - border]


def psnr_y(pred: torch.Tensor, target: torch.Tensor, border: int = 0) -> torch.Tensor:
    """Per-image PSNR on Y after shave, shape ``(B,)``.

    Raises if ``border`` does not fit the image -- silently shrinking it would
    give different images different borders and make the average incomparable.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    y_p = shave(rgb_to_y(pred), border)
    y_t = shave(rgb_to_y(target), border)
    return psnr_per_image(y_p, y_t)


def ssim_y(pred: torch.Tensor, target: torch.Tensor, border: int = 0) -> torch.Tensor:
    """Per-image SSIM on Y after shave, shape ``(B,)``."""
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    y_p = shave(rgb_to_y(pred), border)
    y_t = shave(rgb_to_y(target), border)
    return ssim_per_image(y_p, y_t)


def evaluate_y(pred: torch.Tensor, target: torch.Tensor, border: int = 0) -> dict:
    """Literature-protocol PSNR/SSIM for a batch, aggregated per image."""
    p = psnr_y(pred, target, border=border)
    s = ssim_y(pred, target, border=border)
    return {"psnr": aggregate([float(v) for v in p]),
            "ssim": aggregate([float(v) for v in s]),
            "psnr_per_image": [float(v) for v in p],
            "ssim_per_image": [float(v) for v in s],
            "protocol": f"y_shave(border={border})" if border else "y(border=0)"}
