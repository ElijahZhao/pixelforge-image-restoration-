"""Classical (non-learned) baselines used when no trained weights are present.

These let the website run end-to-end out of the box. They are *not* the ML
models — once you train and export real weights into ``serve/models/``, the
service switches to the learned models automatically.

  - Super-Resolution : bicubic upscale + unsharp masking (a classic sharpening
    trick that visibly improves edge crispness over plain upscaling).
  - Low-Light        : adaptive gamma correction (brightens dark images while
    leaving already-bright images essentially unchanged) — a robustness-friendly
    low-light baseline.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def _to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _to_image(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")


def sr_classical(img: Image.Image, scale: int) -> Image.Image:
    """Bicubic upscale followed by unsharp masking."""
    up = img.resize((img.width * scale, img.height * scale), Image.BICUBIC)
    arr = _to_array(up)
    # Unsharp mask: sharpen = img + amount * (img - blurred)
    blurred = up.filter(ImageFilter.GaussianBlur(radius=1.5))
    barr = _to_array(blurred)
    sharp = arr + 0.6 * (arr - barr)
    return _to_image(sharp)


def lowlight_classical(img: Image.Image) -> Image.Image:
    """Adaptive gamma correction for low-light images.

    Darker images get a smaller gamma (brighter); images that are already
    well-lit are left essentially unchanged. A simple, robust, dependency-free
    classical baseline for low-light enhancement.
    """
    arr = _to_array(img)
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    mean_lum = float(lum.mean())
    # Target mid-grey 0.5; gamma < 1 brightens, gamma = 1 leaves alone.
    gamma = float(np.clip(np.log(0.5) / np.log(max(mean_lum, 1e-3)), 0.3, 1.0))
    out = arr ** gamma
    # Mild contrast stretch to recover punch after brightening.
    out = np.clip((out - 0.5) * 1.1 + 0.5, 0.0, 1.0)
    return _to_image(out)


def run_classical(img: Image.Image, task: str, scale: int) -> Image.Image:
    if task == "sr":
        return sr_classical(img, scale)
    if task == "lowlight":
        return lowlight_classical(img)
    raise ValueError(task)
