"""Load exported TorchScript models and run inference.

Export real trained weights with ``train/export.py`` into ``serve/models/``:
    serve/models/sr_srcnn_scale2.pt
    serve/models/sr_generator_scale4.pt
    serve/models/lowlight.pt

When a weight file is missing, the corresponding task silently falls back to
``classical.py`` in ``app.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF

MODELS_DIR = Path(__file__).resolve().parent / "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_sr_cache: dict = {}
_lowlight_model = None


def _load(path: str):
    """Load a TorchScript model, failing soft instead of crashing the service.

    A corrupt or torch-incompatible ``.pt`` must not take down the whole API
    (the deploy/streamlit copy already does this; see R1). On failure we warn
    and return ``None`` so callers fall back to the classical baseline. The
    failure is cached (None) so we don't retry-load on every request.
    """
    try:
        return torch.jit.load(path, map_location=DEVICE).eval()
    except Exception as exc:  # noqa: BLE001 - defensive: never 500 on bad weights
        warnings.warn(
            f"failed to load model from {path}: {exc!r}; "
            f"falling back to classical baseline",
            stacklevel=2,
        )
        return None


def get_sr_model(scale: int):
    key = f"sr_{scale}"
    if key not in _sr_cache:
        matches = list(MODELS_DIR.glob(f"sr_*_scale{scale}.pt"))
        if matches:
            # May cache None on load failure (see _load).
            _sr_cache[key] = _load(str(matches[0]))
    return _sr_cache.get(key)


def get_lowlight_model():
    global _lowlight_model
    if _lowlight_model is None:
        p = MODELS_DIR / "lowlight.pt"
        if p.exists():
            _lowlight_model = _load(str(p))
    return _lowlight_model


@torch.no_grad()
def predict_sr(img: Image.Image, scale: int) -> Image.Image:
    """Super-resolve ``img`` by ``scale``.

    Honesty note (see DIAGNOSIS_ROUND10/16/18): the model's *real* output is
    ``lr_size * scale``. The previous implementation then ran a second PIL
    BICUBIC resize up to ``orig_size * scale``, silently claiming "x4" while the
    model only ever produced a x4 of the downscaled input. We now return the
    model's true output and never re-upscale it.
    """
    model = get_sr_model(scale)
    if model is None:
        return None
    # The model expects an LR input; we downscale then let the model upscale.
    lr = img.resize((max(1, img.width // scale), max(1, img.height // scale)),
                    Image.BICUBIC)
    x = TF.to_tensor(lr).unsqueeze(0).to(DEVICE)
    out = model(x).clamp(0, 1)
    return TF.to_pil_image(out.squeeze(0).cpu())


def _pad_to_multiple(img: Image.Image, m: int = 32) -> tuple[Image.Image, tuple[int, int]]:
    """Pad right/bottom so both sides are multiples of ``m``.

    The low-light U-Net downsamples by 2 several times, so it requires input
    sides divisible by 32; otherwise the decoder concatenation fails with
    "Sizes of tensors must match". We pad, run, then crop back to the original
    size so any user-supplied image works.
    """
    w, h = img.size
    pw, ph = (-w) % m, (-h) % m
    if pw or ph:
        img = ImageOps.expand(img, border=(0, 0, pw, ph), fill=0)
    return img, (w, h)


@torch.no_grad()
def predict_lowlight(img: Image.Image) -> Image.Image:
    model = get_lowlight_model()
    if model is None:
        return None
    padded, (w, h) = _pad_to_multiple(img, 32)
    x = TF.to_tensor(padded).unsqueeze(0).to(DEVICE)
    out = model(x).clamp(0, 1)
    out = TF.to_pil_image(out.squeeze(0).cpu())
    return out.crop((0, 0, w, h))


# --- exposure gate (see DIAGNOSIS_ROUND20 & the Streamlit copy) ------------- #
# The low-light U-Net is trained on LOL-v1 = real night PHOTOS. On a bright or
# synthetic image (e.g. a game screenshot) it still applies its learned
# illumination correction, but with no true underexposure to recover it crushes
# the shadows and the picture gets DARKER (measured: median luminance
# 0.26 -> 0.06). We require BOTH a dark 10th percentile and a dark mean so a
# normal photo with a few shadows (bright subject on a dark background) is not
# mistaken for an underexposed shot.
LOWLIGHT_DARK_P10 = 0.22
LOWLIGHT_DARK_MEAN = 0.36


def looks_underexposed(img: Image.Image) -> tuple[bool, dict]:
    """Heuristically decide whether ``img`` is a genuinely low-light photo."""
    import numpy as np

    arr = np.asarray(img.convert("RGB"), dtype="float32") / 255.0
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    mean_lum = float(lum.mean())
    p10 = float(np.percentile(lum, 10))
    p50 = float(np.percentile(lum, 50))
    stats = {"mean": mean_lum, "p10": p10, "p50": p50}
    is_dark = (p10 <= LOWLIGHT_DARK_P10) and (mean_lum <= LOWLIGHT_DARK_MEAN)
    return is_dark, stats
