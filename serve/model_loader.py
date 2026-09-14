"""Load exported TorchScript models and run inference.

Export real trained weights with ``train/export.py`` into ``serve/models/``:
    serve/models/sr_srcnn_scale2.pt
    serve/models/sr_generator_scale4.pt
    serve/models/lowlight.pt

When a weight file is missing, the corresponding task silently falls back to
``classical.py`` in ``app.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as TF

MODELS_DIR = Path(__file__).resolve().parent / "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_sr_cache: dict = {}
_lowlight_model = None


def _load(path: str):
    return torch.jit.load(path, map_location=DEVICE).eval()


def get_sr_model(scale: int):
    key = f"sr_{scale}"
    if key not in _sr_cache:
        p = MODELS_DIR / f"sr_*_scale{scale}.pt"
        matches = list(MODELS_DIR.glob(f"sr_*_scale{scale}.pt"))
        if matches:
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
    model = get_sr_model(scale)
    if model is None:
        return None
    # The model expects an LR input; we downscale then let the model upscale.
    lr = img.resize((max(1, img.width // scale), max(1, img.height // scale)),
                    Image.BICUBIC)
    x = TF.to_tensor(lr).unsqueeze(0).to(DEVICE)
    out = model(x).clamp(0, 1)
    out = TF.to_pil_image(out.squeeze(0).cpu())
    # Match the original aspect by resizing to scale * original size.
    return out.resize((img.width * scale, img.height * scale), Image.BICUBIC)


@torch.no_grad()
def predict_lowlight(img: Image.Image) -> Image.Image:
    model = get_lowlight_model()
    if model is None:
        return None
    x = TF.to_tensor(img).unsqueeze(0).to(DEVICE)
    out = model(x).clamp(0, 1)
    return TF.to_pil_image(out.squeeze(0).cpu())
