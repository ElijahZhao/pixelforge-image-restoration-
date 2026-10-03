"""Hugging Face Spaces entry point for the PixelForge image-restoration demo.

Self-contained: bundles the classical baselines + TorchScript model loader +
Gradio UI in a single file so the Space has no local-package import issues.

Deploy
------
1. Create a new Space (SDK: Gradio, hardware: CPU basic is enough).
2. Upload this file as ``app.py``, plus ``requirements.txt`` and the
   ``models/`` folder (``sr_generator_scale4.pt`` + ``lowlight.pt``).
3. The Space builds and serves a public link automatically.

The trained models switch in automatically when present; otherwise the
classical baselines keep the demo fully usable.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageOps

MODELS_DIR = Path(__file__).resolve().parent / "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Classical baselines (mirror of serve/classical.py)
# --------------------------------------------------------------------------- #
def _to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")


def sr_classical(img: Image.Image, scale: int) -> Image.Image:
    """Bicubic upscale + unsharp masking."""
    up = img.resize((img.width * scale, img.height * scale), Image.BICUBIC)
    arr = _to_array(up)
    blurred = _to_array(up.filter(ImageFilter.GaussianBlur(radius=1.5)))
    return _to_image(arr + 0.6 * (arr - blurred))


def lowlight_classical(img: Image.Image) -> Image.Image:
    """Adaptive gamma correction."""
    arr = _to_array(img)
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    mean_lum = float(lum.mean())
    gamma = float(np.clip(np.log(0.5) / np.log(max(mean_lum, 1e-3)), 0.3, 1.0))
    out = arr ** gamma
    return _to_image((out - 0.5) * 1.1 + 0.5)


# --------------------------------------------------------------------------- #
# TorchScript model loading (mirror of serve/model_loader.py — keep in sync)
# --------------------------------------------------------------------------- #
_sr_cache: dict = {}
_lowlight_model = None


def _load(path: str):
    try:
        return torch.jit.load(path, map_location=DEVICE).eval()
    except Exception as exc:  # noqa: BLE001 - corrupt/incompatible weights must not 500
        import warnings

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
def predict_sr(img: Image.Image, scale: int) -> Image.Image | None:
    model = get_sr_model(scale)
    if model is None:
        return None
    lr = img.resize((max(1, img.width // scale), max(1, img.height // scale)), Image.BICUBIC)
    x = TF.to_tensor(lr).unsqueeze(0).to(DEVICE)
    out = model(x).clamp(0, 1)
    out = TF.to_pil_image(out.squeeze(0).cpu())
    # Return the model's TRUE output (lr*scale). Do not re-upscale to img*scale.
    return out


def _pad_to_multiple(img: Image.Image, m: int = 32) -> tuple[Image.Image, tuple[int, int]]:
    """Pad right/bottom so both sides are multiples of ``m`` (U-Net needs this)."""
    w, h = img.size
    pw, ph = (-w) % m, (-h) % m
    if pw or ph:
        img = ImageOps.expand(img, border=(0, 0, pw, ph), fill=0)
    return img, (w, h)


@torch.no_grad()
def predict_lowlight(img: Image.Image) -> Image.Image | None:
    model = get_lowlight_model()
    if model is None:
        return None
    # U-Net downsamples by 2 several times, so sides must be a multiple of 32.
    padded, (w, h) = _pad_to_multiple(img, 32)
    x = TF.to_tensor(padded).unsqueeze(0).to(DEVICE)
    out = model(x).clamp(0, 1)
    out = TF.to_pil_image(out.squeeze(0).cpu())
    return out.crop((0, 0, w, h))


# --------------------------------------------------------------------------- #
# Gradio UI
# --------------------------------------------------------------------------- #
def _engine_status() -> str:
    sr2 = "ML" if get_sr_model(2) is not None else "classical baseline (no weight)"
    sr4 = "ML" if get_sr_model(4) is not None else "classical baseline"
    low = "ML" if get_lowlight_model() is not None else "classical baseline"
    return (
        f"**Engine in use** — SR ×2: `{sr2}` · SR ×4: `{sr4}` · Low-light: `{low}`  \n"
        f"Trained models are loaded from `models/` when present; "
        f"otherwise the classical baselines keep the demo fully usable."
    )


def process(image, task, scale):
    if image is None:
        raise gr.Error("Please upload an image first.")
    image = image.convert("RGB")
    scale = int(scale)
    if task == "sr":
        out = predict_sr(image, scale) or sr_classical(image, scale)
    else:
        out = predict_lowlight(image) or lowlight_classical(image)
    return image, out


with gr.Blocks(title="PixelForge · Image Restoration") as demo:
    gr.Markdown("# PixelForge · 图像修复 Demo\nSuper-Resolution (超分) & Low-Light Enhancement (低光增强)")
    status = gr.Markdown(_engine_status())
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Input image")
            task = gr.Radio(["sr", "lowlight"], value="sr", label="Task")
            scale = gr.Radio(["2", "4"], value="4", label="SR scale (×4 使用自训模型)")
            btn = gr.Button("Enhance", variant="primary")
        with gr.Column():
            before = gr.Image(type="pil", label="Before")
            after = gr.Image(type="pil", label="After (enhanced)")
    btn.click(process, inputs=[inp, task, scale], outputs=[before, after])
    gr.Markdown(
        "Trained on AutoDL RTX 3080 Ti · SR ×4 (perceptual) · Low-light "
        "(PSNR 18.18, full-image validation protocol)."
    )


if __name__ == "__main__":
    demo.launch()
