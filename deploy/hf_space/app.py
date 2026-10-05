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
import threading

import gradio as gr
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter

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
# TorchScript model loading (mirror of serve/model_loader.py; keep in sync)
# --------------------------------------------------------------------------- #
# Gradio serves requests from a thread pool, so the check-then-act caches below
# are hit concurrently. The lock guarantees a cold start loads each weight once
# instead of once per in-flight request.
_MODEL_LOCK = threading.Lock()
_sr_cache: dict = {}
# Sentinel: distinct from a *successful-but-None* load. Using bare None meant a
# corrupt lowlight.pt was re-loaded (and re-warned) on every single request,
# because `_lowlight_model is None` stayed true after the failed load.
_LOWLIGHT_UNSET = object()
_lowlight_model = _LOWLIGHT_UNSET


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
        with _MODEL_LOCK:
            if key not in _sr_cache:
                matches = list(MODELS_DIR.glob(f"sr_*_scale{scale}.pt"))
                # Cache the miss too, so a missing weight doesn't re-glob the
                # filesystem on every request.
                _sr_cache[key] = _load(str(matches[0])) if matches else None
    return _sr_cache.get(key)


def get_lowlight_model():
    global _lowlight_model
    if _lowlight_model is _LOWLIGHT_UNSET:
        with _MODEL_LOCK:
            if _lowlight_model is _LOWLIGHT_UNSET:
                p = MODELS_DIR / "lowlight.pt"
                # Either branch replaces the sentinel (including a failed load
                # returning None), so a broken weight is not reloaded per request.
                _lowlight_model = _load(str(p)) if p.exists() else None
    return _lowlight_model


@torch.no_grad()
def predict_sr(img: Image.Image, scale: int):
    """Run SR. Returns ``(lr_input, model_output)`` or ``None``.

    ``lr_input`` is what the model ACTUALLY sees (``img`` downscaled by
    ``scale``). The UI shows a 3-panel view so a full-resolution
    original is never compared against a reconstruction that only had
    ``1/scale^2`` of the pixels to work with.
    """
    model = get_sr_model(scale)
    if model is None:
        return None
    lr = img.resize((max(1, img.width // scale), max(1, img.height // scale)), Image.BICUBIC)
    x = TF.to_tensor(lr).unsqueeze(0).to(DEVICE)
    out = model(x).clamp(0, 1)
    out = TF.to_pil_image(out.squeeze(0).cpu())
    # Return the model's TRUE output (lr*scale) plus the true LR input.
    return lr, out


def _pad_to_multiple(img: Image.Image, m: int = 32) -> tuple[Image.Image, tuple[int, int]]:
    """Pad right/bottom so both sides are multiples of ``m``, by REFLECTION.

    The U-Net needs sides divisible by 32. A flat fill (we used 0/black) is an
    artificial edge the model never saw in LOL training data, so it "corrects"
    it and leaves a visible seam along the padded sides (measured +0.106 blue
    bias on the right-edge band vs +0.023 with reflection).
    """
    w, h = img.size
    pw, ph = (-w) % m, (-h) % m
    if pw or ph:
        arr = np.pad(np.asarray(img.convert("RGB")), ((0, ph), (0, pw), (0, 0)),
                     mode="reflect")
        img = Image.fromarray(arr, "RGB")
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


# --- exposure gate + output guard ------------------------------------------- #
# The low-light U-Net is trained on LOL-v1 = real night PHOTOS. On synthetic art
# or an already-bright image it still applies its learned illumination
# correction, but with no true underexposure to recover it crushes the shadows
# and the picture gets DARKER (measured: median luminance 0.26 -> 0.06).
#
#   1. INPUT gate  -- only run the model on something plausibly underexposed.
#      Cheap but NOT sufficient: dark artwork also looks dark.
#   2. OUTPUT guard -- after running, verify the model actually HELPED. An
#      "enhancement" that darkens its input is wrong by definition, so we
#      discard it. Self-validating: a real night photo brightens by +0.47 mean
#      luminance, while every failing case darkens (-0.02 .. -0.08).
LOWLIGHT_DARK_P10 = 0.22
LOWLIGHT_DARK_MEAN = 0.36
LOWLIGHT_MIN_GAIN = 0.005


def _mean_luminance(img: Image.Image) -> float:
    arr = np.asarray(img.convert("RGB"), dtype="float32") / 255.0
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return float(lum.mean())


def looks_underexposed(img: Image.Image) -> tuple[bool, dict]:
    """Heuristically decide whether ``img`` is a genuinely low-light photo."""
    arr = np.asarray(img.convert("RGB"), dtype="float32") / 255.0
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    mean_lum = float(lum.mean())
    p10 = float(np.percentile(lum, 10))
    p50 = float(np.percentile(lum, 50))
    stats = {"mean": mean_lum, "p10": p10, "p50": p50}
    is_dark = (p10 <= LOWLIGHT_DARK_P10) and (mean_lum <= LOWLIGHT_DARK_MEAN)
    return is_dark, stats


# --------------------------------------------------------------------------- #
# Gradio UI
# --------------------------------------------------------------------------- #
def _engine_status() -> str:
    sr2 = "ML" if get_sr_model(2) is not None else "classical baseline (no weight)"
    sr4 = "ML" if get_sr_model(4) is not None else "classical baseline"
    low = "ML" if get_lowlight_model() is not None else "classical baseline"
    return (
        f"**Engine in use**: SR ×2: `{sr2}` · SR ×4: `{sr4}` · Low-light: `{low}`  \n"
        f"Trained models are loaded from `models/` when present; "
        f"otherwise the classical baselines keep the demo fully usable."
    )


def process(image, task, scale):
    """Returns (original, lr_or_baseline, after).

    For SR the middle panel is the model's real low-res input (nearest-upscaled
    for display). For low-light the middle panel is the classical adaptive-gamma
    baseline, so the model can be compared against a correct-but-simple
    reference (see the exposure gate below).
    """
    if image is None:
        raise gr.Error("Please upload an image first.")
    image = image.convert("RGB")
    scale = int(scale)
    if task == "sr":
        pair = predict_sr(image, scale)
        if pair is not None:
            lr, out = pair
            mid = lr.resize(out.size, Image.NEAREST)
        else:
            out = sr_classical(image, scale)
            mid = image
    else:
        # Compute BOTH so panel ② (classical) is always available as a fair
        # reference, and gate the model on whether the input is really dark.
        classical_out = lowlight_classical(image)
        is_dark, _stats = looks_underexposed(image)
        ml_out = predict_lowlight(image) if is_dark else None
        # OUTPUT guard: discard the model's output if it DARKENED the image
        # (an "enhancement" that darkens is wrong by definition). Catches dark
        # synthetic art, which passes the input gate because it genuinely is dark.
        if ml_out is not None and (_mean_luminance(ml_out) - _mean_luminance(image)) < LOWLIGHT_MIN_GAIN:
            ml_out = None
        if ml_out is not None:
            out = ml_out
            mid = classical_out
        else:
            out = classical_out
            mid = image
    return image, mid, out


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
            before = gr.Image(type="pil", label="① 原图 / Original")
            mid = gr.Image(type="pil", label="② 模型输入（SR）或 classical 基线（低光）")
            after = gr.Image(type="pil", label="③ 输出 / Enhanced")
    btn.click(process, inputs=[inp, task, scale], outputs=[before, mid, after])
    gr.Markdown(
        "**SR 怎么看**：超分把「低分辨率」映射成「高分辨率」，中间面板才是模型的真正输入"
        "（由原图降采样得到），③ 是重建结果，应比 ② 清晰很多。① 本来就高清，"
        "超分不会、也不该声称能超过它的真实细节。想看公平对比请上传**低分辨率**图。\n\n"
        "**Low-light 怎么看**：中间面板是 **classical 自适应伽马基线**（正常提亮），"
        "③ 是自训 U-Net。若 ③ 比 ① 更暗，说明这张图**不是低光照片**、模型未被启用"
        "该 U-Net 在 LOL-v1（真实夜间**照片**）上训练，只对确实欠曝的照片有帮助。"
        "常见不适用情况：图像曝光已足够（不够暗），或它是**截图/合成图**而非照片。"
        "此时模型仍会执行照度校正，但因没有真正的欠曝可恢复，会压死暗部、使画面更暗"
        "（实测亮度中位数 0.26 → 0.06）。**想看模型真实效果，请上传确实很暗的照片。**\n\n"
        "Trained on AutoDL RTX 3080 Ti · SR ×4 (perceptual) · Low-light "
        "(PSNR 18.18, full-image validation protocol)."
    )


if __name__ == "__main__":
    demo.launch()
