"""Streamlit Community Cloud entry point for the PixelForge image-restoration demo.

Self-contained: bundles the classical baselines + TorchScript model loader +
Streamlit UI in a single file so the deployed app has no local-package imports.

Deploy (Streamlit Community Cloud, free)
----------------------------------------
1. Push this repo to GitHub (already done).
2. Go to https://share.streamlit.io → "New app" → pick this repo/branch.
3. Set the main file path to ``deploy/streamlit/streamlit_app.py``.
4. Deploy — you get a public ``*.streamlit.app`` link.

The trained models switch in automatically when present in ``models/``;
otherwise the classical baselines keep the demo fully usable.

UI
--
Bilingual (English default, one-click switch to Chinese) and a dark/light
theme toggle, both driven by ``st.session_state``. All UI strings live in
``TEXTS``; all colours live in ``_CSS_DARK`` / ``_CSS_LIGHT``. Switching either
option just re-renders — no inference logic depends on language or theme.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageFilter, ImageOps

# NOTE: torch / torchvision are imported LAZILY (inside the functions below) to
# minimise the memory footprint on Streamlit Community Cloud's free tier. Importing
# torch eagerly at module load can push a ~1 GB instance over its limit and get the
# process OOM-killed (which shows up as a bare "Oh no." page with no traceback).

MODELS_DIR = Path(__file__).resolve().parent / "models"

# --------------------------------------------------------------------------- #
# Bilingual UI strings. English is the default; ``LANG`` selects the active set.
# Keys are shared across both languages so switching never leaves a blank.
# --------------------------------------------------------------------------- #
TEXTS = {
    "en": {
        "page_title": "PixelForge · Image Restoration",
        "sidebar_header": "👾 Controls",
        "lang_label": "Language / 语言",
        "theme_dark": "🌙 Dark",
        "theme_light": "☀️ Light",
        "theme_label": "Theme",
        "task_label": "Task",
        "task_sr": "Super-Resolution (SR)",
        "task_lowlight": "Low-Light Enhancement",
        "scale_label": "SR scale",
        "scale_help": "×4 uses the self-trained SRResNet + perceptual-loss model",
        "trained_caption": "Trained on AutoDL RTX 3080 Ti · Low-light full-image validation PSNR 18.18",
        "upload_label": "Upload an image",
        "spinner": "Running inference…",
        "tag_ml": "ML model",
        "tag_classical": "classical baseline",
        "warn_sr_no_weight": "SR ×{scale}: no self-trained weight found — using the **classical bicubic baseline**.",
        "ok_sr": "SR ×{scale}: produced by the **self-trained model** (true resolution = downscaled input ×{scale}).",
        "warn_ll_no_weight": "Low-light: no self-trained weight — using the **classical adaptive-gamma baseline**.",
        "ok_ll": "Low-light: produced by the **self-trained U-Net**.",
        "note_ll_notdark": (
            "**This image is not a low-light photo**, so the self-trained "
            "low-light U-Net was skipped. That model is trained on LOL-v1 "
            "(real night *photographs*); it only helps a genuinely underexposed "
            "photo. Two common reasons it does not apply here: the image is "
            "already well-exposed (it is not dark enough to be 'low light'), "
            "or it is a **screenshot / synthetic image** rather than a "
            "photograph. In either case the model would still run its "
            "illumination correction with no real underexposure to recover, so "
            "it **crushes the shadows and makes the picture darker** (measured "
            "on a game screenshot: median luminance 0.26 → 0.06). You are "
            "seeing the **classical adaptive-gamma baseline**, which brightens. "
            "Upload a genuinely dark **photograph** to exercise the model."
        ),
        "info_ll_3panel": (
            "**How to read these three panels**: ① your input, ② the classical "
            "adaptive-gamma baseline, ③ the self-trained U-Net. Where ③ is "
            "darker than ①, the model is fighting an input it was not trained "
            "for (see the note above) — ② is the safer choice for that image."
        ),
        "cap_original": "① Original (your upload)",
        "cap_lr": "② Model input (LR {w}×{h}, upscaled for display)",
        "cap_sr_out": "③ PixelForge SR output (true ×{scale})",
        "cap_before": "Before",
        "cap_after": "After (enhanced)",
        "sec_upload": "Upload",
        "sec_result": "Result",
        "info_3panel": (
            "**How to read these three panels**: super-resolution maps a "
            "*low-resolution* image to a *high-resolution* one, so the middle "
            "panel is the model's real input (your original downscaled ×{scale}). "
            "The model only ever saw those pixels; panel ③ is its reconstruction "
            "and should look much sharper than ②. Panel ① is the reference — it "
            "is already high-res, and **SR neither can nor claims to beat its "
            "true detail**. For a fair comparison upload a **low-resolution** "
            "image (or just compare ② → ③)."
        ),
        "download": "⬇️ Download result PNG",
        "empty_info": (
            "👾 Upload an image to begin. Low-light works best on dark photos; "
            "super-resolution is meant for low-resolution inputs."
        ),
        "footer": "PIXELFORGE · press start to restore your images",
    },
    "zh": {
        "page_title": "PixelForge · 图像修复",
        "sidebar_header": "👾 参数",
        "lang_label": "语言 / Language",
        "theme_dark": "🌙 黑夜",
        "theme_light": "☀️ 白天",
        "theme_label": "主题",
        "task_label": "任务",
        "task_sr": "超分辨率 (SR)",
        "task_lowlight": "低光增强 (Low-Light)",
        "scale_label": "SR 放大倍数",
        "scale_help": "×4 使用自训 SRResNet + 感知损失模型",
        "trained_caption": "模型在 AutoDL RTX 3080 Ti 上训练 · Low-light 全图验证 PSNR 18.18",
        "upload_label": "上传图片",
        "spinner": "推理中…",
        "tag_ml": "ML 自训模型",
        "tag_classical": "classical 基线",
        "warn_sr_no_weight": "SR ×{scale}：无对应自训权重，当前使用 **classical bicubic 基线**。",
        "ok_sr": "SR ×{scale}：由 **自训模型** 输出（真实分辨率 = 下采样输入 ×{scale}）。",
        "warn_ll_no_weight": "低光：无自训权重，当前使用 **classical 自适应伽马基线**。",
        "ok_ll": "低光：由 **自训 U-Net** 输出。",
        "note_ll_notdark": (
            "**这张图不是低光照片**，因此已跳过自训低光 U-Net。该模型是在 "
            "LOL-v1（真实夜间**照片**）上训练的，只对「确实欠曝的照片」有帮助。"
            "常见的不适用情况有两种：图像本身曝光已足够（不够暗，算不上低光），"
            "或者它是**截图 / 合成图**而非真实照片。这两种情况下，模型仍会执行"
            "学到的照度校正，但由于没有真正的欠曝可恢复，结果会把**暗部压死、"
            "使画面更暗**（实测游戏截图：亮度中位数 0.26 → 0.06）。"
            "你现在看到的是 **classical 自适应伽马基线**，它是正常提亮的。"
            "想真正测试该模型，请上传一张确实很暗的**照片**。"
        ),
        "info_ll_3panel": (
            "**怎么看这三张图**：① 你的输入，② classical 自适应伽马基线，"
            "③ 自训 U-Net。若 ③ 比 ① 还暗，说明模型正在处理一张它没被训练过的输入"
            "（见上方说明）——对该图而言 ② 是更稳妥的选择。"
        ),
        "cap_original": "① 原图 (your upload)",
        "cap_lr": "② 模型实际输入 (低清 {w}×{h}，放大显示)",
        "cap_sr_out": "③ PixelForge 超分输出 (真实 ×{scale})",
        "cap_before": "Before",
        "cap_after": "After (enhanced)",
        "sec_upload": "上传",
        "sec_result": "结果",
        "info_3panel": (
            "**怎么看这三张图**：超分把「低分辨率」映射成「高分辨率」，"
            "所以中间那张才是模型的真正输入（由你的原图降采样 ×{scale} 得到）。"
            "模型只见过中间这张的像素，③ 是它重建出的结果——③ 应比 ② 清晰很多。"
            "① 是参考原图：它本来就高清，**超分不会、也不该声称能超过它的真实细节**。"
            "想看公平对比，请上传**低分辨率**图片（或直接看 ②→③）。"
        ),
        "download": "⬇️ 下载结果 PNG",
        "empty_info": (
            "👾 请上传一张图片开始体验。低光任务建议用较暗的照片；"
            "超分建议用低分辨率图。"
        ),
        "footer": "PIXELFORGE · press start to restore your images",
    },
}


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


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
# TorchScript model loading (mirror of serve/model_loader.py, with size fix)
#
# IMPORTANT: on Streamlit Community Cloud the free tier ships ~1 GB RAM. Importing
# torch + loading TorchScript weights is memory-hungry, so we:
#   * only PROBE file existence when rendering (never torch.jit.load at render time)
#   * load lazily on the first actual inference, and
#   * fail soft -- if loading raises (e.g. OOM), fall back to the classical baseline
#     instead of crashing the whole app.
# --------------------------------------------------------------------------- #
def _sr_weight_path(scale: int):
    matches = sorted(MODELS_DIR.glob(f"sr_*_scale{scale}.pt"))
    return matches[0] if matches else None


def _lowlight_weight_path():
    p = MODELS_DIR / "lowlight.pt"
    return p if p.exists() else None


@st.cache_resource(show_spinner=False)
def _load_sr(path_str: str):
    try:
        import torch

        return torch.jit.load(path_str, map_location=_device()).eval()
    except Exception:  # noqa: BLE001 - never let model loading take down the app
        return None


@st.cache_resource(show_spinner=False)
def _load_lowlight(path_str: str):
    try:
        import torch

        return torch.jit.load(path_str, map_location=_device()).eval()
    except Exception:  # noqa: BLE001
        return None


def _pad_to_multiple(img: Image.Image, m: int = 32) -> tuple[Image.Image, tuple[int, int]]:
    """Pad right/bottom so both sides are multiples of ``m``.

    The low-light U-Net downsamples by 2 several times, so input sides must be
    divisible by 32; otherwise the decoder concatenation fails. We pad, run,
    then crop back so any user-supplied image size works.
    """
    w, h = img.size
    pw, ph = (-w) % m, (-h) % m
    if pw or ph:
        img = ImageOps.expand(img, border=(0, 0, pw, ph), fill=0)
    return img, (w, h)


def predict_sr(img: Image.Image, scale: int):
    """Run SR. Returns ``(lr_input, model_output)`` or ``None``.

    ``lr_input`` is the image the model ACTUALLY sees (``img`` downscaled by
    ``scale``), returned so the UI can show the honest 3-panel view:
    original / low-res input / super-resolved output. A x4 model can only ever
    reconstruct from ``1/16`` of the pixels, so comparing its output against a
    full-resolution original is misleading — see the 3-panel caption.
    """
    path = _sr_weight_path(scale)
    if path is None:
        return None
    model = _load_sr(str(path))
    if model is None:
        return None
    import torch
    import torchvision.transforms.functional as TF

    with torch.no_grad():
        lr = img.resize((max(1, img.width // scale), max(1, img.height // scale)), Image.BICUBIC)
        x = TF.to_tensor(lr).unsqueeze(0).to(_device())
        out = model(x).clamp(0, 1)
        out = TF.to_pil_image(out.squeeze(0).cpu())
    # Return the model's TRUE output (lr*scale), plus the true LR input.
    return lr, out


def predict_lowlight(img: Image.Image) -> Image.Image | None:
    path = _lowlight_weight_path()
    if path is None:
        return None
    model = _load_lowlight(str(path))
    if model is None:
        return None
    import torch
    import torchvision.transforms.functional as TF

    padded, (w, h) = _pad_to_multiple(img, 32)
    with torch.no_grad():
        x = TF.to_tensor(padded).unsqueeze(0).to(_device())
        out = model(x).clamp(0, 1)
        out = TF.to_pil_image(out.squeeze(0).cpu())
    return out.crop((0, 0, w, h))


# --- is-this-actually-a-low-light-photo? ------------------------------------ #
# The low-light U-Net is trained on LOL-v1: real night PHOTOS, where "low light"
# means photon-starved sensor data with a specific noise/gamma signature. Fed a
# synthetic bright image (e.g. a game screenshot), it still applies its learned
# illumination correction, but with no real underexposure to recover it just
# crushes the shadows and amplifies compression artefacts -- i.e. the picture
# gets DARKER, not brighter (verified: median luminance 0.26 -> 0.06 on a real
# screenshot). So we gate the model on a simple exposure check and fall back to
# the (correctly brightening) classical gamma baseline when the image is not
# actually underexposed.
LOWLIGHT_DARK_P10 = 0.22   # a real night photo has a very dark 10th percentile
LOWLIGHT_DARK_MEAN = 0.36  # ...and a low overall mean luminance


def looks_underexposed(img: Image.Image) -> tuple[bool, dict]:
    """Heuristically decide whether ``img`` is a genuinely low-light photo.

    Returns ``(is_dark, stats)``. We require BOTH a dark lower decile *and* a
    dark mean, so a normal photo that merely contains a few shadows (a bright
    subject on a dark background) is not mistaken for an underexposed shot.
    """
    arr = _to_array(img)
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    mean_lum = float(lum.mean())
    p10 = float(np.percentile(lum, 10))
    p50 = float(np.percentile(lum, 50))
    stats = {"mean": mean_lum, "p10": p10, "p50": p50}
    is_dark = (p10 <= LOWLIGHT_DARK_P10) and (mean_lum <= LOWLIGHT_DARK_MEAN)
    return is_dark, stats


# --------------------------------------------------------------------------- #
# Streamlit UI —— 复古像素 / 游戏风（双语 + 暗/亮双主题）
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="PixelForge · Image Restoration",
                   page_icon="👾", layout="centered")

# Two colour schemes share identical selectors; only the :root variables and a
# few hard-coded glows differ. `_css()` returns the active sheet.
_CSS_COMMON = """
<style>
html, body, [class*="css"] {
  font-family: var(--pf-font);
  letter-spacing: 0.2px;
}
.stApp { background: var(--pf-app-bg); }
/* Hide ONLY leaf "chrome" elements, never the containers that host the
   sidebar collapse/expand control. Streamlit moves that control around
   between versions — in new versions it lives inside [data-testid="stToolbar"]
   / stDecoration — so hiding those whole containers removed the button
   entirely (the bug the user hit: no way to open the sidebar at all).
   We therefore target the specific decorative children instead.
   The deploy-button container is `stAppDeployButton` in current releases
   (verified against the live DOM, not guessed). */
#MainMenu { visibility: hidden; }
[data-testid="stAppDeployButton"],
[data-testid="stToolbar"] [data-testid="stMainMenu"],
[data-testid="stStatusWidget"],
[data-testid="stDecoration"] { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
/* Belt-and-braces: whatever wrapper Streamlit uses this release, keep every
   known sidebar toggle visible and clickable. Streamlit's test-id for the
   expand control changed across versions: newer releases use
   `stExpandSidebarButton` (verified against the live DOM), older ones
   `stSidebarCollapsedControl`. Cover both, plus the in-sidebar collapse one. */
[data-testid="stExpandSidebarButton"],
[data-testid="stExpandSidebarButton"] *,
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapsedControl"] *,
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] * { visibility: visible !important; opacity: 1 !important; }

.pf-hero {
  text-align: center;
  padding: 34px 16px 26px;
  margin-bottom: 22px;
  border: 3px solid var(--pf-border);
  border-radius: 6px;
  background: var(--pf-hero-bg);
  box-shadow: var(--pf-hero-shadow);
  position: relative;
  overflow: hidden;
}
/* Pixel-corner accents on the hero frame. */
.pf-hero::before, .pf-hero::after {
  content: "";
  position: absolute;
  width: 14px; height: 14px;
  border: 3px solid var(--pf-cyan);
}
.pf-hero::before { top: 6px; left: 6px; border-right: 0; border-bottom: 0; }
.pf-hero::after { bottom: 6px; right: 6px; border-left: 0; border-top: 0; }
.pf-title {
  font-family: var(--pf-font);
  font-size: 34px;
  font-weight: 700;
  line-height: 1.5;
  color: var(--pf-title-color);
  text-shadow: var(--pf-title-shadow);
  margin: 0 0 14px;
  letter-spacing: 4px;
  word-spacing: 4px;
}
.pf-sub {
  font-size: 13px;
  color: var(--pf-accent-text);
  margin: 0 auto;
  letter-spacing: 2px;
}

/* Status pills row under the hero. */
.pf-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
  margin-top: 18px;
}
.pf-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.5px;
  border-radius: 999px;
  border: 1px solid var(--pf-pill-border);
  background: var(--pf-pill-bg);
  color: var(--pf-pill-fg);
}
.pf-pill.on { border-color: var(--pf-pill-on-border); color: var(--pf-pill-on-fg); }
.pf-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--pf-pill-dot);
  box-shadow: 0 0 6px var(--pf-pill-dot);
}

/* Section cards: give the scattered widgets a visible frame. */
.pf-section-title {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 4px 0 10px;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 2px;
  color: var(--pf-accent-text);
  text-transform: uppercase;
}
.pf-section-title::after {
  content: "";
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--pf-border), transparent);
}

.stButton > button, .stDownloadButton > button {
  font-family: "Courier New", monospace;
  font-weight: 700;
  letter-spacing: 1px;
  border: 2px solid var(--pf-cyan);
  border-radius: 4px;
  background: var(--pf-btn-bg);
  color: var(--pf-btn-fg);
  transition: all 0.12s ease;
  box-shadow: 3px 3px 0 var(--pf-btn-shadow);
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: var(--pf-cyan);
  color: var(--pf-badge-fg);
  transform: translate(-1px, -1px);
  box-shadow: 4px 4px 0 var(--pf-btn-shadow-hover);
}

[data-testid="stSidebar"] {
  background: var(--pf-sidebar-bg);
  border-right: 2px solid var(--pf-sidebar-border);
}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
  padding-top: 1.2rem;
}
[data-testid="stFileUploader"] {
  border: 2px dashed var(--pf-border);
  border-radius: 6px;
  padding: 14px;
  background: var(--pf-card-bg);
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--pf-cyan);
  box-shadow: 0 0 0 3px var(--pf-focus-ring);
}
[data-testid="stImage"] img {
  border: 3px solid var(--pf-img-border);
  border-radius: 4px;
  box-shadow: 4px 4px 0 var(--pf-img-shadow);
}
/* Numbered corner tab on each result panel. */
.pf-panel-cap {
  display: block;
  margin-top: 8px;
  padding: 6px 10px;
  font-size: 11px;
  letter-spacing: 0.5px;
  text-align: center;
  color: var(--pf-pill-fg);
  background: var(--pf-card-bg);
  border: 1px solid var(--pf-border);
  border-radius: 4px;
}
[data-testid="stAlert"] {
  border-radius: 4px;
  border-left: 4px solid var(--pf-purple);
}
[data-testid="stAlert"] a { color: var(--pf-accent-text); }

/* Pixel-style footer. */
.pf-footer {
  margin-top: 26px;
  padding: 16px;
  text-align: center;
  font-size: 11px;
  letter-spacing: 1px;
  color: var(--pf-footer-fg);
  border-top: 2px solid var(--pf-border);
}
.pf-footer a { color: var(--pf-accent-text); text-decoration: none; }
.pf-footer a:hover { text-decoration: underline; }
</style>
"""

# Light-theme readability sheet. Injected ONLY when light mode is active.
# (A previous attempt scoped these under html[data-pf-theme="light"] and set
# that attribute with a <script> — but Streamlit strips <script> from
# st.markdown, so the attribute was never applied and light mode never took
# effect. Verified in a real browser. Injecting conditionally avoids JS
# entirely and is guaranteed to work.)
_CSS_LIGHT_FIXES = """
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
[data-testid="stSidebar"] * { color: var(--pf-text); }
[data-testid="stCaptionContainer"], small, .stMarkdown p,
[data-testid="stWidgetLabel"] p, [data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small { color: var(--pf-text-muted) !important; }
[data-testid="stFileUploader"] { background: #ffffff; color: var(--pf-text-muted); }
/* The uploader's inner dropzone + its button ship with their own dark
   background in Streamlit's default theme; on a white page they read as a
   jarring dark slab. Repaint them light. */
[data-testid="stFileUploaderDropzone"] { background: #ffffff !important; }
[data-testid="stFileUploaderDropzone"] button {
  background: #eef2ff !important;
  color: var(--pf-text) !important;
  border-color: var(--pf-border) !important;
}
[data-testid="stFileUploaderDropzone"] svg { fill: var(--pf-text-muted); }
[data-testid="stAlert"] { background: #eef2ff; color: var(--pf-text); }
[data-testid="stAlert"] * { color: var(--pf-text) !important; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: var(--pf-text); }
hr { border-color: rgba(15, 23, 42, 0.15); }
</style>
"""

_CSS_VARS_DARK = """
<style>
:root {
  --pf-cyan: #22d3ee;
  --pf-purple: #a855f7;
  --pf-pink: #f472b6;
  --pf-font: "Courier New", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --pf-app-bg:
    radial-gradient(circle at 15% 10%, rgba(168, 85, 247, 0.22), transparent 42%),
    radial-gradient(circle at 85% 25%, rgba(34, 211, 238, 0.18), transparent 40%),
    repeating-linear-gradient(0deg, rgba(255,255,255,0.022) 0px, rgba(255,255,255,0.022) 1px, transparent 1px, transparent 3px),
    #0b0a1f;
  --pf-border: rgba(34, 211, 238, 0.35);
  --pf-hero-bg: linear-gradient(160deg, rgba(34,211,238,0.10), rgba(168,85,247,0.16));
  --pf-hero-shadow: 0 0 0 3px rgba(11,10,31,0.9), 0 0 24px rgba(168,85,247,0.35);
  --pf-title-color: #fff;
  --pf-title-shadow: 3px 3px 0 #a855f7, 6px 6px 0 rgba(34,211,238,0.55);
  --pf-accent-text: var(--pf-cyan);
  --pf-badge-fg: #0b0a1f;
  --pf-btn-bg: rgba(34, 211, 238, 0.10);
  --pf-btn-fg: #eaffff;
  --pf-btn-shadow: rgba(168, 85, 247, 0.45);
  --pf-btn-shadow-hover: rgba(244, 114, 182, 0.6);
  --pf-sidebar-bg: linear-gradient(180deg, #171338, #0f0c2b);
  --pf-sidebar-border: rgba(168, 85, 247, 0.35);
  --pf-img-border: rgba(34, 211, 238, 0.5);
  --pf-img-shadow: rgba(168, 85, 247, 0.35);
  --pf-card-bg: rgba(23, 19, 56, 0.55);
  --pf-pill-bg: rgba(23, 19, 56, 0.75);
  --pf-pill-fg: #cbd5e1;
  --pf-pill-border: rgba(148, 163, 184, 0.35);
  --pf-pill-on-fg: #eaffff;
  --pf-pill-on-border: rgba(34, 211, 238, 0.75);
  --pf-pill-dot: #22d3ee;
  --pf-focus-ring: rgba(34, 211, 238, 0.20);
  --pf-footer-fg: #64748b;
}
</style>
"""

_CSS_VARS_LIGHT = """
<style>
:root {
  --pf-cyan: #0e7490;
  --pf-purple: #6d28d9;
  --pf-pink: #be185d;
  --pf-font: "Courier New", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --pf-app-bg:
    radial-gradient(circle at 15% 10%, rgba(168, 85, 247, 0.08), transparent 42%),
    radial-gradient(circle at 85% 25%, rgba(8, 145, 178, 0.08), transparent 40%),
    #f5f7fb;
  --pf-border: rgba(8, 145, 178, 0.5);
  --pf-hero-bg: linear-gradient(160deg, rgba(8,145,178,0.10), rgba(124,58,237,0.12));
  --pf-hero-shadow: 0 0 0 3px rgba(255,255,255,0.9), 0 0 18px rgba(124,58,237,0.20);
  --pf-title-color: #1e1b4b;
  --pf-title-shadow: 3px 3px 0 rgba(124,58,237,0.35), 6px 6px 0 rgba(8,145,178,0.25);
  --pf-accent-text: #0e7490;
  --pf-badge-fg: #ffffff;
  --pf-btn-bg: rgba(8, 145, 178, 0.10);
  --pf-btn-fg: #0e7490;
  --pf-btn-shadow: rgba(124, 58, 237, 0.30);
  --pf-btn-shadow-hover: rgba(219, 39, 119, 0.4);
  --pf-sidebar-bg: linear-gradient(180deg, #eef2ff, #e0e7ff);
  --pf-sidebar-border: rgba(124, 58, 237, 0.25);
  --pf-img-border: rgba(8, 145, 178, 0.55);
  --pf-img-shadow: rgba(124, 58, 237, 0.22);
  --pf-card-bg: rgba(255, 255, 255, 0.75);
  --pf-pill-bg: #ffffff;
  --pf-pill-fg: #334155;
  --pf-pill-border: rgba(100, 116, 139, 0.35);
  --pf-pill-on-fg: #0e7490;
  --pf-pill-on-border: rgba(8, 145, 178, 0.65);
  --pf-pill-dot: #0e7490;
  --pf-focus-ring: rgba(8, 145, 178, 0.18);
  --pf-footer-fg: #64748b;
  /* Streamlit-native text colours for the light theme (fixes low-contrast
     grey-on-white on captions / info boxes / sidebar labels / uploader). */
  --pf-text: #0f172a;
  --pf-text-muted: #334155;
}
</style>
"""

# --- session state defaults -------------------------------------------------
st.session_state.setdefault("lang", "en")      # English by default
st.session_state.setdefault("theme", "dark")   # dark by default

# Inject ONLY the active theme's variables, under plain :root, plus the common
# sheet. The earlier "inject both, scope by html[data-pf-theme], flip via JS"
# approach was WRONG: Streamlit sanitises <script> out of st.markdown, so the
# attribute was never set and the theme never changed (verified in a real
# browser: documentElement had no data-pf-theme). Rendering the active sheet
# directly is guaranteed to work with zero JS. The light-theme readability
# overrides are still scoped by a *selector* (html[data-pf-theme="light"]) in
# _CSS_COMMON, which we now drive by emitting that attribute as a real DOM
# attribute on a wrapper we fully control — see _THEME_ATTR below.
st.markdown(_CSS_VARS_DARK if st.session_state.theme == "dark" else _CSS_VARS_LIGHT,
            unsafe_allow_html=True)
st.markdown(_CSS_COMMON, unsafe_allow_html=True)
if st.session_state.theme == "light":
    st.markdown(_CSS_LIGHT_FIXES, unsafe_allow_html=True)

# --- sidebar controls (language + theme first, then task/scale) -------------
# Widgets are bound directly to session_state via `key=`. The earlier version
# set both `index=` AND then overwrote session_state by hand — the classic
# Streamlit anti-pattern that makes the widget and the state disagree and
# triggers an extra, janky rerun on every change. With `key=` there is exactly
# one rerun per click and the state stays authoritative.
with st.sidebar:
    st.radio(
        "Language / 语言", ["en", "zh"],
        format_func=lambda k: "English" if k == "en" else "中文",
        horizontal=True, key="lang",
    )
    T = TEXTS[st.session_state.lang]

    st.radio(
        T["theme_label"], ["dark", "light"],
        format_func=lambda k: T["theme_dark"] if k == "dark" else T["theme_light"],
        horizontal=True, key="theme",
    )

    st.divider()
    st.header(T["sidebar_header"])
    # NOTE: option VALUES are fixed ("sr"/"lowlight"); only the LABELS are
    # translated via format_func, so task routing never depends on language.
    st.radio(T["task_label"], ["sr", "lowlight"],
             format_func=lambda k: T["task_sr"] if k == "sr" else T["task_lowlight"],
             key="task")
    st.radio(T["scale_label"], ["2", "4"], index=1, help=T["scale_help"], key="scale")
    st.divider()
    st.caption(T["trained_caption"])

st.markdown(f'<p class="pf-section-title">{T["sec_upload"]}</p>',
            unsafe_allow_html=True)
uploaded = st.file_uploader(T["upload_label"], type=["png", "jpg", "jpeg", "bmp", "webp"])

# --- render-time engine probe (files only; no torch.jit.load) ---------------
sr2_ready = _sr_weight_path(2) is not None
sr4_ready = _sr_weight_path(4) is not None
low_ready = _lowlight_weight_path() is not None
sr2_tag = T["tag_ml"] if sr2_ready else T["tag_classical"]
sr4_tag = T["tag_ml"] if sr4_ready else T["tag_classical"]
low_tag = T["tag_ml"] if low_ready else T["tag_classical"]

st.markdown(
    f"""
    <div class="pf-hero">
      <p class="pf-title">PIXELFORGE</p>
      <p class="pf-sub">&gt; SUPER-RESOLUTION &amp; LOW-LIGHT ENHANCEMENT&lt;</p>
      <div class="pf-pills">
        <span class="pf-pill {'on' if sr2_ready else ''}">
          <span class="pf-dot"></span>SR ×2 · {sr2_tag}</span>
        <span class="pf-pill {'on' if sr4_ready else ''}">
          <span class="pf-dot"></span>SR ×4 · {sr4_tag}</span>
        <span class="pf-pill {'on' if low_ready else ''}">
          <span class="pf-dot"></span>LOW-LIGHT · {low_tag}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    use_sr = st.session_state.task == "sr"
    scale = st.session_state.scale

    st.markdown(f'<p class="pf-section-title">{T["sec_result"]}</p>',
                unsafe_allow_html=True)
    sr_pair = None
    ml_lr = None
    ll_triple = None          # (original, classical, model) for the 3-panel view
    ll_used_model = False
    ll_notdark = False
    with st.spinner(T["spinner"]):
        if use_sr:
            scale_i = int(scale)
            sr_pair = predict_sr(img, scale_i)
            if sr_pair is None:
                out = sr_classical(img, scale_i)
                st.warning(T["warn_sr_no_weight"].format(scale=scale_i))
            else:
                ml_lr, out = sr_pair
                st.success(T["ok_sr"].format(scale=scale_i))
        else:
            # Always compute BOTH so the 3-panel view can show them side by side.
            classical_out = lowlight_classical(img)
            is_dark, _stats = looks_underexposed(img)
            ml_out = predict_lowlight(img) if is_dark else None
            if ml_out is None:
                # Either no weight, or the image is not actually underexposed.
                out = classical_out
                if _lowlight_weight_path() is None:
                    st.warning(T["warn_ll_no_weight"])
                else:
                    ll_notdark = True
                    st.info(T["note_ll_notdark"])
            else:
                ll_used_model = True
                out = ml_out
                st.success(T["ok_ll"])
            ll_triple = (img, classical_out, ml_out)

    if use_sr and sr_pair is not None and ml_lr is not None:
        # 3-panel view: original / model's real (low-res) input / SR output.
        # The middle panel is what stops a full-res upload from being compared
        # against a reconstruction that only had 1/scale^2 of the pixels.
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(img, width="stretch")
            st.markdown(f'<span class="pf-panel-cap">{T["cap_original"]}</span>',
                        unsafe_allow_html=True)
        with c2:
            lr_display = ml_lr.resize(out.size, Image.NEAREST)
            st.image(lr_display, width="stretch")
            st.markdown('<span class="pf-panel-cap">'
                        + T["cap_lr"].format(w=ml_lr.width, h=ml_lr.height)
                        + "</span>", unsafe_allow_html=True)
        with c3:
            st.image(out, width="stretch")
            st.markdown('<span class="pf-panel-cap">'
                        + T["cap_sr_out"].format(scale=scale)
                        + "</span>", unsafe_allow_html=True)
        st.info(T["info_3panel"].format(scale=scale))
    elif not use_sr and ll_triple is not None and ll_triple[2] is not None:
        # Low-light 3-panel view: input / classical baseline / self-trained model.
        # Shown ONLY when the model actually ran, so the honest side-by-side is
        # visible exactly when the domain-gap question can arise. (When the
        # model was skipped we fall through to the simple Before/After pair.)
        ll_in, ll_cls, ll_mod = ll_triple
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(ll_in, width="stretch")
            st.markdown(f'<span class="pf-panel-cap">{T["cap_before"]}</span>',
                        unsafe_allow_html=True)
        with c2:
            st.image(ll_cls, width="stretch")
            st.markdown('<span class="pf-panel-cap">'
                        + T["tag_classical"] + "</span>",
                        unsafe_allow_html=True)
        with c3:
            st.image(ll_mod, width="stretch")
            st.markdown('<span class="pf-panel-cap">'
                        + T["tag_ml"] + "</span>", unsafe_allow_html=True)
        st.info(T["info_ll_3panel"])
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.image(img, width="stretch")
            st.markdown(f'<span class="pf-panel-cap">{T["cap_before"]}</span>',
                        unsafe_allow_html=True)
        with c2:
            st.image(out, width="stretch")
            st.markdown(f'<span class="pf-panel-cap">{T["cap_after"]}</span>',
                        unsafe_allow_html=True)

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    st.download_button(T["download"], buf.getvalue(),
                       file_name="pixelforge_after.png", mime="image/png")
else:
    st.info(T["empty_info"])

st.markdown(
    f'<div class="pf-footer">PIXELFORGE · PyTorch · FastAPI · Next.js · '
    f'TorchScript · <a href="https://github.com/ElijahZhao/'
    f'pixelforge-image-restoration-" target="_blank">GitHub</a><br>'
    f'{T["footer"]}</div>',
    unsafe_allow_html=True,
)
