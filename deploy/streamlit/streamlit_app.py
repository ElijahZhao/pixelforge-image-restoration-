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
        "cap_original": "① Original (your upload)",
        "cap_lr": "② Model input (LR {w}×{h}, upscaled for display)",
        "cap_sr_out": "③ PixelForge SR output (true ×{scale})",
        "cap_before": "Before",
        "cap_after": "After (enhanced)",
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
        "cap_original": "① 原图 (your upload)",
        "cap_lr": "② 模型实际输入 (低清 {w}×{h}，放大显示)",
        "cap_sr_out": "③ PixelForge 超分输出 (真实 ×{scale})",
        "cap_before": "Before",
        "cap_after": "After (enhanced)",
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
/* Hide only the "chrome" (menu / deploy button / footer), NOT the header
   itself. The sidebar's re-open control (stSidebarCollapsedControl, the «
   button shown after collapsing) lives INSIDE the header — hiding
   header[data-testid="stHeader"] made a collapsed sidebar impossible to
   re-open. This is the fix for that bug. */
#MainMenu, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
/* Keep the collapsed-sidebar re-open button visible and clickable. */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapsedControl"] button { visibility: visible; }

.pf-hero {
  text-align: center;
  padding: 30px 16px 24px;
  margin-bottom: 20px;
  border: 3px solid var(--pf-border);
  border-radius: 4px;
  background: var(--pf-hero-bg);
  box-shadow: var(--pf-hero-shadow);
  position: relative;
  overflow: visible;
}
.pf-title {
  font-family: var(--pf-font);
  font-size: 28px;
  font-weight: 700;
  line-height: 1.7;
  color: var(--pf-title-color);
  text-shadow: var(--pf-title-shadow);
  margin: 0 0 16px;
  letter-spacing: 2px;
  word-spacing: 4px;
}
.pf-sub { font-size: 13px; color: var(--pf-accent-text); margin: 0; }
.pf-badge {
  display: inline-block;
  margin-top: 12px;
  padding: 5px 12px;
  font-size: 11px;
  color: var(--pf-badge-fg);
  background: var(--pf-cyan);
  border-radius: 3px;
  font-weight: 700;
}

.stButton > button, .stDownloadButton > button {
  font-family: "Courier New", monospace;
  font-weight: 700;
  border: 2px solid var(--pf-cyan);
  border-radius: 3px;
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
[data-testid="stFileUploader"] {
  border: 2px dashed var(--pf-border);
  border-radius: 4px;
  padding: 6px;
}
[data-testid="stImage"] img {
  border: 3px solid var(--pf-img-border);
  border-radius: 3px;
  box-shadow: 4px 4px 0 var(--pf-img-shadow);
}
[data-testid="stAlert"] {
  border-radius: 3px;
  border-left: 4px solid var(--pf-purple);
}
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
}
</style>
"""

_CSS_VARS_LIGHT = """
<style>
:root {
  --pf-cyan: #0891b2;
  --pf-purple: #7c3aed;
  --pf-pink: #db2777;
  --pf-font: "Courier New", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --pf-app-bg:
    radial-gradient(circle at 15% 10%, rgba(168, 85, 247, 0.10), transparent 42%),
    radial-gradient(circle at 85% 25%, rgba(8, 145, 178, 0.10), transparent 40%),
    #f5f7fb;
  --pf-border: rgba(8, 145, 178, 0.45);
  --pf-hero-bg: linear-gradient(160deg, rgba(8,145,178,0.08), rgba(124,58,237,0.10));
  --pf-hero-shadow: 0 0 0 3px rgba(255,255,255,0.9), 0 0 18px rgba(124,58,237,0.18);
  --pf-title-color: #1e1b4b;
  --pf-title-shadow: 3px 3px 0 rgba(124,58,237,0.35), 6px 6px 0 rgba(8,145,178,0.25);
  --pf-accent-text: #0e7490;
  --pf-badge-fg: #ffffff;
  --pf-btn-bg: rgba(8, 145, 178, 0.08);
  --pf-btn-fg: #0e7490;
  --pf-btn-shadow: rgba(124, 58, 237, 0.30);
  --pf-btn-shadow-hover: rgba(219, 39, 119, 0.4);
  --pf-sidebar-bg: linear-gradient(180deg, #eef2ff, #e0e7ff);
  --pf-sidebar-border: rgba(124, 58, 237, 0.25);
  --pf-img-border: rgba(8, 145, 178, 0.5);
  --pf-img-shadow: rgba(124, 58, 237, 0.22);
}
</style>
"""

# --- session state defaults -------------------------------------------------
st.session_state.setdefault("lang", "en")      # English by default
st.session_state.setdefault("theme", "dark")   # dark by default

st.markdown(_CSS_VARS_DARK if st.session_state.theme == "dark" else _CSS_VARS_LIGHT,
            unsafe_allow_html=True)
st.markdown(_CSS_COMMON, unsafe_allow_html=True)

# --- sidebar controls (language + theme first, then task/scale) -------------
with st.sidebar:
    lang_choice = st.radio(
        "Language / 语言", ["English", "中文"],
        index=0 if st.session_state.lang == "en" else 1,
        horizontal=True,
    )
    st.session_state.lang = "en" if lang_choice == "English" else "zh"
    T = TEXTS[st.session_state.lang]

    theme_choice = st.radio(
        T["theme_label"], [T["theme_dark"], T["theme_light"]],
        index=0 if st.session_state.theme == "dark" else 1,
        horizontal=True,
    )
    st.session_state.theme = "dark" if theme_choice == T["theme_dark"] else "light"

    st.divider()
    st.header(T["sidebar_header"])
    # NOTE: option VALUES are fixed ("sr"/"lowlight"); only the LABELS are
    # translated via format_func, so task routing never depends on language.
    task = st.radio(T["task_label"], ["sr", "lowlight"],
                    format_func=lambda k: T["task_sr"] if k == "sr" else T["task_lowlight"])
    scale = st.radio(T["scale_label"], ["2", "4"], index=1, help=T["scale_help"])
    st.divider()
    st.caption(T["trained_caption"])

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
      <p class="pf-badge">SR ×2 · {sr2_tag} ｜ SR ×4 · {sr4_tag} ｜ LOW-LIGHT · {low_tag}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    use_sr = task == "sr"

    sr_pair = None
    ml_lr = None
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
            ml_out = predict_lowlight(img)
            out = ml_out or lowlight_classical(img)
            if ml_out is None:
                st.warning(T["warn_ll_no_weight"])
            else:
                st.success(T["ok_ll"])

    if use_sr and sr_pair is not None and ml_lr is not None:
        # 3-panel view: original / model's real (low-res) input / SR output.
        # The middle panel is what stops a full-res upload from being compared
        # against a reconstruction that only had 1/scale^2 of the pixels.
        c1, c2, c3 = st.columns(3)
        c1.image(img, caption=T["cap_original"], width="stretch")
        lr_display = ml_lr.resize(out.size, Image.NEAREST)
        c2.image(lr_display,
                 caption=T["cap_lr"].format(w=ml_lr.width, h=ml_lr.height),
                 width="stretch")
        c3.image(out, caption=T["cap_sr_out"].format(scale=scale), width="stretch")
        st.info(T["info_3panel"].format(scale=scale))
    else:
        c1, c2 = st.columns(2)
        c1.image(img, caption=T["cap_before"], width="stretch")
        c2.image(out, caption=T["cap_after"], width="stretch")

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    st.download_button(T["download"], buf.getvalue(),
                       file_name="pixelforge_after.png", mime="image/png")
else:
    st.info(T["empty_info"])

st.divider()
st.caption(T["footer"])
