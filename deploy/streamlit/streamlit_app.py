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
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st
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
# TorchScript model loading (mirror of serve/model_loader.py, with size fix)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _load_sr(scale: int):
    matches = sorted(MODELS_DIR.glob(f"sr_*_scale{scale}.pt"))
    if not matches:
        return None
    return torch.jit.load(str(matches[0]), map_location=DEVICE).eval()


@st.cache_resource(show_spinner=False)
def _load_lowlight():
    p = MODELS_DIR / "lowlight.pt"
    if not p.exists():
        return None
    return torch.jit.load(str(p), map_location=DEVICE).eval()


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


@torch.no_grad()
def predict_sr(img: Image.Image, scale: int) -> Image.Image | None:
    model = _load_sr(scale)
    if model is None:
        return None
    lr = img.resize((max(1, img.width // scale), max(1, img.height // scale)), Image.BICUBIC)
    x = TF.to_tensor(lr).unsqueeze(0).to(DEVICE)
    out = model(x).clamp(0, 1)
    out = TF.to_pil_image(out.squeeze(0).cpu())
    return out.resize((img.width * scale, img.height * scale), Image.BICUBIC)


@torch.no_grad()
def predict_lowlight(img: Image.Image) -> Image.Image | None:
    model = _load_lowlight()
    if model is None:
        return None
    padded, (w, h) = _pad_to_multiple(img, 32)
    x = TF.to_tensor(padded).unsqueeze(0).to(DEVICE)
    out = model(x).clamp(0, 1)
    out = TF.to_pil_image(out.squeeze(0).cpu())
    return out.crop((0, 0, w, h))


# --------------------------------------------------------------------------- #
# Streamlit UI —— 复古像素 / 游戏风
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="PixelForge · Image Restoration",
                   page_icon="👾", layout="centered")

PIXEL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

:root {
  --pf-cyan: #22d3ee;
  --pf-purple: #a855f7;
  --pf-pink: #f472b6;
  --pf-bg: #0b0a1f;
  --pf-card: rgba(23, 19, 56, 0.72);
  --pf-border: rgba(34, 211, 238, 0.35);
}

/* 全局等宽 + 轻微像素渲染 */
html, body, [class*="css"] {
  font-family: "Courier New", ui-monospace, SFMono-Regular, Menlo, monospace;
  letter-spacing: 0.2px;
}

/* 深空背景 + 霓虹光晕 + 扫描线 */
.stApp {
  background:
    radial-gradient(circle at 15% 10%, rgba(168, 85, 247, 0.22), transparent 42%),
    radial-gradient(circle at 85% 25%, rgba(34, 211, 238, 0.18), transparent 40%),
    repeating-linear-gradient(0deg, rgba(255,255,255,0.022) 0px, rgba(255,255,255,0.022) 1px, transparent 1px, transparent 3px),
    var(--pf-bg);
}

/* 隐藏 Streamlit 框架元素，减少"框架感" */
#MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* 自定义 Hero 标题 */
.pf-hero {
  text-align: center;
  padding: 30px 16px 24px;
  margin-bottom: 20px;
  border: 3px solid var(--pf-border);
  border-radius: 4px;
  background: linear-gradient(160deg, rgba(34,211,238,0.10), rgba(168,85,247,0.16));
  box-shadow: 0 0 0 3px rgba(11,10,31,0.9), 0 0 24px rgba(168,85,247,0.35);
  position: relative;
  overflow: visible;
}
.pf-title {
  font-family: 'Press Start 2P', "Courier New", monospace;
  font-size: 28px;
  line-height: 1.7;
  color: #fff;
  text-shadow: 3px 3px 0 #a855f7, 6px 6px 0 rgba(34,211,238,0.55);
  margin: 0 0 16px;
  letter-spacing: 2px;
  word-spacing: 4px;
}
.pf-sub {
  font-size: 13px;
  color: var(--pf-cyan);
  margin: 0;
}
.pf-badge {
  display: inline-block;
  margin-top: 12px;
  padding: 5px 12px;
  font-size: 11px;
  color: var(--pf-bg);
  background: var(--pf-cyan);
  border-radius: 3px;
  font-weight: 700;
}

/* 像素风按钮 */
.stButton > button, .stDownloadButton > button {
  font-family: "Courier New", monospace;
  font-weight: 700;
  border: 2px solid var(--pf-cyan);
  border-radius: 3px;
  background: rgba(34, 211, 238, 0.10);
  color: #eaffff;
  transition: all 0.12s ease;
  box-shadow: 3px 3px 0 rgba(168, 85, 247, 0.45);
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: var(--pf-cyan);
  color: var(--pf-bg);
  transform: translate(-1px, -1px);
  box-shadow: 4px 4px 0 rgba(244, 114, 182, 0.6);
}

/* 侧栏 / 卡片容器统一暗紫描边 */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #171338, #0f0c2b);
  border-right: 2px solid rgba(168, 85, 247, 0.35);
}
[data-testid="stFileUploader"] {
  border: 2px dashed var(--pf-border);
  border-radius: 4px;
  padding: 6px;
}

/* 图片容器加像素描边 */
[data-testid="stImage"] img {
  border: 3px solid rgba(34, 211, 238, 0.5);
  border-radius: 3px;
  box-shadow: 4px 4px 0 rgba(168, 85, 247, 0.35);
}

/* 提示条重着色 */
[data-testid="stAlert"] {
  border-radius: 3px;
  border-left: 4px solid var(--pf-purple);
}
</style>
"""

st.markdown(PIXEL_CSS, unsafe_allow_html=True)

sr4_ready = _load_sr(4) is not None
low_ready = _load_lowlight() is not None
sr4_tag = "ML 自训模型" if sr4_ready else "classical 基线"
low_tag = "ML 自训模型" if low_ready else "classical 基线"

st.markdown(
    f"""
    <div class="pf-hero">
      <p class="pf-title">PIXELFORGE</p>
      <p class="pf-sub">&gt; SUPER-RESOLUTION &amp; LOW-LIGHT ENHANCEMENT&lt;</p>
      <p class="pf-badge">SR ×4 · {sr4_tag} ｜ LOW-LIGHT · {low_tag}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("👾 参数")
    task = st.radio("任务", ["超分辨率 (SR)", "低光增强 (Low-Light)"])
    scale = st.radio("SR 放大倍数", ["2", "4"], index=1,
                     help="×4 使用自训 SRResNet + 感知损失模型")
    st.divider()
    st.caption("模型在 AutoDL RTX 3080 Ti 上训练 · Low-light Best PSNR 19.26")

uploaded = st.file_uploader("上传图片", type=["png", "jpg", "jpeg", "bmp", "webp"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    use_sr = task.startswith("超分")

    with st.spinner("推理中…"):
        if use_sr:
            scale_i = int(scale)
            out = predict_sr(img, scale_i) or sr_classical(img, scale_i)
        else:
            out = predict_lowlight(img) or lowlight_classical(img)

    c1, c2 = st.columns(2)
    c1.image(img, caption="Before", width="stretch")
    c2.image(out, caption="After (enhanced)", width="stretch")

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    st.download_button("⬇️ 下载结果 PNG", buf.getvalue(),
                       file_name="pixelforge_after.png", mime="image/png")
else:
    st.info("👾 请上传一张图片开始体验。低光任务建议用较暗的照片；超分建议用低分辨率图。")

st.divider()
st.caption("PIXELFORGE · press start to restore your images")
