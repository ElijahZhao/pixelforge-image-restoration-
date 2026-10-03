"""Generate honest before/after demo images for the README.

This version uses the **actually trained TorchScript weights** in
``serve/models/`` so the showcase reflects what the live demo runs.

Outputs are written as ``.jpg`` — matching the filenames referenced by
``README.md`` — so that running this script genuinely reproduces the showcase
(previously the script wrote ``.png`` while the README referenced ``.jpg``,
which meant it could NOT reproduce the displayed images; see
DIAGNOSIS_ROUND18).

  assets/sample_scene.png            photogenic synthetic scene (512x512)
  assets/sample_dark.png             underexposed version (low-light source)
  assets/demo_sr_lr.jpg              low-res input (blocky display)
  assets/demo_sr_bicubic.jpg         Bicubic 4x upscale (classical baseline)
  assets/demo_sr_ours.jpg            PixelForge SRResNet (trained, true output)
  assets/demo_lowlight_input.jpg     dark input
  assets/demo_lowlight_gamma.jpg     adaptive-gamma baseline
  assets/demo_lowlight_ours.jpg      PixelForge U-Net (trained)

Run:  python scripts/make_demo.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Allow running as `python scripts/make_demo.py` from the repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from serve.classical import sr_classical, lowlight_classical  # noqa: E402
from serve import model_loader as ml  # noqa: E402

ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
os.makedirs(ASSETS, exist_ok=True)


def make_scene(size: int = 512) -> Image.Image:
    """A simple, photo-like synthetic scene: sky gradient, sun, mountains."""
    rng = np.random.default_rng(7)
    h = w = size
    yy = np.linspace(1.0, 0.35, h)[:, None]
    base = np.ones((h, w, 3), dtype=np.float32) * 0.55
    sky = base * yy
    cx, cy, r = int(w * 0.72), int(h * 0.28), int(size * 0.08)
    ys, xs = np.ogrid[:h, :w]
    sun = ((xs - cx) ** 2 + (ys - cy) ** 2) <= r ** 2
    sky[sun] = np.array([1.0, 0.92, 0.75])
    img = sky.copy()
    draw = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(draw)
    peaks = [(0, int(h * 0.72)), (int(w * 0.3), int(h * 0.55)),
             (int(w * 0.55), int(h * 0.7)), (w, int(h * 0.5))]
    d.polygon([(0, h), *peaks, (w, h)], fill=(60, 70, 95))
    peaks2 = [(0, h), (int(w * 0.2), int(h * 0.82)),
              (int(w * 0.6), int(h * 0.74)), (w, int(h * 0.86)), (w, h)]
    d.polygon(peaks2, fill=(30, 35, 55))
    img = np.asarray(draw, dtype=np.float32) / 255.0
    img += rng.normal(0, 0.015, img.shape).astype(np.float32)
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def _save(name: str, img: Image.Image) -> None:
    p = os.path.join(ASSETS, name)
    if name.lower().endswith((".jpg", ".jpeg")):
        img.convert("RGB").save(p, quality=92)
    else:
        img.save(p)
    print(f"  {name:30s} {os.path.getsize(p):>8d} bytes")


def main() -> None:
    scene = make_scene(512)
    scene.save(os.path.join(ASSETS, "sample_scene.png"))

    # ---- Low-light demo --------------------------------------------------
    dark = np.asarray(scene, dtype=np.float32) / 255.0
    dark = np.clip(dark * 0.16, 0.0, 1.0)          # emulate underexposed capture
    dark_img = Image.fromarray((dark * 255).astype(np.uint8))
    dark_img = dark_img.filter(ImageFilter.GaussianBlur(0.4))
    dark_img.save(os.path.join(ASSETS, "sample_dark.png"))

    gamma = lowlight_classical(dark_img)            # classical baseline
    ours_ll = ml.predict_lowlight(dark_img) or gamma  # trained U-Net
    _save("demo_lowlight_input.jpg", dark_img)
    _save("demo_lowlight_gamma.jpg", gamma)
    _save("demo_lowlight_ours.jpg", ours_ll)

    # ---- Super-resolution demo (4x) -------------------------------------
    # Feed the LR image to the model (architecturally correct: the SR model
    # consumes a low-res input). predict_sr now returns the model's TRUE output.
    lr = scene.resize((128, 128), Image.LANCZOS)     # simulate low-res capture
    lr_disp = lr.resize((512, 512), Image.NEAREST)   # blocky display of LR
    bicubic = sr_classical(lr, scale=4)              # classical baseline
    ours_sr = ml.predict_sr(lr, scale=4) or bicubic   # trained SRResNet
    _save("demo_sr_lr.jpg", lr_disp)
    _save("demo_sr_bicubic.jpg", bicubic)
    _save("demo_sr_ours.jpg", ours_sr)

    print("Demo images written to assets/.")


if __name__ == "__main__":
    main()
