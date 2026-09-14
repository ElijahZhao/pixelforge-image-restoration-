"""Generate real before/after demo images using the classical baselines.

Runs entirely on CPU (no trained weights needed) so the portfolio has
*actual* sample outputs instead of placeholders. Produces:

  assets/sample_scene.png      normal-light source scene (512x512)
  assets/sample_dark.png       darkened version for low-light demo
  assets/demo_sr_before.png    bicubic 4x upscale (blurry reference)
  assets/demo_sr_after.png     classical SR (bicubic + unsharp)
  assets/demo_lowlight_before.png   dark input
  assets/demo_lowlight_after.png    adaptive-gamma enhanced

Run:  python scripts/make_demo.py
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from serve.classical import sr_classical, lowlight_classical

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
ASSETS = os.path.abspath(ASSETS)
os.makedirs(ASSETS, exist_ok=True)


def make_scene(size: int = 512) -> Image.Image:
    """A simple, photo-like synthetic scene: sky gradient, sun, mountains."""
    rng = np.random.default_rng(7)
    h = w = size
    # Sky vertical gradient (top lighter -> horizon)
    yy = np.linspace(1.0, 0.35, h)[:, None]
    base = np.ones((h, w, 3), dtype=np.float32) * 0.55
    sky = base * yy  # darker toward bottom
    # Sun (bright disc) upper area
    cx, cy, r = int(w * 0.72), int(h * 0.28), int(size * 0.08)
    ys, xs = np.ogrid[:h, :w]
    sun = ((xs - cx) ** 2 + (ys - cy) ** 2) <= r ** 2
    sky[sun] = np.array([1.0, 0.92, 0.75])
    # Mountains: a few overlapping triangles (dark silhouettes)
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
    # Mild film grain for realism
    img += rng.normal(0, 0.015, img.shape).astype(np.float32)
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def main() -> None:
    scene = make_scene(512)
    scene.save(os.path.join(ASSETS, "sample_scene.png"))

    # ---- Low-light demo ----
    dark = np.asarray(scene, dtype=np.float32) / 255.0
    dark = np.clip(dark * 0.18, 0.0, 1.0)  # emulate underexposed capture
    dark_img = Image.fromarray((dark * 255).astype(np.uint8))
    dark_img = dark_img.filter(ImageFilter.GaussianBlur(0.4))
    dark_img.save(os.path.join(ASSETS, "sample_dark.png"))
    lit = lowlight_classical(dark_img)
    dark_img.save(os.path.join(ASSETS, "demo_lowlight_before.png"))
    lit.save(os.path.join(ASSETS, "demo_lowlight_after.png"))

    # ---- Super-resolution demo (4x) ----
    small = scene.resize((128, 128), Image.LANCZOS)  # simulate low-res capture
    before = small.resize((512, 512), Image.BICUBIC)  # blurry reference
    before.save(os.path.join(ASSETS, "demo_sr_before.png"))
    after = sr_classical(small, scale=4)
    after.save(os.path.join(ASSETS, "demo_sr_after.png"))

    print("Demo images written to assets/:")
    for f in sorted(os.listdir(ASSETS)):
        if f.startswith("demo_") or f.startswith("sample_"):
            p = os.path.join(ASSETS, f)
            print(f"  {f:28s} {os.path.getsize(p):>8d} bytes")


if __name__ == "__main__":
    main()
