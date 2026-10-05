"""Regenerate the README comparison figures from the demo images.

Reproducibility fix: the previous ``assets/compare_sr_4x.png`` /
``assets/compare_lowlight.png`` were produced by a throwaway one-off snippet
that was never committed, so the figures could not be regenerated. This
script composes the 3-panel figures from ``assets/demo_*.jpg`` (which
``make_demo.py`` produces via the CORRECT pipeline: LR fed straight to the
traced model, see the comment in ``make_demo.py``).

Captions state DIV2K/LOL full-image validation AVERAGES (from
``scripts/eval_baseline.py``), explicitly labeled as such; they are not the
per-image scores of the synthetic demo scene.

Run:  python scripts/make_compare.py
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw

ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))

W_PANEL, H_PANEL = 350, 350          # panel image area
PAD = 12
TITLE_H, CAP_H, NOTE_H = 34, 30, 40
BG = (18, 18, 22)
FG = (235, 235, 235)
MUTED = (150, 150, 160)
ACCENT = (140, 220, 160)


def _text(d: ImageDraw.ImageDraw, xy, s, fill=FG, size=16, anchor=None):
    d.text(xy, s, fill=fill, anchor=anchor)


def compose(rows, out_name: str, note: str) -> None:
    """rows: list of (title, image_path, caption, caption_color)."""
    n = len(rows)
    W = PAD + n * (W_PANEL + PAD)
    H = TITLE_H + H_PANEL + CAP_H + NOTE_H + PAD
    fig = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(fig)
    for i, (title, path, cap, cap_color) in enumerate(rows):
        x0 = PAD + i * (W_PANEL + PAD)
        _text(d, (x0 + W_PANEL // 2, 8), title, fill=FG, size=18,
              anchor="ma")
        img = Image.open(os.path.join(ASSETS, path)).convert("RGB")
        img = img.resize((W_PANEL, H_PANEL), Image.LANCZOS)
        fig.paste(img, (x0, TITLE_H))
        _text(d, (x0 + W_PANEL // 2, TITLE_H + H_PANEL + 8), cap,
              fill=cap_color, size=15, anchor="ma")
    _text(d, (W // 2, TITLE_H + H_PANEL + CAP_H + 6), note,
          fill=MUTED, size=13, anchor="ma")
    out = os.path.join(ASSETS, out_name)
    fig.save(out)
    print(f"  {out_name:26s} {os.path.getsize(out):>8d} bytes")


def main() -> None:
    compose(
        [
            ("LR input (128px)", "demo_sr_lr.jpg",
             "(shown enlarged)", MUTED),
            ("Bicubic x4 baseline", "demo_sr_bicubic.jpg",
             "PSNR 26.69 / SSIM 0.754", FG),
            ("PixelForge SRResNet x4", "demo_sr_ours.jpg",
             "PSNR 27.47 / SSIM 0.780 (+0.77 dB)", ACCENT),
        ],
        "compare_sr_4x.png",
        "PSNR/SSIM are DIV2K full-image validation AVERAGES (eval_baseline.py), "
        "not this scene. Ours is the model's true 4x reconstruction of the 128px LR input.",
    )
    compose(
        [
            ("Dark input", "demo_lowlight_input.jpg",
             "(underexposed x0.16)", MUTED),
            ("No-op baseline", "demo_lowlight_input.jpg",
             "PSNR 7.77 / SSIM 0.192", FG),
            ("PixelForge U-Net", "demo_lowlight_ours.jpg",
             "PSNR 18.18 / SSIM 0.739 (+10.41 dB)", ACCENT),
        ],
        "compare_lowlight.png",
        "PSNR/SSIM are LOL full-image validation AVERAGES (eval_baseline.py), "
        "not this scene. The no-op baseline is the identity map: its output IS the dark input.",
    )
    print("Comparison figures written to assets/.")


if __name__ == "__main__":
    main()
