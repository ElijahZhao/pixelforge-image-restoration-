---
title: PixelForge Image Restoration
emoji: 🖼️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# PixelForge · Image Restoration Demo

Interactive demo for **super-resolution** and **low-light enhancement**,
backed by the trained models from the
[PixelForge](https://github.com/ElijahZhao/pixelforge-image-restoration-) project.

- **Super-Resolution ×4** — SRResNet generator trained on DIV2K
- **Low-light Enhancement** — U-Net-style model trained on LOL

When the exported TorchScript weights are present in `models/`, the demo runs the
learned models; otherwise it falls back to classical baselines (bicubic +
unsharp mask / adaptive gamma).
