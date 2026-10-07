"""Optional: one-command Gradio demo (no frontend build needed).

    pip install gradio
    python -m serve.gradio_demo          # from the repo root
    # or, from inside serve/:
    python gradio_demo.py

Great for a quick local prototype / HF Spaces before the full Next.js site is
ready. Reuses the same classical + ML inference logic as ``app.py`` (including
the low-light exposure gate and output guard; otherwise this entry point would
still exhibit the "enhancement makes it darker" behaviour).
"""

from __future__ import annotations

import gradio as gr

# Support BOTH invocation styles: `python -m serve.gradio_demo` (package import,
# needs the relative form) and `python serve/gradio_demo.py` (script import, where
# the sibling modules are on sys.path directly). The previous unconditional
# `from classical import ...` broke the documented `-m` form with ImportError.
try:  # pragma: no cover - exercised by whichever entry point the user picks
    from .classical import run_classical
    from .model_loader import (
        predict_sr,
        predict_lowlight,
        looks_underexposed,
        mean_luminance,
        LOWLIGHT_MIN_GAIN,
    )
except ImportError:  # run as a plain script
    from classical import run_classical  # type: ignore
    from model_loader import (  # type: ignore
        predict_sr,
        predict_lowlight,
        looks_underexposed,
        mean_luminance,
        LOWLIGHT_MIN_GAIN,
    )


def process(image, task, scale):
    if image is None:
        return None, None
    scale = int(scale)
    if task == "sr":
        out = predict_sr(image, scale) or run_classical(image, "sr", scale)
        return image, out
    # Low-light: only run the learned model on a genuinely dark input, and discard
    # its output if it made the image darker (see serve/model_loader.py).
    is_dark, _ = looks_underexposed(image)
    out = predict_lowlight(image) if is_dark else None
    if out is not None and (mean_luminance(out) - mean_luminance(image)) < LOWLIGHT_MIN_GAIN:
        out = None
    return image, out or run_classical(image, "lowlight", scale)


demo = gr.Interface(
    fn=process,
    inputs=[
        gr.Image(type="pil", label="Input image"),
        gr.Radio(["sr", "lowlight"], value="sr", label="Task"),
        gr.Radio(["2", "4"], value="4", label="SR scale"),
    ],
    outputs=[gr.Image(label="Before"), gr.Image(label="After (enhanced)")],
    title="CV Image Restoration Demo",
    description="Super-resolution & low-light enhancement. Uses trained models "
                "if exported, otherwise classical baselines.",
)

if __name__ == "__main__":
    demo.launch()
