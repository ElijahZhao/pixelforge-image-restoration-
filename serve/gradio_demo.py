"""Optional: one-command Gradio demo (no frontend build needed).

    pip install gradio
    python serve/gradio_demo.py

Great for a quick local prototype / HF Spaces before the full Next.js site is
ready. Reuses the same classical + ML inference logic as ``app.py``.
"""

from __future__ import annotations

import io

import gradio as gr
from PIL import Image

from classical import run_classical
from model_loader import predict_sr, predict_lowlight


def process(image, task, scale):
    if task == "sr":
        out = predict_sr(image, int(scale)) or run_classical(image, "sr", int(scale))
        return image, out
    out = predict_lowlight(image) or run_classical(image, "lowlight", int(scale))
    return image, out


demo = gr.Interface(
    fn=process,
    inputs=[
        gr.Image(type="pil", label="Input image"),
        gr.Radio(["sr", "lowlight"], value="sr", label="Task"),
        gr.Radio(["2", "4"], value="2", label="SR scale"),
    ],
    outputs=[gr.Image(label="Before"), gr.Image(label="After (enhanced)")],
    title="CV Image Restoration Demo",
    description="Super-resolution & low-light enhancement. Uses trained models "
                "if exported, otherwise classical baselines.",
)

if __name__ == "__main__":
    demo.launch()
