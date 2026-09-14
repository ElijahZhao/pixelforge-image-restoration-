"""FastAPI inference service for the image restoration / enhancement website.

Endpoints
---------
  GET  /api/health          -> service status (which engines are ML vs classical)
  POST /api/predict         -> upload an image, get before/after as base64 PNG

The service uses trained TorchScript models when present in ``serve/models/``;
otherwise it transparently falls back to classical baselines so the site works
out of the box.

Run locally:
    uvicorn serve.app:app --reload --port 8000
"""

from __future__ import annotations

import base64
import io
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from classical import run_classical
from model_loader import get_sr_model, get_lowlight_model, predict_sr, predict_lowlight

app = FastAPI(title="CV Restoration API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


def _img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "engines": {
            "sr_scale2": "ml" if get_sr_model(2) else "classical",
            "sr_scale4": "ml" if get_sr_model(4) else "classical",
            "lowlight": "ml" if get_lowlight_model() else "classical",
        },
    }


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(...),
    task: str = Form("sr"),
    scale: int = Form(2),
):
    if task not in ("sr", "lowlight"):
        return JSONResponse(
            status_code=422, content={"error": "task must be 'sr' or 'lowlight'"}
        )
    if scale not in (2, 4):
        scale = 2

    try:
        original = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except Exception:
        return JSONResponse(
            status_code=422, content={"error": "uploaded file is not a valid image"}
        )

    if task == "sr":
        ml_result = predict_sr(original, scale)
        engine = "ml" if ml_result is not None else "classical"
        result = ml_result if ml_result is not None else run_classical(original, task, scale)
        # "before" = original shown at the upscaled size (blurry reference)
        before = original.resize((original.width * scale, original.height * scale),
                                Image.BICUBIC)
    else:  # lowlight
        ml_result = predict_lowlight(original)
        engine = "ml" if ml_result is not None else "classical"
        result = ml_result if ml_result is not None else run_classical(original, task, scale)
        before = original

    return {
        "task": task,
        "scale": scale,
        "engine": engine,
        "before": _img_to_b64(before),
        "after": _img_to_b64(result),
        "note": (
            "Classical baseline (train & export real weights to upgrade to ML)."
            if engine == "classical"
            else "Powered by a trained PyTorch model."
        ),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
