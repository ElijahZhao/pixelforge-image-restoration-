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
import os

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .classical import run_classical
from .model_loader import (
    get_sr_model,
    get_lowlight_model,
    predict_sr,
    predict_lowlight,
    looks_underexposed,
)

app = FastAPI(title="CV Restoration API", version="1.0.0")

# ---------------------------------------------------------------------------
# Resource limits (F13 — see DIAGNOSIS_ROUND6/14). Without these, a single large
# upload can produce a multi-GB base64 response (SR x4 grows output linearly)
# and OOM the host. All limits are overridable via env vars.
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))   # 10 MB
MAX_INPUT_PIXELS = int(os.getenv("MAX_INPUT_PIXELS", str(4_000_000)))          # ~4 MP

# Production: set ALLOWED_ORIGINS=https://your-domain.com,https://admin.your-domain.com
# Multiple origins can be comma-separated. Defaults to wildcard for local dev.
_ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _ALLOWED_ORIGINS.split(",") if o.strip()]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
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

    # F13: reject oversized uploads before decoding them.
    raw = await image.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error": f"upload too large (>{MAX_UPLOAD_BYTES} bytes)"},
        )

    try:
        original = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        # Catches malformed images AND PIL DecompressionBombError (the default
        # MAX_IMAGE_PIXELS ~89M guard). We surface it as 422 rather than 500;
        # the 4MP cap above is the first line of defence, this is the backstop.
        return JSONResponse(
            status_code=422, content={"error": "uploaded file is not a valid image"}
        )

    # F13: bound the work by input pixel count.
    if original.width * original.height > MAX_INPUT_PIXELS:
        return JSONResponse(
            status_code=413,
            content={"error": f"image too large (>{MAX_INPUT_PIXELS} pixels); "
                              f"downscale it first."},
        )

    lr_display = None
    if task == "sr":
        ml_result = predict_sr(original, scale)
        engine = "ml" if ml_result is not None else "classical"
        result = ml_result if ml_result is not None else run_classical(original, task, scale)
        # "before" = the same low-res input upscaled by the classical bicubic
        # baseline, so before/after share one resolution and one reference frame
        # (the model's true output is lr*scale; see DIAGNOSIS_ROUND10/16/18).
        lr = original.resize((max(1, original.width // scale),
                              max(1, original.height // scale)), Image.BICUBIC)
        before = lr.resize((lr.width * scale, lr.height * scale), Image.BICUBIC)
        # Extra field for the 3-panel frontends: the model's ACTUAL low-res input
        # (nearest-upscaled to `before`'s size for display). This makes the demo
        # honest about "SR maps a low-res image to a high-res one" and avoids the
        # misleading "original vs reconstruction" comparison (a full-res upload
        # can never be beaten by a model that only sees lr=input/scale).
        lr_display = lr.resize(before.size, Image.NEAREST)
        # The classical SR baseline returns orig*scale, while the ML output and
        # `before` are ~orig. Force `after` to `before`'s size so the comparison
        # slider stays pixel-aligned. This is a display-only resize, NOT a hidden
        # upscale — in the ML path the sizes already match (no-op).
        if result.size != before.size:
            result = result.resize(before.size, Image.BICUBIC)
    else:  # lowlight
        is_dark, _stats = looks_underexposed(original)
        ml_result = predict_lowlight(original) if is_dark else None
        engine = "ml" if ml_result is not None else "classical"
        result = ml_result if ml_result is not None else run_classical(original, task, scale)
        before = original

    # F8: when no trained weight exists for the chosen task/scale, say so
    # explicitly instead of silently falling back to a classical baseline.
    if engine == "classical" and task == "lowlight":
        note = ("Classical adaptive-gamma baseline. The self-trained low-light "
                "U-Net is trained on LOL-v1 (real night PHOTOS); on a bright or "
                "synthetic image it crushes shadows and DARKENS the picture "
                "(measured: median luminance 0.26 -> 0.06), so it was skipped "
                "here. Upload a genuinely dark photo to exercise the model.")
    elif engine == "classical":
        note = ("Classical baseline in use — no trained weight found for "
                f"{task} x{scale} (train & export one to switch to ML).")
    elif task == "sr":
        note = ("Super-resolution maps a LOW-RES input to a high-res output. "
                f"For a fair comparison, the middle panel shows the model's true "
                f"input (original downscaled by x{scale}). A model that only sees "
                f"1/{scale**2} of the pixels cannot out-detail a full-res original.")
    else:
        note = "Powered by a trained PyTorch model."

    payload = {
        "task": task,
        "scale": scale,
        "engine": engine,
        "before": _img_to_b64(before),
        "after": _img_to_b64(result),
        "note": note,
    }
    # 3-panel frontends use `lr` (the model's honest input); 2-panel ones ignore it.
    if lr_display is not None:
        payload["lr"] = _img_to_b64(lr_display)
    return payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
