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
import time

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
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
    mean_luminance,
    LOWLIGHT_MIN_GAIN,
)

app = FastAPI(title="CV Restoration API", version="1.1.0")

# ---------------------------------------------------------------------------
# Resource limits. Without these, a single large
# upload can produce a multi-GB base64 response (SR x4 grows output linearly)
# and OOM the host. All limits are overridable via env vars.
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))   # 10 MB
MAX_INPUT_PIXELS = int(os.getenv("MAX_INPUT_PIXELS", str(4_000_000)))          # ~4 MP

# Align PIL's own decompression-bomb ceiling with our pixel cap, so an image
# whose header understates its real dimensions still trips PIL's guard rather
# than allocating unbounded memory on decode. PIL warns at *this* value and
# raises at 2x it.
Image.MAX_IMAGE_PIXELS = MAX_INPUT_PIXELS

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


# ---------------------------------------------------------------------------
# Rate limiting. Inference is CPU-bound and can run for seconds, so a client
# that hammers /api/predict can starve every other request. There is no auth
# on this service, so a per-IP token bucket is the cheapest way to bound that.
#
# Deliberately in-process and in-memory: this is a single-instance demo, and
# pulling in a Redis-backed limiter would add a dependency and an ops concern
# for no real gain here. If the service is ever scaled to multiple instances,
# this limiter must move to a shared store or it will over-admit by N.
#
# Defaults: 30 requests burst, refilled at 0.5/s (~30/min). Override via env.
# ---------------------------------------------------------------------------
RATE_LIMIT_CAPACITY = int(os.getenv("RATE_LIMIT_CAPACITY", "30"))
RATE_LIMIT_REFILL_PER_SEC = float(os.getenv("RATE_LIMIT_REFILL_PER_SEC", "0.5"))

_rate_buckets: dict[str, tuple[float, float]] = {}  # ip -> (tokens, last_ts)


def _client_ip(request) -> str:
    # Trust X-Forwarded-For's first hop only when present (we may run behind a
    # proxy); otherwise fall back to the socket peer.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_limited(ip: str) -> bool:
    """Return True if this IP is over budget. Cheap token-bucket refill."""
    now = time.monotonic()
    tokens, last = _rate_buckets.get(ip, (float(RATE_LIMIT_CAPACITY), now))
    tokens = min(float(RATE_LIMIT_CAPACITY), tokens + (now - last) * RATE_LIMIT_REFILL_PER_SEC)
    if tokens < 1.0:
        _rate_buckets[ip] = (tokens, now)
        return True
    _rate_buckets[ip] = (tokens - 1.0, now)
    return False


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
    request: Request,
    image: UploadFile = File(...),
    task: str = Form("sr"),
    scale: int = Form(2),
):
    # Throttle before doing any work: this endpoint is unauthenticated and the
    # body read below is already cheap to abuse.
    if _rate_limited(_client_ip(request)):
        return JSONResponse(
            status_code=429,
            content={"error": "too many requests; slow down"},
            headers={"Retry-After": "2"},
        )

    # This handler is `async` only so it can `await image.read()`.
    # The inference below is CPU-bound and can take seconds; running it directly
    # in the event loop would stall every other request (including /api/health).
    # We therefore hand the heavy work to a worker thread via run_in_threadpool.
    return await run_in_threadpool(
        _predict_sync,
        await image.read(),
        task,
        scale,
    )


def _predict_sync(raw: bytes, task: str, scale: int):
    """Blocking inference body; runs in a worker thread (see `predict`)."""
    if task not in ("sr", "lowlight"):
        return JSONResponse(
            status_code=422, content={"error": "task must be 'sr' or 'lowlight'"}
        )
    if scale not in (2, 4):
        scale = 2

    # Reject oversized uploads before decoding them.
    if len(raw) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error": f"upload too large (>{MAX_UPLOAD_BYTES} bytes)"},
        )

    # Validate the PIXEL COUNT from the header *before* decoding.
    # A byte-size cap alone does not bound memory: a 61 KB PNG can declare
    # 8000x8000 = 64 MP, and `.convert("RGB")` allocates ~385 MB to decode it
    # (measured). Repeating that request is a trivial OOM/DoS. PIL exposes
    # `.size` from the header without decoding, so we reject on dimensions
    # first and only then pay for the decode. `Image.MAX_IMAGE_PIXELS` (set to
    # our cap above) makes PIL raise during `open()` for oversized headers;
    # we surface that as 413 so the message matches the real reason.
    try:
        probe = Image.open(io.BytesIO(raw))
        pw, ph = probe.size
    except Image.DecompressionBombError:
        return JSONResponse(
            status_code=413,
            content={"error": f"image too large (limit {MAX_INPUT_PIXELS} pixels); "
                              f"downscale it first."},
        )
    except Exception:
        return JSONResponse(
            status_code=422, content={"error": "uploaded file is not a valid image"}
        )
    if pw * ph > MAX_INPUT_PIXELS:
        return JSONResponse(
            status_code=413,
            content={"error": f"image too large ({pw}x{ph} = {pw * ph} pixels; "
                              f"limit {MAX_INPUT_PIXELS}); downscale it first."},
        )

    try:
        original = probe.convert("RGB")
    except Image.DecompressionBombError:
        # Header understated the true size; PIL trips during decode.
        return JSONResponse(
            status_code=413,
            content={"error": f"image too large (limit {MAX_INPUT_PIXELS} pixels); "
                              f"downscale it first."},
        )
    except Exception:
        # Backstop for any other decode failure. 422 rather than 500.
        return JSONResponse(
            status_code=422, content={"error": "uploaded file is not a valid image"}
        )

    lr_display = None
    if task == "sr":
        ml_result = predict_sr(original, scale)
        engine = "ml" if ml_result is not None else "classical"
        result = ml_result if ml_result is not None else run_classical(original, task, scale)
        # "before" = the same low-res input upscaled by the classical bicubic
        # baseline, so before/after share one resolution and one reference frame
        # (the model's true output is lr*scale).
        lr = original.resize((max(1, original.width // scale),
                              max(1, original.height // scale)), Image.BICUBIC)
        before = lr.resize((lr.width * scale, lr.height * scale), Image.BICUBIC)
        # Extra field for the 3-panel frontends: the model's ACTUAL low-res input
        # (nearest-upscaled to `before`'s size for display). This keeps the demo
        # aligned with what SR really does ("map a low-res image to a high-res
        # one") and avoids the misleading "original vs reconstruction"
        # comparison (a full-res upload can never be beaten by a model that only
        # sees lr=input/scale).
        lr_display = lr.resize(before.size, Image.NEAREST)
        # The classical SR baseline returns orig*scale, while the ML output and
        # `before` are ~orig. Force `after` to `before`'s size so the comparison
        # slider stays pixel-aligned. This is a display-only resize, NOT a hidden
        # upscale; in the ML path the sizes already match (no-op).
        if result.size != before.size:
            result = result.resize(before.size, Image.BICUBIC)
    else:  # lowlight
        is_dark, _stats = looks_underexposed(original)
        ml_result = predict_lowlight(original) if is_dark else None
        guard_rejected = False
        if ml_result is not None:
            # OUTPUT guard: an enhancement that DARKENS its input is wrong by
            # definition -> discard it. Catches dark synthetic art, which passes
            # the input gate because it genuinely is dark.
            gain = mean_luminance(ml_result) - mean_luminance(original)
            if gain < LOWLIGHT_MIN_GAIN:
                ml_result = None
                guard_rejected = True
        engine = "ml" if ml_result is not None else "classical"
        result = ml_result if ml_result is not None else run_classical(original, task, scale)
        before = original

    # When no trained weight exists for the chosen task/scale, say so
    # explicitly instead of silently falling back to a classical baseline.
    if engine == "classical" and task == "lowlight" and guard_rejected:
        note = ("Classical adaptive-gamma baseline. The self-trained low-light "
                "U-Net was tried and DISCARDED because its output was DARKER "
                "than the input -- an 'enhancement' that darkens is wrong by "
                "definition. This happens when the image is dark but is not an "
                "underexposed photograph (e.g. night-scene artwork or a game "
                "screenshot): the model is trained on LOL-v1 real night photos "
                "and crushes shadows outside that domain.")
    elif engine == "classical" and task == "lowlight":
        note = ("Classical adaptive-gamma baseline. This image is not a low-light "
                "PHOTO, so the self-trained low-light U-Net was skipped: it is "
                "trained on LOL-v1 (real night photographs) and only helps a "
                "genuinely underexposed photo. The image is either already "
                "well-exposed, or it is a screenshot / synthetic image rather "
                "than a photograph. Running the model anyway would crush the "
                "shadows and DARKEN the picture (measured: median luminance "
                "0.26 -> 0.06). Upload a genuinely dark photo to exercise it.")
    elif engine == "classical":
        note = ("Classical baseline in use: no trained weight found for "
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
    # 3-panel frontends use `lr` (the model's actual input); 2-panel ones ignore it.
    if lr_display is not None:
        payload["lr"] = _img_to_b64(lr_display)
    return payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
