"""End-to-end inference test with the *real* committed weights.

The smoke test (`test_api_smoke.py`) deliberately exercises the classical
fallback path so it can run anywhere without torch or weights. That leaves a
blind spot: nothing asserted that the shipped ``serve/models/*.pt`` files
actually load and produce correctly-shaped output through the full service
stack. A weight that is corrupt, exported for the wrong torch version, or
wired to the wrong task would keep CI green while the deployed demo silently
degraded to the classical baseline.

This module closes that gap. It loads the committed TorchScript weights
through the real ``/api/predict`` route and asserts the ``ml`` engine is the
one that answered, with output dimensions matching the documented contract:

  * SR  x4  : a low-res input is upscaled 4x by the model.
  * Low-light: output keeps the input resolution.

It runs on CPU (no GPU, no network) and skips cleanly when torch or the
weights are unavailable, so it is safe in any environment.

Run:
    pytest tests/test_inference_integration.py
"""

from __future__ import annotations

import base64
import io

from fastapi.testclient import TestClient
from PIL import Image
import pytest

import serve.app as serve_app
from serve.app import app
from serve import model_loader

client = TestClient(app)

torch = pytest.importorskip("torch", reason="torch not installed")


@pytest.fixture(autouse=True)
def _relax_rate_limit(monkeypatch):
    """Keep the per-IP limiter out of the way (mirrors the smoke test)."""
    monkeypatch.setattr(serve_app, "RATE_LIMIT_CAPACITY", 10_000)
    monkeypatch.setattr(serve_app, "RATE_LIMIT_REFILL_PER_SEC", 10_000.0)
    serve_app._rate_buckets.clear()
    yield
    serve_app._rate_buckets.clear()


@pytest.fixture(autouse=True)
def _fresh_model_cache():
    """Drop cached models around each test so we exercise real loading.

    The loader caches a model (including a cached ``None`` for a failed load)
    for the process lifetime. Clearing the cache here means these tests do not
    accidentally pass on a branch where the weight never loads.
    """
    model_loader._sr_cache.clear()
    model_loader._lowlight_model = model_loader._LOWLIGHT_UNSET
    yield
    model_loader._sr_cache.clear()
    model_loader._lowlight_model = model_loader._LOWLIGHT_UNSET


def _png_bytes(size=(64, 64), color=(100, 110, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _decode_png(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    assert raw.startswith(b"\x89PNG\r\n\x1a\n"), "response is not a PNG"
    return Image.open(io.BytesIO(raw))


def _require_weight(scale_or_task: str):
    """Skip the test when the weight it depends on is not committed."""
    if scale_or_task == "sr4":
        p = model_loader.MODELS_DIR / "sr_generator_scale4.pt"
    elif scale_or_task == "sr2":
        p = model_loader.MODELS_DIR / "sr_generator_scale2.pt"
    else:
        p = model_loader.MODELS_DIR / "lowlight.pt"
    if not p.exists():
        pytest.skip(f"{p.name} not present; real-weight test not applicable")


def test_sr_scale4_uses_ml_engine_and_upscales_4x():
    """SR x4 must load the real weight and return a 4x-upscaled PNG."""
    _require_weight("sr4")
    # 64x64 -> the model sees 16x16 (64//4) and outputs 64x64.
    resp = client.post(
        "/api/predict",
        files={"image": ("in.png", _png_bytes((64, 64)), "image/png")},
        data={"task": "sr", "scale": "4"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["engine"] == "ml", (
        "committed SR x4 weight did not load; service fell back to classical"
    )
    out = _decode_png(body["after"])
    # Model's true output is lr_size * 4 = (64//4) * 4 = 64.
    assert out.size == (64, 64), f"unexpected SR output size {out.size}"


def test_sr_scale2_uses_ml_engine_and_keeps_resolution():
    """SR x2 ships a committed weight (sr_generator_scale2.pt); the service
    must load it and report `ml` as the engine, same size out."""
    _require_weight("sr2")
    resp = client.post(
        "/api/predict",
        files={"image": ("in.png", _png_bytes((64, 64)), "image/png")},
        data={"task": "sr", "scale": "2"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["engine"] == "ml", (
        "committed SR x2 weight did not load; service fell back to classical"
    )
    # Model's true output is lr_size * 2 = (64//2) * 2 = 64.
    out = _decode_png(resp.json()["after"])
    assert out.size == (64, 64), f"unexpected SR x2 output size {out.size}"


def test_lowlight_uses_ml_engine_and_keeps_resolution():
    """Low-light must load the real weight on a dark input, same size out."""
    _require_weight("lowlight")
    # A genuinely dark image so the input gate lets it through to the model.
    resp = client.post(
        "/api/predict",
        files={"image": ("dark.png", _png_bytes((64, 64), (12, 12, 14)), "image/png")},
        data={"task": "lowlight", "scale": "2"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["engine"] == "ml", (
        "committed low-light weight did not load; service fell back to classical"
    )
    out = _decode_png(body["after"])
    assert out.size == (64, 64), f"low-light changed resolution: {out.size}"
