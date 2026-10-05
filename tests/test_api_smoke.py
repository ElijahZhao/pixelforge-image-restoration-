"""API smoke test for the FastAPI inference service.

Uses FastAPI's in-process ``TestClient``: no live server, no network, no
browser. It exercises *every* route and the resource-guard branches so a
regression in the service layer fails CI before deployment.

The service transparently falls back to classical baselines when no trained
weight is present, so this test runs fully offline (no ImageNet / VGG
downloads, no GPU).

Run:
    pytest tests/test_api_smoke.py
"""

from __future__ import annotations

import base64
import io

from fastapi.testclient import TestClient
from PIL import Image
import pytest

import serve.app as serve_app
from serve.app import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _relax_rate_limit(monkeypatch):
    """Keep the per-IP limiter out of the way for the functional tests.

    The smoke tests fire several requests from the same test client; without
    this they could trip the bucket as more cases are added. The limiter has
    its own dedicated test (`test_predict_rate_limited`) that patches the
    capacity down instead, so this fixture does not hide a regression.
    """
    monkeypatch.setattr(serve_app, "RATE_LIMIT_CAPACITY", 10_000)
    monkeypatch.setattr(serve_app, "RATE_LIMIT_REFILL_PER_SEC", 10_000.0)
    serve_app._rate_buckets.clear()
    yield
    serve_app._rate_buckets.clear()


def _is_png_b64(s: str) -> bool:
    """The API returns RAW base64 (the frontend prepends the `data:` URI
    prefix when rendering). Verify it decodes to a real PNG."""
    try:
        return base64.b64decode(s).startswith(b"\x89PNG\r\n\x1a\n")
    except Exception:
        return False


def _png(size=(32, 32), color=(128, 128, 128)) -> bytes:
    """Return a small valid PNG as bytes."""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["engines"]) == {"sr_scale2", "sr_scale4", "lowlight"}
    # Every engine slot must report a concrete mode, never null/unknown.
    assert all(v in ("ml", "classical") for v in body["engines"].values())


def test_predict_sr_scale2():
    r = client.post(
        "/api/predict",
        files={"image": ("scene.png", _png(), "image/png")},
        data={"task": "sr", "scale": "2"},
    )
    assert r.status_code == 200
    b = r.json()
    assert b["task"] == "sr"
    assert b["scale"] == 2
    assert b["engine"] in ("ml", "classical")
    # before/after are valid raw base64 PNGs (3-panel frontends also get `lr`).
    assert _is_png_b64(b["before"])
    assert _is_png_b64(b["after"])
    assert "lr" in b and _is_png_b64(b["lr"])


def test_predict_sr_scale4():
    r = client.post(
        "/api/predict",
        files={"image": ("scene.png", _png(), "image/png")},
        data={"task": "sr", "scale": "4"},
    )
    assert r.status_code == 200
    b = r.json()
    assert b["scale"] == 4
    assert _is_png_b64(b["before"])
    assert _is_png_b64(b["after"])


def test_predict_lowlight():
    # A genuinely dark image so the low-light path is actually exercised.
    r = client.post(
        "/api/predict",
        files={"image": ("dark.png", _png(color=(20, 20, 20)), "image/png")},
        data={"task": "lowlight", "scale": "2"},
    )
    assert r.status_code == 200
    b = r.json()
    assert b["task"] == "lowlight"
    assert b["engine"] in ("ml", "classical")
    assert _is_png_b64(b["before"])
    assert _is_png_b64(b["after"])


def test_predict_rejects_bad_task():
    r = client.post(
        "/api/predict",
        files={"image": ("x.png", _png(), "image/png")},
        data={"task": "bogus"},
    )
    assert r.status_code == 422


def test_predict_rejects_non_image():
    r = client.post(
        "/api/predict",
        files={"image": ("x.txt", b"this is not an image", "text/plain")},
        data={"task": "sr"},
    )
    assert r.status_code == 422


def test_predict_rejects_oversized_upload(monkeypatch):
    # Byte-size guard: > MAX_UPLOAD_BYTES (default 10 MB) is rejected before
    # decode. We lower the cap so the test does not ship an 11 MB payload.
    monkeypatch.setattr("serve.app.MAX_UPLOAD_BYTES", 1024)
    big = b"\x00" * 2048
    r = client.post(
        "/api/predict",
        files={"image": ("big.bin", big, "application/octet-stream")},
        data={"task": "sr"},
    )
    assert r.status_code == 413


def test_predict_rejects_oversized_dimensions(monkeypatch):
    # Pixel-count guard: a header declaring far more pixels than
    # the cap is rejected without decoding. Lower the cap so a normal 32x32
    # image trips it; the real default is 4 MP.
    monkeypatch.setattr("serve.app.MAX_INPUT_PIXELS", 100)
    r = client.post(
        "/api/predict",
        files={"image": ("scene.png", _png((32, 32)), "image/png")},
        data={"task": "sr"},
    )
    assert r.status_code == 413


def test_predict_rate_limited(monkeypatch):
    # The `_relax_rate_limit` fixture normally raises the ceiling to keep the
    # functional tests independent. Here we do the opposite: shrink the bucket
    # to a single token and no refill, so the second request must be refused.
    monkeypatch.setattr(serve_app, "RATE_LIMIT_CAPACITY", 1)
    monkeypatch.setattr(serve_app, "RATE_LIMIT_REFILL_PER_SEC", 0.0)
    serve_app._rate_buckets.clear()

    payload = {"files": {"image": ("a.png", _png((16, 16)), "image/png")},
               "data": {"task": "sr"}}
    first = client.post("/api/predict", **payload)
    assert first.status_code != 429  # first request spends the only token
    second = client.post("/api/predict", **payload)
    assert second.status_code == 429
    assert second.headers.get("Retry-After") == "2"
