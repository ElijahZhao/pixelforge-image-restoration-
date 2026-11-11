"""Regression tests for the service-layer guards.

Each test here pins a defect that was found by reading the code and confirmed
by running it, so the same bug cannot come back quietly:

1. ``lowlight_classical`` divided by zero on a pure-white image
   (``log(1.0) == 0``), and the clip turned the resulting ``-inf`` into
   ``0.3`` -- the opposite of the intended "leave a well-lit image alone".
2. The per-IP rate limiter was bypassable by rotating the ``X-Forwarded-For``
   header, because the header was trusted from any client.
3. The 10 MB upload cap was only enforced AFTER the whole body had been
   buffered, so an oversized request was still fully downloaded.

Run:
    pytest tests/test_api_guards.py
"""

from __future__ import annotations

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
import pytest

import serve.app as serve_app
from serve.app import app
from serve.classical import lowlight_classical


@pytest.fixture(autouse=True)
def _clean_rate_limit(monkeypatch):
    """Reset the limiter around every test, and keep it out of the way of the
    functional ones (the limiter has its own test below)."""
    monkeypatch.setattr(serve_app, "RATE_LIMIT_CAPACITY", 10_000)
    monkeypatch.setattr(serve_app, "RATE_LIMIT_REFILL_PER_SEC", 10_000.0)
    # `raising=False` so this file still runs (and FAILS with a clear
    # assertion, not a fixture error) against a build that predates the
    # proxy-trust fix.
    monkeypatch.setattr(serve_app, "TRUSTED_PROXY_IPS", set(), raising=False)
    serve_app._rate_buckets.clear()
    yield
    serve_app._rate_buckets.clear()


def _png(size=(32, 32), color=(128, 128, 128)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


client = TestClient(app)


# --------------------------------------------------------------------------- #
# 1. The white-image divide-by-zero
# --------------------------------------------------------------------------- #

def test_lowlight_classical_white_image_does_not_divide_by_zero():
    """A pure-white image must not raise or warn inside the gamma formula."""
    white = Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))
    with np.errstate(all="raise"):
        out = lowlight_classical(white)
    arr = np.asarray(out, dtype=np.float32)
    # gamma must be 1.0 ("leave it alone"), so the pixels stay at 255.
    assert arr.min() == 255.0, "a white image must stay white"


def test_lowlight_classical_keeps_dark_images_brightened():
    """The dark branch must still brighten -- the guard must not disable it."""
    dark = Image.fromarray((np.random.rand(32, 32, 3) * 20).astype(np.uint8))
    out = np.asarray(lowlight_classical(dark), dtype=np.float32)
    assert out.mean() > 30.0, "a dark image must still be brightened"


# --------------------------------------------------------------------------- #
# 2. The rate limiter must not be switchable off by the client
# --------------------------------------------------------------------------- #

def test_forwarded_for_header_cannot_reset_the_bucket():
    """Rotating X-Forwarded-For must not hand the client a fresh bucket."""
    serve_app.RATE_LIMIT_CAPACITY = 5
    serve_app.RATE_LIMIT_REFILL_PER_SEC = 0.0
    data = _png()
    codes = [
        client.post("/api/predict",
                    files={"image": ("a.png", data, "image/png")},
                    data={"task": "sr", "scale": "4"},
                    headers={"X-Forwarded-For": f"10.0.0.{i}"}).status_code
        for i in range(12)
    ]  # noqa: B018 - the status code is the value under test
    assert 429 in codes, "spoofed X-Forwarded-For bypassed the rate limiter"


def test_forwarded_for_header_is_honoured_behind_a_trusted_proxy(monkeypatch):
    """With the peer listed as a trusted proxy, the header is the client IP."""
    monkeypatch.setattr(serve_app, "TRUSTED_PROXY_IPS", {"testclient"}, raising=False)
    serve_app.RATE_LIMIT_CAPACITY = 2
    serve_app.RATE_LIMIT_REFILL_PER_SEC = 0.0
    data = _png()
    # One IP exhausts its bucket...
    for _ in range(2):
        _ = client.post("/api/predict",
                        files={"image": ("a.png", data, "image/png")},
                        data={"task": "sr", "scale": "4"},
                        headers={"X-Forwarded-For": "1.2.3.4"}).status_code
    r = client.post("/api/predict",
                    files={"image": ("a.png", data, "image/png")},
                    data={"task": "sr", "scale": "4"},
                    headers={"X-Forwarded-For": "1.2.3.4"})
    assert r.status_code == 429, "the proxy-supplied IP was not used for limiting"
    # ...while a different client IP still gets its own bucket.
    r2 = client.post("/api/predict",
                     files={"image": ("a.png", data, "image/png")},
                     data={"task": "sr", "scale": "4"},
                     headers={"X-Forwarded-For": "5.6.7.8"})
    assert r2.status_code == 200


def test_rate_limit_buckets_do_not_grow_without_bound():
    """Many distinct client IPs must not pin unbounded memory."""
    serve_app.RATE_LIMIT_CAPACITY = 1
    serve_app.RATE_LIMIT_REFILL_PER_SEC = 0.0
    serve_app._MAX_BUCKETS = 50
    data = _png()
    for i in range(120):
        _ = client.post("/api/predict",
                        files={"image": ("a.png", data, "image/png")},
                        data={"task": "sr", "scale": "4"},
                        headers={"X-Forwarded-For": f"172.16.0.{i}"}).status_code
    assert len(serve_app._rate_buckets) <= serve_app._MAX_BUCKETS

# --------------------------------------------------------------------------- #
# 3. The upload cap must be enforced before the body is buffered
# --------------------------------------------------------------------------- #

def test_oversized_content_length_is_rejected_before_reading(monkeypatch):
    """A declared oversize body must be refused without being downloaded."""
    monkeypatch.setattr(serve_app, "MAX_UPLOAD_BYTES", 1024)

    def _boom(*a, **kw):  # pragma: no cover - must never run
        raise AssertionError("the body was read before the size check")

    monkeypatch.setattr(serve_app, "_predict_sync", _boom)
    r = client.post("/api/predict",
                    files={"image": ("a.png", _png(), "image/png")},
                    data={"task": "sr", "scale": "4"},
                    headers={"Content-Length": str(50 * 1024 * 1024)})
    assert r.status_code == 413
    assert "too large" in r.json()["error"]


def test_actual_oversize_body_is_still_rejected(monkeypatch):
    """The post-read byte check stays as the backstop for chunked uploads."""
    monkeypatch.setattr(serve_app, "MAX_UPLOAD_BYTES", 1024)
    payload = _png() + b"\x00" * 4096
    r = client.post("/api/predict",
                    files={"image": ("a.png", payload, "image/png")},
                    data={"task": "sr", "scale": "4"})
    assert r.status_code == 413
