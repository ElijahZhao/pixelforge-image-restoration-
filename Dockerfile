# Inference service image for PixelForge (serve/).
#
# Scope note: this image runs the FastAPI backend only. The Next.js frontend
# in web/ is a separate build target - it deploys as static/Node output
# (Vercel or its own container) and talks to this service over HTTP. Bundling
# both into one image would drag a Node toolchain into an image that only
# needs Python, for no benefit.
#
# Build:
#   docker build -t pixelforge-serve .
# Run:
#   docker run -p 8000:8000 pixelforge-serve
#   docker run -p 8000:8000 -e ALLOWED_ORIGINS=https://your-frontend pixelforge-serve

FROM python:3.11-slim

# torch's CPU wheels pull in libgomp at import time; without it the process
# dies on `import torch` with a cryptic shared-object error. Everything else
# is self-contained in the wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so the layer is cached across code edits. We install the
# lock file (fully pinned) rather than requirements.txt, so the image matches
# what CI audits with pip-audit.
COPY requirements.lock.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.lock.txt

# Bake the trained weights into the image: serve/model_loader.py looks for
# them under serve/models/ and falls back to the classical baseline when they
# are missing. Shipping them means the container starts in ML mode, not
# fallback mode.
COPY serve/ ./serve/
COPY train/ ./train/

# Run as a non-root user. The service only reads weights and writes nothing
# to disk, so it needs no ownership of the app tree.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Healthcheck hits the endpoint the app already exposes, so an orchestrator
# can tell "process up" from "process actually serving".
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"

CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
