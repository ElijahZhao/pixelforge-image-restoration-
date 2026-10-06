<p align="center">
  <img src="assets/banner.svg" alt="PixelForge" width="100%"/>
</p>

<p align="center">
  <a href="https://github.com/ElijahZhao/pixelforge-image-restoration-/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ElijahZhao/pixelforge-image-restoration-?style=flat-square" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <a href="https://pixelforge-image-restoration.streamlit.app/"><img src="https://img.shields.io/badge/Live%20Demo-PixelForge-9b59b6?style=flat-square&logo=streamlit&logoColor=white" alt="Live Demo"></a>
  <a href="https://github.com/ElijahZhao/pixelforge-image-restoration-/actions/workflows/ci.yml"><img src="https://github.com/ElijahZhao/pixelforge-image-restoration-/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

<p align="center">
  <b>PixelForge</b> is an end-to-end image restoration project:<br/>
  self-trained PyTorch models for single-image <b>super-resolution</b> and <b>low-light enhancement</b>,
  served through a small web frontend.
</p>

> **Live demo:** <https://pixelforge-image-restoration.streamlit.app/>
> Runs the self-trained weights for both tasks (not the classical fallback).
> The Chinese README ([README.md](README.md)) is the primary document and carries
> more detail; this file is the English entry point.

---

## Results

Per-epoch metrics are committed under [`results/`](results/) (`train_log_sr_generator.csv`, `train_log_lowlight_srcnn.csv`); anyone can recompute them with `scripts/eval_baseline.py`.

| Task | Baseline | Ours (retrained) | Gain |
|---|---|---|---|
| **Super-resolution ×4** | Bicubic: PSNR 26.69 / SSIM 0.754 | **PSNR 27.47 / SSIM 0.780** | **+0.77 dB / +0.026** |
| **Low-light enhancement** | No-op: PSNR 7.77 / SSIM 0.192 | **PSNR 18.18 / SSIM 0.739** | **+10.41 dB / +0.547** |

Both rows are measured by [`scripts/eval_baseline.py`](scripts/eval_baseline.py) on the
same validation set with the same PSNR/SSIM implementation, so the only variable is
the method. Per-epoch logs are committed under [`results/`](results/) and can be recomputed.

What this does **not** claim: SOTA. SR ×4 at 27.47 dB is above the bicubic baseline but
below large-dataset academic SOTA (30+ dB) - the training set and model capacity are small.
The low-light gain is relative to a no-op baseline, not to other published methods.

`SRCNN ×2` has no trained weight (`TBD`); a 2× request is served by the classical
bicubic path, not a model.

---

## Architecture

```
data/          datasets (DIV2K / LOL), not committed
train/         models, dataset loaders, PSNR/SSIM metrics, train + export scripts
serve/         FastAPI inference service + classical fallbacks + TorchScript weights
deploy/        Streamlit demo and an HF Spaces alternative
web/           Next.js 14 + Tailwind frontend (upload + before/after slider)
scripts/       evaluation, demo image generation, data setup
docs/          API + operations docs, development history, retrain evidence
```

Request flow: `web/` → `POST /api/predict` → `serve/app.py` picks the ML engine if a
weight is loaded for that task/scale, otherwise a classical baseline (bicubic + unsharp
for SR, adaptive gamma for low-light). `GET /api/health` reports which engine each slot
is using. See [`docs/API.md`](docs/API.md) for the full interface.

---

## Quick start

**Backend**

```bash
pip install -r requirements.txt
uvicorn serve.app:app --reload --port 8000
```

**Frontend** (separate terminal)

```bash
cd web
pnpm install
pnpm dev
```

Open http://localhost:3000, upload an image, pick a task, hit **Enhance**, drag the
slider. No GPU needed - the repo ships trained weights under `serve/models/`, and the
service switches to them automatically once they are present.

**Docker** (backend only)

```bash
docker build -t pixelforge-serve .
docker run -p 8000:8000 pixelforge-serve
```

The image bundles the trained weights and starts in ML mode. The frontend is a separate
build target and is not part of this image.

---

## Tests

```bash
python -m pytest --cov --cov-report=term-missing   # 29 passed, coverage gate 75%
cd web && pnpm test                                # frontend unit tests
cd web && pnpm exec tsc --noEmit                   # type-check
```

The E2E browser test (`tests/e2e/test_e2e.py`) drives a real Chromium through the full
upload flow. It boots two servers and a browser, so it is skipped under pytest by default;
run it directly, or set `PIXELFORGE_RUN_E2E=1`.

CI (`.github/workflows/ci.yml`) runs three jobs: pytest with a coverage gate, `pip-audit`
over the lock file, and the frontend type-check + tests + build.

---

## Training

Requires a GPU. Prepare data first (builds directories and prints download URLs; does not
fetch the multi-GB datasets):

```bash
bash scripts/download_data.sh
```

```bash
# Super-resolution (4×, with perceptual loss)
python train/train.py --task sr --model generator --scale 4 \
    --data_root data --epochs 200 --batch_size 8 --lr 1e-4 --perceptual

# Low-light (U-Net)
python train/train.py --task lowlight --data_root data \
    --epochs 200 --batch_size 8 --lr 2e-4

# Export to TorchScript for the service
python train/export.py --checkpoint models/sr_generator_scale4_best.pth \
    --out serve/models/sr_generator_scale4.pt --task sr --scale 4
```

Drop the exported `.pt` into `serve/models/` and the service picks it up on restart.

---

## Development notes

This project was built and driven by the author: environment setup, retraining,
result verification and iteration decisions are the author's responsibility.
Per-epoch training logs (`results/`) and the same-caliber evaluation script
(`scripts/eval_baseline.py`) make the reported metrics reproducible.

---

## License

MIT - see [LICENSE](LICENSE). Citation info in [CITATION.cff](CITATION.cff).
