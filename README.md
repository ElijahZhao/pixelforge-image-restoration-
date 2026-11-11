<p align="center">
  <img src="assets/banner.svg" alt="PixelForge" width="100%"/>
</p>

<p align="center">
  <a href="https://pixelforge-image-restoration.streamlit.app/"><img src="https://img.shields.io/badge/Live%20Demo-Try%20it-9b59b6?style=flat-square&logo=streamlit&logoColor=white" alt="Live Demo"></a>
  <a href="https://doi.org/10.20944/preprints202611.0550.v1"><img src="https://img.shields.io/badge/DOI-10.20944%2Fpreprints202611.0550.v1-1f6feb?style=flat-square" alt="DOI"></a>
  <a href="https://github.com/ElijahZhao/pixelforge-image-restoration-/actions/workflows/ci.yml"><img src="https://github.com/ElijahZhao/pixelforge-image-restoration-/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <a href="LICENSE"><img src="https://img.shields.io/github/license/ElijahZhao/pixelforge-image-restoration-?style=flat-square" alt="License"></a>
</p>

<p align="center">
  <b>PixelForge</b> — an end-to-end image-restoration pipeline you can clone and run.<br/>
  Three self-trained PyTorch models for ×2/×4 super-resolution and low-light enhancement,
  built from data to training, evaluation and deployment. No external inference API.
</p>

<p align="center">
  <a href="https://pixelforge-image-restoration.streamlit.app/"><b>Live demo</b></a> &nbsp;·&nbsp;
  <a href="#-quick-start">Quick start</a> &nbsp;·&nbsp;
  <a href="#-results">Results</a> &nbsp;·&nbsp;
  <a href="#-paper--citation">Paper</a> &nbsp;·&nbsp;
  <a href="README.zh-CN.md">中文</a>
</p>

---

## 🎯 What it is

A complete, reproducible vision pipeline written from scratch:

```
data → training → evaluation → export (TorchScript) → FastAPI service → web UI
```

The three inference weights ship in `serve/models/` (4.9 + 3.2 + 2.3 MB), so a clone runs out of
the box on CPU — the demo, the API and the whole test suite need no GPU. Every number below is
measured on the same validation set with the same metric implementation, under a protocol that is
named rather than implied.

## ✨ Highlights

- **Three models, written from scratch.** SRCNN (0.069 M, used for the seed-variance study), an SRResNet generator (1.22 M at ×4, 0.78 M at ×2) and a low-light U-Net (0.565 M). The only pretrained component is the VGG16 used for the perceptual term during training — it is not part of inference.
- **Every task beats its classical baseline.** +0.77 dB (×4) and +1.31 dB (×2) over bicubic, +10.55 dB over a no-op baseline on low light — with the very weights shipped in this repo.
- **Reproducible by construction.** 126 per-image PSNR/SSIM CSVs back the 126 reported protocol cells; `scripts/verify_public_release.py` re-averages them and checks every cell against the matrix.
- **Ships as a product.** FastAPI with per-IP rate limiting and upload guards, a bilingual Streamlit demo, a Next.js frontend, and a non-root Docker image.
- **Guarded by CI.** 101 tests behind a 75% coverage gate, a dependency audit, a frontend type-check and build, and a paper job that regenerates every table and re-checks the manuscript.

## 🖼️ Showcase

| Super-resolution ×4 | Low-light enhancement |
|:---:|:---:|
| ![SR ×4: bicubic vs PixelForge](assets/compare_sr_4x.png) | ![Low light: adaptive gamma vs PixelForge](assets/compare_lowlight.png) |
| Bicubic 26.69 dB / 0.754 → **27.47 / 0.780** | No-op 7.77 / 0.192 → **18.32 / 0.746** |

Both panels are produced on CPU by `scripts/make_compare.py` from the shipped weights.

**Running live** — the public demo, three-panel view (your upload, the model input, the output).
The same page carries a dark/light theme toggle and a Chinese/English switch.

![Live demo, low-light three-panel view](assets/demo_live_lowlight_1.png)

<details>
<summary>More screenshots (super-resolution ×4 and ×2, dark and light theme)</summary>

![Live demo, SR ×4 three-panel view](assets/demo_live_sr4.png)

![Live demo, SR ×2 three-panel view](assets/demo_live_sr2.png)

| Original upload (446×683) | Model input (223×341) | PixelForge ×2 output (446×682) |
|:---:|:---:|:---:|
| ![SR ×2 original](assets/demo_sr2_original.png) | ![SR ×2 model input](assets/demo_sr2_model_input.png) | ![SR ×2 output](assets/demo_sr2_output.png) |

</details>

## 📊 Results

| Task | Classical baseline | PixelForge | Gain |
|---|---|---|---|
| **Super-resolution ×4** | Bicubic 26.69 dB / 0.754 | **27.47 / 0.780** | **+0.77 dB / +0.026** |
| **Super-resolution ×2** | Bicubic 31.04 / 0.894 | **32.35 / 0.917** | **+1.31 dB / +0.023** |
| **Low-light** | No-op 7.77 / 0.192 | **18.32 / 0.746** | **+10.55 dB / +0.553** |

All rows: full-image validation (DIV2K val 100 images, LOL 15 pairs), RGB PSNR/SSIM, per-image
mean, LR produced by PIL bicubic downsampling — the same protocol used to pick the shipped weights.
The two super-resolution rows read **27.4977 / 32.4243 dB** when re-measured by
`scripts/protocol_matrix.py` under protocol **P8** (full image, RGB, per-image mean).

Naming the protocol is necessary but not sufficient, because a name does not pin the measurement.
Same weights, same 100 images:

| Change | Δ PSNR |
|---|---|
| PIL bicubic → torchvision bicubic + antialias, both in float | **0.012 dB** (27.4977 → 27.5098) |
| `antialias` on → off, same torchvision call | **3.73 dB** (28.95 → 25.21) |
| Same operator, but the LR image round-tripped through 8-bit | **1.96 dB** (27.51 → 25.55) |

The third row is not hypothetical: it is what one of our own kernels recorded under the label
"torchvision bicubic + antialias". `to_pil_image` clamps to [0, 1] and bicubic downscaling
overshoots, so the ringing that carries edge energy was being truncated.

<details>
<summary>Seed variance, compute ledger, and what is <em>not</em> claimed</summary>

**Seed variance** — training-internal protocol (random crop, per-image mean), *not* comparable to
the full-image table above. Five independent runs per task:

| Task | final-epoch PSNR (mean ± std, n=5) | 95% CI |
|---|---|---|
| SR ×2 (SRCNN, 100 ep) | 30.686 ± 0.069 | [30.601, 30.771] |
| SR ×4 (SRResNet, 200 ep) | 26.422 ± 0.036 | [26.378, 26.466] |
| Low-light (U-Net, 200 ep) | 18.032 ± 0.104 | [17.903, 18.161] |

Seeds move the result by 0.04–0.10 dB; toggling `antialias` moves it by 3.73 dB. **Seed choice is
not what determines the number — the protocol label is.** Logs: [`results/seeds/`](results/seeds/).

**What it cost to measure.** Every number came from free-tier Kaggle T4 kernels, logged per run:
34 kernel runs, 29.05 h of wall-clock machine time (21.72 h GPU, 7.33 h CPU), of which 6.84 h went
into 11 runs that failed or were voided. 2 500 training epochs across 15 runs (3 tasks × 5 seeds).
Details: [`results/compute_ledger.md`](results/compute_ledger.md).

**Not claimed.** No SOTA: SR ×4 at 27.47 dB beats bicubic but sits below large-dataset academic
results (30+ dB); the training set and the networks are deliberately small. The low-light gain is
relative to a no-op baseline, not to published methods.

</details>

## 🚀 Quick start

**A. Public demo — nothing to install**

<https://pixelforge-image-restoration.streamlit.app/> — the free tier sleeps when idle, so the
first visit may take a few seconds to wake up.

**B. Run the same demo locally**

```bash
pip install -r deploy/streamlit/requirements.txt   # CPU-only torch, pinned for the free tier
streamlit run deploy/streamlit/streamlit_app.py
```

**C. Self-host the API and the frontend**

```bash
# Backend — torch is not pinned in requirements.txt, install a build that fits your machine
pip install -r requirements.txt
pip install torch torchvision                      # CPU build for dev; CUDA build on a GPU box
uvicorn serve.app:app --reload --port 8000

# Frontend (second terminal)
cd web
cp .env.local.example .env.local                   # defaults to http://localhost:8000
pnpm install
pnpm dev                                           # http://localhost:3000
```

**D. Docker (backend only)**

```bash
docker build -t pixelforge-serve .
docker run -p 8000:8000 pixelforge-serve
```

The image bundles the weights and starts in ML mode as a non-root user (uid 10001). The frontend
is a separate build target and is not part of this image.

## 📁 Structure

```
pixelforge-image-restoration/
├── assets/              # banner and real comparison / demo figures
├── data/                # dataset notes (data itself is not committed)
├── models/              # training checkpoints (best of each run)
├── results/             # protocol matrix, per-image CSVs, seed logs, compute ledger
├── train/               # models.py, datasets.py, metrics.py, train.py, export.py, tests/
├── serve/               # FastAPI app, classical fallbacks, model loader, models/*.pt
├── deploy/streamlit/    # the public demo (self-contained, CPU-only)
├── web/                 # Next.js 14 + Tailwind frontend
├── paper/               # LaTeX manuscript, FACTS.md, refs.bib, tables/
├── preprint/            # published and submitted PDFs, source zip, graphical abstract
├── scripts/             # evaluation, protocol matrix, figure and paper gates
├── tests/               # API smoke, real-weight integration, E2E browser flow
├── docs/API.md          # HTTP interface reference
├── Dockerfile           # inference service image
└── DEPLOY.md            # deployment guide
```

## 🛠️ Stack

| Layer | Tech |
|---|---|
| Modelling | PyTorch 2.x, TorchVision, TorchScript |
| Training | Adam, cosine annealing, AMP, Charbonnier loss, optional VGG perceptual term |
| Metrics | PSNR, SSIM (11×11 Gaussian window, σ = 1.5) |
| Backend | FastAPI / Uvicorn, Pillow |
| Frontend | Next.js 14, React 18, TypeScript, Tailwind CSS |
| Deploy | Streamlit Community Cloud, Docker, Vercel (optional) |

## 🧪 Tests

```bash
python -m pytest                                    # 101 passed, 1 skipped (the E2E module)
python -m pytest --cov --cov-report=term-missing    # coverage gate: 75%
cd web && pnpm test && pnpm exec tsc --noEmit
```

The skip is `tests/e2e/test_e2e.py`, which drives a real Chromium through the upload flow; run it
directly or set `PIXELFORGE_RUN_E2E=1`. The preprint quotes 81 tests — that was the count when it was
published; the seven service-guard tests and twelve data-pipeline tests added afterwards are why the
repo now reports 101. CI runs
four jobs — tests with the coverage gate,
`pip-audit` over the lock file, the frontend type-check + tests + build, and a paper job that
regenerates every table, rebuilds the manuscript, checks its layout, re-derives the headline
numbers independently and audits 27 references (83 checks).

## ❓ FAQ

**Do I need a GPU?**
No. The public demo, the local demo, the API and the full test suite run on CPU. Only training
needs a GPU.

**Which tasks have a trained weight?**
All three: SR ×4, SR ×2 and low light. If a weight file is ever missing, the service falls back to
the classical method for that slot instead of failing — `GET /api/health` reports which engine each
slot is using.

**The SR ×4 output was once described as "softer than bicubic" — is that true?**
No, and it is not softer. Re-measured on 9 images under four gradient definitions, the model is
sharper than bicubic in **9 of 9** cases (luma gradient mean **1.92 vs 1.45**). The perceptual term
is also not what buys the PSNR: trained with `--perceptual` the model scores 27.61 / 0.7835, without
it 27.60 / 0.7836 — a 0.01 dB difference. An earlier revision of this README carried the opposite
claim (1.39 vs 1.45); it was wrong on both halves and was withdrawn.

**Why is the demo slow on first visit?**
The free Streamlit tier sleeps when idle. Inference itself is CPU-side and quick for typical photos.

**Can it run behind a restricted network?**
Yes — see [DEPLOY.md](DEPLOY.md) for pushing the image through a mirror when GitHub is unreachable.

## 📄 Paper & citation

The evaluation discipline this repo enforces in code is written up as a preprint:

- **Title**: *The Protocol Is the Result: How Evaluation Choices Flip the Verdict on Lightweight Image Restoration*
- **Author**: Yulin Zhao (Elijah Zhao) — Independent Researcher
- **Venue**: Preprints.org, Article, posted 5 November 2026 (not peer-reviewed, CC BY 4.0)
- **DOI**: [10.20944/preprints202611.0550.v1](https://doi.org/10.20944/preprints202611.0550.v1)
- **Files**: published 15-page version in [`preprint/published/`](preprint/published/), submitted
  version and source in [`preprint/submission/`](preprint/submission/), LaTeX in [`paper/`](paper/)

<details>
<summary>BibTeX</summary>

```bibtex
@article{zhao2026protocol,
  title   = {The Protocol Is the Result: How Evaluation Choices Flip the Verdict
             on Lightweight Image Restoration},
  author  = {Zhao, Yulin},
  journal = {Preprints},
  year    = {2026},
  doi     = {10.20944/preprints202611.0550.v1},
  note    = {Preprint, not peer-reviewed}
}
```

</details>

## 📝 Changelog

Full history in [CHANGELOG.md](CHANGELOG.md). Recent:

- **1.12.2** — Two silent defects in the training data pipeline: the SR validation centre-crop read outside the image and filled the border with black (55.6% of the patch on a 64×64 image), and the low-light validation crashed on any native size that is not a multiple of 8. Neither affected the published numbers, and both are now pinned by regression tests.
- **1.12.1** — Service-layer hardening: the per-IP rate limiter can no longer be switched off with a spoofed `X-Forwarded-For`, oversized uploads are refused before the body is buffered, and the classical low-light baseline no longer divides by zero on white images.
- **1.12.0** — Preprint published; DOI added to both READMEs, the Streamlit sidebar and `CITATION.cff`.
- **1.11.3** — Corrected the compute figures to match the ledger (34 kernel runs / 29.05 h), the seed table's low-light label (U-Net, not SRCNN) and the CI description (four jobs).
- **1.11.0** — Author set to Independent Researcher; paper section added; process notes moved out of the public tree.

## ⚖️ License

MIT — see [LICENSE](LICENSE). Interface reference in [`docs/API.md`](docs/API.md), deployment notes
in [DEPLOY.md](DEPLOY.md), contribution guide in [CONTRIBUTING.md](CONTRIBUTING.md), citation
metadata in [CITATION.cff](CITATION.cff). Datasets ([DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/),
[LOL](https://github.com/weichen582/RetinexNet)) are not redistributed here; they are available from
their original sources.

---

<p align="center">
  <sub>PixelForge · end-to-end image restoration &nbsp;·&nbsp; <a href="https://pixelforge-image-restoration.streamlit.app/">Live Demo</a></sub>
</p>
