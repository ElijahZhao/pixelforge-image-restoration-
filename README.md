# PixelForge — AI Image Restoration (CV Portfolio)

> A computer-vision portfolio project for graduate-school applications.
> Self-trained PyTorch models for **single-image super-resolution** and
> **low-light image enhancement**, served through an interactive web app.

This is **not** an API-wrapping toy. The full pipeline is implemented and
reproducible:

```
data/        public datasets (download instructions inside)
train/       PyTorch models, datasets, metrics, training & export scripts
serve/       FastAPI inference service (+ classical fallbacks)
web/         Next.js frontend (upload → before/after slider)
results/     training logs + quantitative comparison table
```

## Highlights for admissions committees
- **Real ML, not API glue.** SRCNN / SRResNet (super-resolution) and a U-Net
  (low-light) implemented and trained in PyTorch, with PSNR/SSIM evaluation.
- **End-to-end engineering.** Data pipeline → training → evaluation → model
  export (TorchScript) → FastAPI serving → Next.js frontend.
- **Reproducible.** Every script is runnable; training runs on free GPUs.
- **Interactive demo.** Visitors upload an image and drag a slider to compare
  before/after — instantly understandable.

## Quick start (local, no GPU, no training needed)
The service ships with **classical baselines**, so it runs out of the box:

```bash
# 1) Backend (FastAPI)
pip install -r requirements.txt
uvicorn serve.app:app --reload --port 8000

# 2) Frontend (Next.js) in another terminal
cd web
cp .env.local.example .env.local   # points to http://localhost:8000
pnpm install
pnpm dev
```

Open http://localhost:3000, upload an image, choose a task, press **Enhance**.

## Train the real models (free GPU)
```bash
# Download DIV2K + LOL into data/ (see data/README.md)
pip install -r requirements.txt

# Super-resolution (advanced generator, 4x, perceptual loss)
python train/train.py --task sr --model generator --scale 4 \
    --data_root data --epochs 200 --batch_size 8 --lr 1e-4 --perceptual

# Low-light enhancement
python train/train.py --task lowlight --data_root data \
    --epochs 200 --batch_size 8 --lr 2e-4

# Export to TorchScript for serving
python train/export.py --checkpoint models/sr_generator_scale4_best.pth \
    --out serve/models/sr_generator_scale4.pt --task sr --scale 4
python train/export.py --checkpoint models/lowlight_generator_best.pth \
    --out serve/models/lowlight.pt --task lowlight
```
Drop the exported `.pt` files into `serve/models/`. The service automatically
switches from classical baselines to your trained models (see `/api/health`).

## Deploy
- **Frontend**: `vercel` (import `web/`). Set `NEXT_PUBLIC_API_URL` to your
  backend URL.
- **Backend**: Hugging Face Spaces (`gradio_demo.py` for a 1-click demo, or
  `app.py` on a small VPS). See `DEPLOY.md`.

## Project description (for SOP / resume)
> *PixelForge is an end-to-end computer-vision system I built to restore and
> enhance images. I implemented a super-resolution pipeline (SRCNN and a deeper
> SRResNet trained with a perceptual loss) and a U-Net for low-light
> enhancement, trained them on DIV2K and LOL, and evaluated with PSNR/SSIM. I
> exported the models to TorchScript and served them through a FastAPI backend
> with a Next.js front end featuring an interactive before/after comparison.*

See `web/app/method/page.tsx` for the full English method write-up.
