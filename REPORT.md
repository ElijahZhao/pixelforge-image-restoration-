# PixelForge — a project report

*A from-scratch image-restoration pipeline: the models, the training, and a demo you can actually click.*

---

I got tired of image-enhancement tools where I had no idea what was happening inside. You upload a
photo, something magic runs, you get a result. For this project I built the whole thing myself —
the models, the training code, the evaluation, and a demo people can use. No calling someone
else's API. If a step matters, I wrote it.

It does three things: upscale a photo 4×, upscale it 2×, and brighten a dark photo. For each one
there's a model I trained, and the weights ship in the repo, so it runs the moment you clone it.

## Why train my own instead of downloading one

You can pull a pretrained super-resolution model off the shelf in five minutes. I didn't, on
purpose. I wanted to understand the pipeline end to end — how data gets loaded, how a loss
function actually shapes what the model learns, why a model that scores *higher* can *look worse*.
Training my own forced me to confront all of that instead of hiding behind a download button.

The honest part: my models are not state of the art. SR ×4 lands at 27.47 dB PSNR. Academic models
trained on huge datasets sit at 30+ dB. That gap is real, and it comes from two things I controlled:
a small training set and deliberately small networks. I'm fine with that. The point was to build
the machine, not to win a benchmark.

## What's actually in the box

- **Three trained weights:** `sr_generator_scale4.pt` (SRResNet, 27.47 dB), `sr_generator_scale2.pt`
  (SRCNN, 32.35 dB), `lowlight.pt` (U-Net, 18.32 dB).
- **A full training loop:** data loading, Adam with cosine annealing, mixed precision, optional
  perceptual loss, PSNR/SSIM evaluation, TorchScript export.
- **A Streamlit demo** with a bilingual UI and a light/dark theme. This is the live thing — it's
  deployed on Streamlit Community Cloud and you can use it right now.
- **A FastAPI backend and a Next.js frontend**, also in the repo, if you want to self-host.

The numbers below are measured on the same validation set with the same code, so the only variable
is the method:

| Task | Baseline | Mine | Gain |
|---|---|---|---|
| Super-resolution ×4 | Bicubic 26.69 / 0.754 | **27.47 / 0.780** | +0.77 dB / +0.026 |
| Super-resolution ×2 | Bicubic 31.04 / 0.894 | **32.35 / 0.917** | +1.31 dB / +0.023 |
| Low-light enhancement | No-op 7.77 / 0.192 | **18.32 / 0.746** | +10.55 dB / +0.553 |

One thing worth being straight about: the SR ×4 model scores higher than bicubic but looks a little
softer and greyer. That's not a bug. It uses a perceptual loss, which optimises for "similar in
feature space" rather than "sharp pixels," so you get higher PSNR with slightly softer edges. I
measured it — the output gradient mean is 1.39 versus 1.45 for bicubic. I kept it in because it's
honest behaviour of the method, not a defect.

## The one deployment that's actually live

I'll be blunt about this, because it's a real constraint, not a half-finished feature. The only
thing running in public is the Streamlit demo. I'm a student. I'm not paying for a server with a GPU
to run inference, and I'm certainly not paying for one that could survive real traffic if this ever
got popular. So the FastAPI service and the Next.js frontend are built, tested, and documented —
they run fine if you self-host them — but they're not deployed anywhere.

The Streamlit app runs on CPU on a free tier, which is plenty for one person uploading a photo at a
time. That's a deliberate scope choice given the budget, and it's why the project is "done" with
exactly one public face.

## How to run it

The public demo needs no setup: <https://pixelforge-image-restoration.streamlit.app>.

Locally, everything runs on CPU:

```bash
pip install -r deploy/streamlit/requirements.txt
streamlit run deploy/streamlit/streamlit_app.py
```

Retraining needs a GPU — I trained the shipped weights on a rented RTX 4090 for a few days. The test
suite is 39 passing tests and runs on CPU, so you don't need a GPU to confirm the code works.

## What I'd do with more time

- Train on a bigger dataset with a deeper network to close the SR ×4 gap to SOTA. The pipeline is
  already there; it's just data and compute.
- Swap the single-instance, in-memory rate limiter for a Redis-backed one so the backend could
  handle multiple users instead of one at a time.
- A short slides deck to go with this report.

---

That's the project. It's complete, it's reproducible, and it does what it says on the tin. The demo
is live; the rest is one `git clone` away.
