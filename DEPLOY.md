# Deployment

## 1. Frontend (Next.js) → Vercel
```bash
cd web
cp .env.local.example .env.local   # or set in Vercel dashboard
# set NEXT_PUBLIC_API_URL to your backend URL (HF Space or VPS)
vercel --prod
```
Vercel auto-detects Next.js. Set the env var `NEXT_PUBLIC_API_URL` in the
project settings to point at the deployed backend.

## 2. Backend (FastAPI) → Hugging Face Spaces (free, optional GPU)
### Option A — Gradio one-click demo
Create a Spaces repo (Gradio SDK), copy `serve/gradio_demo.py`, `serve/classical.py`,
`serve/model_loader.py` and `requirements.txt`, then `python serve/gradio_demo.py`
is auto-launched. Drop exported `.pt` weights into `serve/models/` for ML mode.

### Option B — Full FastAPI service
Use a Docker Spaces or a small VPS. Run:
```bash
pip install -r requirements.txt
uvicorn serve.app:app --host 0.0.0.0 --port 7860
```
Make sure `serve/models/` contains the exported TorchScript weights if you want
the trained models (otherwise it serves classical baselines).

## 3. Local (dev) — both together
```bash
# terminal 1
uvicorn serve.app:app --reload --port 8000
# terminal 2
cd web && pnpm install && pnpm dev
# open http://localhost:3000
```

## Notes
- The frontend calls the backend via `NEXT_PUBLIC_API_URL`. If that env var is
  **unset**, the frontend uses the same-origin relative path `/api/*`, and the
  Next.js dev server (`next.config.mjs` rewrites) proxies it to the local
  FastAPI service at `http://localhost:8000` — perfect for local dev with no
  CORS issues. In production, set `NEXT_PUBLIC_API_URL` to your deployed
  backend (e.g. the HF Space URL) so the browser calls it directly.
- CORS is open (`*`) in `serve/app.py` for convenience — tighten it to your
  frontend domain before going public.
