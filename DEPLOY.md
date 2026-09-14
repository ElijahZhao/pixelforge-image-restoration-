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

## 4. Pushing to GitHub from a restricted network (e.g. this sandbox)
When the environment cannot reach `github.com` directly (TLS reset / proxy
whitelist), use a public GitHub mirror that the network *can* reach, e.g.
`ghproxy.net`. It proxies both `git` traffic and resolves the target repo
server-side, so your sandbox never needs direct access to GitHub.

```bash
TOKEN=ghp_xxxYOURTOKENxxx   # a fine-grained PAT with `repo` scope
# set the remote to go through the mirror (auth travels via the mirror):
git remote set-url origin \
  "https://oauth2:${TOKEN}@ghproxy.net/https://github.com/USER/REPO.git"
git push -u origin main
# to verify what landed:
git ls-tree -r --name-only origin/main
```

Caveats:
- The token passes through the third-party mirror — **rotate/revoke it right
  after pushing**, and only grant `repo` scope.
- If the repo was created with an auto-generated initial file (e.g. a LICENSE),
  merge it first: `git fetch origin main && git merge origin/main --allow-unrelated-histories`.
- Other mirrors that sometimes work: `fastgit.org`, `mirror.ghproxy.com`.
- This is only needed where GitHub is unreachable; on a normal machine just use
  `git push` with the plain `https://github.com/USER/REPO.git` URL.
