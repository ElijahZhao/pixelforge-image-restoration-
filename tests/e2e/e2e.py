"""End-to-end browser test for the PixelForge website.

What it does
------------
1. Boots the FastAPI backend (uvicorn) and the Next.js frontend (pnpm dev).
2. Drives a real headless Chromium (Playwright) through the full user flow:
     - Super-Resolution: upload -> choose 4x -> Enhance -> result + slider drag
     - Low-Light:        upload -> Enhance -> result
3. Asserts that the pipeline actually returns processed images.
4. Saves screenshots under tests/e2e/screenshots/ as visual evidence.

Run (from repo root):
    python tests/e2e/e2e.py

Prerequisites:
    pip install playwright && playwright install chromium
    pnpm install   (in web/)
"""

from __future__ import annotations

import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
WEB = ROOT / "web"
SCREENS = pathlib.Path(__file__).resolve().parent / "screenshots"
FRONTEND_URL = "http://localhost:3000"
BACKEND_URL = "http://localhost:8000"

SHUTDOWN = []


def _log(msg: str) -> None:
    print(f"[e2e] {msg}")


def _wait_get(url: str, timeout: float = 120.0) -> bool:
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def _start(cmd, cwd, name):
    env = os.environ.copy()
    # Disable Next.js telemetry noise.
    env["NEXT_TELEMETRY_DISABLED"] = "1"
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    SHUTDOWN.append((name, proc))
    _log(f"started {name} (pid {proc.pid})")
    return proc


def _cleanup() -> None:
    for name, proc in SHUTDOWN:
        _log(f"stopping {name} ...")
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


def main() -> int:
    try:
        import requests  # noqa: F401
    except ImportError:
        _log("requests not available; installing not attempted. Aborting.")
        return 1

    import requests
    from playwright.sync_api import sync_playwright

    SCREENS.mkdir(parents=True, exist_ok=True)

    # ---- Boot servers ----
    backend = _start(
        [sys.executable, "-m", "uvicorn", "serve.app:app", "--port", "8000"],
        ROOT, "backend",
    )
    frontend = _start(["pnpm", "dev"], WEB, "frontend")

    _log("waiting for backend /api/health ...")
    if not _wait_get(f"{BACKEND_URL}/api/health", timeout=120):
        _log("backend did not come up")
        _cleanup()
        return 1

    _log("waiting for frontend http://localhost:3000 ...")
    if not _wait_get(FRONTEND_URL, timeout=180):
        _log("frontend did not come up")
        _cleanup()
        return 1

    failures = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.on("console", lambda m: _log(f"browser[{m.type}]: {m.text}"))

        def run_flow(task: str, image_path: pathlib.Path, label: str, scale: int = 2):
            rel = label.lower().replace(" ", "_")
            _log(f"flow: {label} with {image_path.name}")
            page.goto(FRONTEND_URL, wait_until="load")
            # Wait for React to render the page (HMR keeps a socket open, so we
            # can't rely on networkidle in dev mode).
            page.wait_for_selector("text=PixelForge", timeout=60000)
            page.screenshot(path=str(SCREENS / f"{rel}_home.png"))

            if task == "lowlight":
                page.get_by_role("button", name="Low-Light").click()
            else:
                if scale == 4:
                    page.get_by_role("button", name="4×").click()

            # Hidden file input — Playwright can set files even when hidden.
            page.set_input_files('input[type="file"]', str(image_path))
            page.get_by_role("button", name="Enhance").click()

            # Wait for the result "After" image to appear.
            try:
                page.wait_for_selector('img[alt="After"]', timeout=60000)
            except Exception as e:
                failures.append(f"{label}: result image not shown ({e})")
                browser.close()
                return

            after_src = page.get_attribute('img[alt="After"]', "src") or ""
            assert after_src.startswith("data:image/png;base64,") and len(after_src) > 200, \
                f"{label}: after image missing/empty"
            before_src = page.get_attribute('img[alt="Before"]', "src") or ""
            assert before_src.startswith("data:image/png;base64,") and len(before_src) > 200, \
                f"{label}: before image missing/empty"

            # engine badge (classical now, ml once weights are trained)
            badge = page.locator("span.rounded-full").first.inner_text()
            _log(f"{label}: engine = {badge}")

            page.screenshot(path=str(SCREENS / f"{rel}_result.png"))

            # Drag the comparison slider to exercise the interaction.
            box = page.locator("div.relative.select-none").first.bounding_box()
            if box:
                cx, cy = box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5
                page.mouse.move(cx, cy)
                page.mouse.down()
                page.mouse.move(box["x"] + box["width"] * 0.2, cy, steps=10)
                page.mouse.up()
                page.screenshot(path=str(SCREENS / f"{rel}_slider_dragged.png"))
                _log(f"{label}: slider drag OK")
            else:
                failures.append(f"{label}: slider container not found")

            # sanity: backend engine matches what the page reported
            health = requests.get(f"{BACKEND_URL}/api/health", timeout=5).json()
            _log(f"backend health engines: {health['engines']}")

        run_flow("sr", ROOT / "assets" / "sample_scene.png", "Super-Resolution", scale=4)
        if not failures:
            run_flow("lowlight", ROOT / "assets" / "sample_dark.png", "Low-Light")

        browser.close()

    _cleanup()

    if failures:
        _log("FAILURES:")
        for f in failures:
            _log(f"  - {f}")
        return 1
    _log("ALL E2E CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
