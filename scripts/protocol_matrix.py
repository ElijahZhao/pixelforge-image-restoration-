#!/usr/bin/env python3
"""Protocol matrix: score the same checkpoint under P1..P6 and show the spread.

Why this exists
---------------
The paper's Figure 1 is "the same model, six protocols, six different numbers".
Building it used to require a Kaggle kernel
(`archive_private/06-实验脚本与输出/kernel-output-protocol-matrix/run.py`),
i.e. the number came from code that lived outside the repo. This script is the
repo-side implementation, so the figure is reproducible from a clean clone:

    git clone ... && python scripts/protocol_matrix.py --ckpt_dir models

The six protocols (each one a choice, each one named in the output):

    P1  random crop, best-of-N, RGB, bicubic AA        -- reproduces the OLD behaviour
    P2  centre crop, RGB, bicubic AA                   -- the training-time val metric
    P3  full image, RGB, bicubic AA                    -- this repo's "authoritative" number
    P4  full image, RGB, + border shave                -- literature, colour
    P5  full image, Y channel, + border shave          -- literature, complete
    P6  full image, Y channel, + shave, NO antialias   -- the dimension that flips the sign

P1 vs P2 vs P3 is *sampling*; P3 vs P4 vs P5 is *metric channel*; P5 vs P6 is
*degradation operator*. Those three are what the matrix decomposes: variance
from the protocol versus distance between models. In the pre-fix code P1 could
report a number 1.6 dB above P2 purely because it selected the best of N random
crops per image afterwards -- a best-of-N artifact, not a better model.

Outputs
-------
``<out>/protocol_matrix.json``   nested: ckpt -> protocol -> metrics
``<out>/protocol_matrix.csv``    tidy one row per (ckpt, protocol)
``<out>/per_image/<ckpt>__<P>.csv``  per-image PSNR/SSIM, so a reviewer can
                                 recompute any aggregate without trusting ours

Usage
-----
    python scripts/protocol_matrix.py --ckpt models/sr_generator_scale4_best.pth --scale 4
    python scripts/protocol_matrix.py --ckpt_dir models/seeds --scale 4 --out results/protocol_matrix
    python scripts/protocol_matrix.py --ckpt_dir models/seeds --scale 4 --protocols P3,P5,P6
    python scripts/protocol_matrix.py --ckpt ... --max_images 20        # quick look
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, os.path.abspath("train"))
from degradation import MODES, build_lr, describe  # noqa: E402
from eval_protocol import psnr_y, ssim_y, shave  # noqa: E402
from metrics import psnr_per_image, ssim_per_image  # noqa: E402
from models import build_model  # noqa: E402

_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#: Protocol definitions. `sampling` x `metric` x `degrade` is the whole axis
#: system; the six named protocols are points in that space, not six code paths.
PROTOCOLS = {
    "P1": dict(sampling="random", metric="rgb", degrade="tv_bicubic_aa",
               agg="best_of_n", label="random crop, best-of-N"),
    "P2": dict(sampling="center", metric="rgb", degrade="tv_bicubic_aa",
               agg="per_image", label="centre crop (training val)"),
    "P3": dict(sampling="full", metric="rgb", degrade="tv_bicubic_aa",
               agg="per_image", label="full image, RGB"),
    "P4": dict(sampling="full", metric="rgb_shave", degrade="tv_bicubic_aa",
               agg="per_image", label="full image, RGB + shave"),
    "P5": dict(sampling="full", metric="y_shave", degrade="tv_bicubic_aa",
               agg="per_image", label="full image, Y + shave"),
    "P6": dict(sampling="full", metric="y_shave", degrade="tv_bicubic_noaa",
               agg="per_image", label="full image, Y + shave, no antialias"),
}


def _load_images(folder, max_images=0):
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(_EXTS))
    if max_images:
        files = files[:max_images]
    return files


def find_checkpoints(ckpt_dir):
    """Every .pth/.pt under a directory, sorted (a seed matrix lives here)."""
    out = []
    for pat in ("**/*.pth", "**/*.pt"):
        out.extend(glob.glob(os.path.join(ckpt_dir, pat), recursive=True))
    return sorted(p for p in out if os.path.isfile(p))


def load_model(path, task, scale):
    """Load either a TorchScript export (``serve/models/*.pt``) or a state dict.

    Both are in the repo for good reason: the deployed weights are TorchScript
    ``.pt`` (they are what the service actually runs), while seed-matrix
    experiments produce ``.pth`` state dicts. A protocol table that can only be
    built from one of them cannot answer "is the *deployed* model's number
    reproducible", which is the question a reviewer asks first.
    """
    try:
        model = torch.jit.load(path, map_location="cpu")
        model.eval()
        return model.to(DEVICE), {"source": "torchscript"}
    except Exception:  # noqa: BLE001 - not a TorchScript file, fall through
        pass

    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    name = ck.get("model") if isinstance(ck, dict) else None
    sc = ck.get("scale", scale) if isinstance(ck, dict) else scale
    model = build_model(task, scale=sc, advanced=(name == "generator"))
    model.load_state_dict(state)
    model.eval()
    meta = {k: ck.get(k) for k in ("seed", "crop_size", "tag", "epoch", "val_psnr")} \
        if isinstance(ck, dict) else {}
    meta["source"] = "state_dict"
    return model.to(DEVICE), meta


def _crop_boxes(h, w, crop, n, rng):
    if crop is None or crop >= min(h, w):
        return [(0, 0, h, w)]
    if n <= 1 and crop == min(h, w):
        return [(0, 0, crop, crop)]
    # numpy Generator uses `integers` (half-open high); `randint` is the legacy
    # RandomState API and silently fails here.
    return [(int(rng.integers(0, h - crop + 1)), int(rng.integers(0, w - crop + 1)),
             crop, crop) for _ in range(n)]


@torch.no_grad()
def score_one(model, hr_pil, proto, scale, border, crop, n_random, rng):
    """Return per-image (psnr, ssim) for one image under one protocol.

    For ``best_of_n`` the returned pair is the max over the N windows, which is
    exactly the old behaviour and exactly why P1 is not comparable to P2.
    """
    w, h = hr_pil.size
    if proto["sampling"] == "center":
        boxes = [((h - min(crop or h, h)) // 2, (w - min(crop or w, w)) // 2,
                  min(crop or h, h), min(crop or w, w))]
    elif proto["sampling"] == "random":
        boxes = _crop_boxes(h, w, crop, n_random, rng)
    else:
        boxes = [(0, 0, h, w)]

    best = None
    vals = []
    for (top, left, ch, cw) in boxes:
        patch = hr_pil.crop((left, top, left + cw, top + ch))
        hr = TF.to_tensor(patch).unsqueeze(0).to(DEVICE)
        lr = build_lr(hr, scale, mode=proto["degrade"])
        out = model(lr).clamp(0, 1)
        if out.shape[-2:] != hr.shape[-2:]:
            # The model's true output is lr*scale; when H/W are not divisible by
            # the scale, the grid differs by a pixel. Resize back so PSNR is
            # measured on the HR grid -- an evaluation-time normalisation, not a
            # second upscale (same convention as scripts/eval_baseline.py).
            out = TF.resize(out, list(hr.shape[-2:]), antialias=False)

        if proto["metric"] == "rgb":
            p = float(psnr_per_image(out, hr).mean())
            s = float(ssim_per_image(out, hr).mean())
        elif proto["metric"] == "rgb_shave":
            p = float(psnr_per_image(shave(out, border), shave(hr, border)).mean())
            s = float(ssim_per_image(shave(out, border), shave(hr, border)).mean())
        elif proto["metric"] == "y":
            p = float(psnr_y(out, hr, border=0).mean())
            s = float(ssim_y(out, hr, border=0).mean())
        elif proto["metric"] == "y_shave":
            p = float(psnr_y(out, hr, border=border).mean())
            s = float(ssim_y(out, hr, border=border).mean())
        else:
            raise ValueError(proto["metric"])

        vals.append((p, s))
        best = (p, s) if best is None or p > best[0] else best

    if proto["agg"] == "best_of_n":
        return best
    if proto["agg"] == "mean_of_n":
        return (statistics.mean(v[0] for v in vals), statistics.mean(v[1] for v in vals))
    return vals[0]


def run(args):
    val_dir = os.path.join(args.data_root, "div2k", "val")
    if args.task == "lowlight":
        val_dir = os.path.join(args.data_root, "lol", "val", "high")
    if not os.path.isdir(val_dir):
        print(f"[!] no validation images at {val_dir}")
        return 2
    files = _load_images(val_dir, args.max_images)
    if not files:
        print(f"[!] {val_dir} is empty")
        return 2

    ckpts = []
    if args.ckpt:
        ckpts.extend(args.ckpt)
    if args.ckpt_dir:
        ckpts.extend(find_checkpoints(args.ckpt_dir))
    if not ckpts:
        print("[!] no checkpoint given (--ckpt or --ckpt_dir)")
        return 2

    wanted = [p.strip().upper() for p in args.protocols.split(",") if p.strip()]
    for p in wanted:
        if p not in PROTOCOLS:
            print(f"[!] unknown protocol {p}; known: {sorted(PROTOCOLS)}")
            return 2

    border = args.shave if args.shave is not None else args.scale
    print(f"[i] {len(files)} images from {val_dir} | device={DEVICE} | "
          f"scale=x{args.scale} | shave={border} | crop={args.crop} | "
          f"n_random={args.n_random}")
    print(f"[i] 退化算子会被逐个协议记录; 训练侧用的是 {describe('tv_bicubic_aa')}")

    os.makedirs(args.out, exist_ok=True)
    per_image_dir = os.path.join(args.out, "per_image")
    os.makedirs(per_image_dir, exist_ok=True)

    result = {"protocols": {p: PROTOCOLS[p] for p in wanted},
              "config": {"scale": args.scale, "shave": border, "crop": args.crop,
                         "n_random": args.n_random, "n_images": len(files),
                         "device": DEVICE, "data": val_dir},
              "checkpoints": {}}
    rows = []

    for ckpt in ckpts:
        name = os.path.splitext(os.path.basename(ckpt))[0]
        try:
            model, meta = load_model(ckpt, args.task, args.scale)
        except Exception as exc:  # noqa: BLE001
            print(f"[!] cannot load {ckpt}: {exc}")
            continue
        tag = f"seed={meta.get('seed')}" if meta.get("seed") else meta.get("source")
        print(f"\n=== {name} ({tag}) ===")
        rec = {"path": ckpt, "meta": meta, "protocols": {}}

        for p in wanted:
            proto = PROTOCOLS[p]
            # A fresh RNG per (checkpoint, protocol) so P1's window draws do not
            # depend on how many protocols ran before it -- otherwise the matrix
            # would not be reproducible from a subset of protocols.
            rng = np.random.default_rng(args.seed)
            t0 = time.time()
            ps, ss = [], []
            for f in files:
                hr = Image.open(f).convert("RGB")
                pv, sv = score_one(model, hr, proto, args.scale, border,
                                   args.crop, args.n_random, rng)
                ps.append(pv)
                ss.append(sv)
            rec["protocols"][p] = {
                "label": proto["label"], "degrade": proto["degrade"],
                "metric": proto["metric"], "sampling": proto["sampling"],
                "agg": proto["agg"],
                "psnr_mean": round(statistics.mean(ps), 4),
                "psnr_std": round(statistics.stdev(ps), 4) if len(ps) > 1 else None,
                "ssim_mean": round(statistics.mean(ss), 4),
                "n_images": len(ps), "sec": round(time.time() - t0, 1),
            }
            with open(os.path.join(per_image_dir, f"{name}__{p}.csv"), "w",
                      newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["image", "psnr", "ssim"])
                for fpath, pv, sv in zip(files, ps, ss):
                    w.writerow([os.path.basename(fpath), f"{pv:.6f}", f"{sv:.6f}"])
            r = rec["protocols"][p]
            print(f"  {p} {r['psnr_mean']:7.3f} dB / {r['ssim_mean']:.4f}  "
                  f"[{proto['label']}] ({r['sec']}s)")
            rows.append({"checkpoint": name, "protocol": p, "label": proto["label"],
                         "degrade": proto["degrade"], "metric": proto["metric"],
                         "sampling": proto["sampling"], "agg": proto["agg"],
                         "psnr_mean": r["psnr_mean"], "psnr_std": r["psnr_std"],
                         "ssim_mean": r["ssim_mean"], "n_images": r["n_images"]})
        result["checkpoints"][name] = rec

    with open(os.path.join(args.out, "protocol_matrix.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    if rows:
        with open(os.path.join(args.out, "protocol_matrix.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # The spread is the result: if P1..P6 vary more than the models do, the
    # paper's claim is about protocols and must be phrased that way.
    print("\n=== spread across protocols (same checkpoint) ===")
    for name, rec in result["checkpoints"].items():
        vals = [v["psnr_mean"] for v in rec["protocols"].values()]
        if len(vals) > 1:
            print(f"  {name}: {min(vals):.3f} .. {max(vals):.3f} dB "
                  f"(spread {max(vals) - min(vals):.3f} dB)")
    print(f"\n[i] wrote {args.out}/protocol_matrix.{{json,csv}} and per_image/*.csv")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", help="checkpoint path (repeatable)")
    ap.add_argument("--ckpt_dir", default="", help="directory of checkpoints (recursive)")
    ap.add_argument("--task", choices=["sr", "lowlight"], default="sr")
    ap.add_argument("--scale", type=int, default=4, choices=[1, 2, 4])
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--protocols", default=",".join(PROTOCOLS),
                    help=f"comma list from {sorted(PROTOCOLS)}")
    ap.add_argument("--crop", type=int, default=96,
                    help="patch side for P1/P2 (default 96 = training val crop)")
    ap.add_argument("--n_random", type=int, default=60,
                    help="random windows per image for P1 (default 60; the old "
                         "code effectively used --epochs per image)")
    ap.add_argument("--shave", type=int, default=None,
                    help="border pixels for shave protocols (default = scale)")
    ap.add_argument("--max_images", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/protocol_matrix")
    a = ap.parse_args(argv)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
