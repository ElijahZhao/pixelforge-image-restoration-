#!/usr/bin/env python3
"""基线评测: 模型 vs bicubic/不处理, 同口径对比（批量 + GPU 加速版）。

相比逐张版本:
  - 所有指标在 DEVICE(GPU) 上算, 不再 CPU/GPU 来回搬
  - 可选 --max_images 限制张数, 快速出结论
  - 每处理若干张打印一次进度, 不再"看着像卡死"

用法:
    python scripts/eval_baseline.py --task sr --scale 4
    python scripts/eval_baseline.py --task sr --scale 4 --max_images 30
    python scripts/eval_baseline.py --task lowlight
自动适配 data/div2k 与 data/lol 两种目录命名。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, os.path.abspath("train"))
from metrics import evaluate_batch  # noqa: E402

_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_images(folder):
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(_EXTS)]


def _pick_dir(data_root, *names):
    for n in names:
        p = os.path.join(data_root, n)
        if os.path.isdir(p):
            return p
    return None


def _ml():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ml", "serve/model_loader.py")
    ml = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ml)
    return ml


def _metric(pred_t, tgt_t):
    """在 DEVICE 上算 psnr/ssim, 返回 (p, s)。"""
    pred_t = pred_t.to(DEVICE)
    tgt_t = tgt_t.to(DEVICE)
    m = evaluate_batch(pred_t, tgt_t)
    return m["psnr"], m["ssim"]


def eval_sr(scale, data_root, max_images):
    sr_dir = _pick_dir(data_root, "div2k", "sr")
    if sr_dir is None:
        print("[!] 未找到 SR 数据目录"); return
    hrs = _load_images(os.path.join(sr_dir, "val"))
    if not hrs:
        print(f"[!] 未找到验证图: {sr_dir}/val"); return
    if max_images:
        hrs = hrs[:max_images]
    print(f"[i] SR 验证目录: {sr_dir}/val ({len(hrs)} 张) | device={DEVICE}")

    ml = _ml()
    model = ml.get_sr_model(scale)
    if model is None:
        print(f"[!] 无 sr scale={scale} 权重 (缺失或损坏)"); return

    bic_p, bic_s, mod_p, mod_s = [], [], [], []
    for i, f in enumerate(hrs, 1):
        hr = Image.open(f).convert("RGB")
        lr = hr.resize((max(1, hr.width // scale), max(1, hr.height // scale)),
                       Image.BICUBIC)
        bic = lr.resize(hr.size, Image.BICUBIC)
        x = TF.to_tensor(lr).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(x).clamp(0, 1)
        # NOTE: the model's TRUE output is lr*scale (it never sees the full-res
        # HR). We bicubic-resize it back to hr.size here *only* so the PSNR/SSIM
        # comparison is measured at the same pixel grid as the bicubic baseline.
        # This resize is an evaluation-time normalization, NOT a hidden second
        # upscale — contrast with serve/model_loader.predict_sr, which returns
        # the model's true lr*scale output to the caller (no resize).
        pred = TF.to_pil_image(out.squeeze(0).cpu()).resize(hr.size, Image.BICUBIC)

        hr_t = TF.to_tensor(hr).unsqueeze(0)
        p, s = _metric(TF.to_tensor(bic).unsqueeze(0), hr_t); bic_p.append(p); bic_s.append(s)
        p, s = _metric(TF.to_tensor(pred).unsqueeze(0), hr_t); mod_p.append(p); mod_s.append(s)
        if i % 10 == 0 or i == len(hrs):
            print(f"    ... {i}/{len(hrs)} 张", flush=True)

    bp, bs = float(np.mean(bic_p)), float(np.mean(bic_s))
    mp, ms = float(np.mean(mod_p)), float(np.mean(mod_s))
    print(f"\n=== SR x{scale} · 验证集 {len(hrs)} 张 · 与 bicubic 对比 ===")
    print(f"  bicubic 基线 : PSNR {bp:6.2f} | SSIM {bs:.4f}")
    print(f"  模型输出     : PSNR {mp:6.2f} | SSIM {ms:.4f}")
    print(f"  增益         : PSNR {mp-bp:+6.2f} dB | SSIM {ms-bs:+.4f}")
    print("  结论         : " + ("✅ 模型打赢基线" if mp > bp else "❌ 未超过基线(需调优)"))


def eval_lowlight(data_root, max_images):
    ll_dir = _pick_dir(data_root, "lol", "lowlight")
    if ll_dir is None:
        print("[!] 未找到低光数据目录"); return
    low_dir = os.path.join(ll_dir, "val", "low")
    high_dir = os.path.join(ll_dir, "val", "high")
    if not (os.path.isdir(low_dir) and os.path.isdir(high_dir)):
        print(f"[!] 未找到 lowlight val: {low_dir} / {high_dir}"); return

    ml = _ml()
    model = ml.get_lowlight_model()
    if model is None:
        print("[!] 无 lowlight 权重 (缺失或损坏)"); return

    lows = _load_images(low_dir)
    if max_images:
        lows = lows[:max_images]
    print(f"[i] 低光验证目录: {low_dir} ({len(lows)} 张) | device={DEVICE}")

    raw_p, raw_s, mod_p, mod_s = [], [], [], []
    used = 0
    for i, lf in enumerate(lows, 1):
        stem = os.path.splitext(os.path.basename(lf))[0]
        hf = None
        for ext in _EXTS:
            cand = os.path.join(high_dir, stem + ext)
            if os.path.exists(cand):
                hf = cand
                break
        if hf is None:
            continue
        low = Image.open(lf).convert("RGB")
        hr = Image.open(hf).convert("RGB")
        if low.size != hr.size:
            continue
        pred = ml.predict_lowlight(low)
        hr_t = TF.to_tensor(hr.resize(pred.size)).unsqueeze(0)
        p, s = _metric(TF.to_tensor(low.resize(pred.size)).unsqueeze(0), hr_t)
        raw_p.append(p); raw_s.append(s)
        p, s = _metric(TF.to_tensor(pred).unsqueeze(0), hr_t)
        mod_p.append(p); mod_s.append(s)
        used += 1
        if i % 10 == 0 or i == len(lows):
            print(f"    ... {i}/{len(lows)} 张", flush=True)

    if used == 0:
        print("[!] 没有可配对的 low/high 样本"); return
    rp, rs = float(np.mean(raw_p)), float(np.mean(raw_s))
    mp, ms = float(np.mean(mod_p)), float(np.mean(mod_s))
    print(f"\n=== LowLight · 验证集 {used} 张 ===")
    print(f"  不处理(low) : PSNR {rp:6.2f} | SSIM {rs:.4f}")
    print(f"  模型输出    : PSNR {mp:6.2f} | SSIM {ms:.4f}")
    print(f"  增益        : PSNR {mp-rp:+6.2f} dB | SSIM {ms-rs:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["sr", "lowlight"], required=True)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--max_images", type=int, default=0,
                    help="只评测前 N 张以快速出结论 (0=全部)")
    a = ap.parse_args()
    if a.task == "sr":
        eval_sr(a.scale, a.data_root, a.max_images)
    else:
        eval_lowlight(a.data_root, a.max_images)


if __name__ == "__main__":
    main()
