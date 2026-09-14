"""Export a trained checkpoint to TorchScript for the FastAPI inference service.

Usage:
    python train/export.py --checkpoint models/sr_generator_scale4_best.pth \
        --out serve/models/sr_generator_scale4.pt --task sr --scale 4
    python train/export.py --checkpoint models/lowlight_generator_best.pth \
        --out serve/models/lowlight.pt --task lowlight
"""

from __future__ import annotations

import argparse
import os

import torch

from models import build_model


def export(checkpoint: str, out: str, task: str, scale: int):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model_name = ckpt.get("model", "srcnn" if task == "sr" else "generator")
    scale = ckpt.get("scale", scale)
    model = build_model(task, scale=scale, advanced=(model_name == "generator"))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    dummy = torch.rand(1, 3, 64, 64) if task == "lowlight" else torch.rand(1, 3, 16, 16)
    traced = torch.jit.trace(model, dummy)
    traced.save(out)
    print(f"Exported TorchScript model -> {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--task", choices=["sr", "lowlight"], required=True)
    p.add_argument("--scale", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export(args.checkpoint, args.out, args.task, args.scale)
