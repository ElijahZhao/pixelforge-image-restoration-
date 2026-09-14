"""Unified training script for SR and Low-Light Enhancement.

Run on a free GPU (Kaggle / Colab). Examples
-------------------------------------------
Super-Resolution (SRCNN baseline, x2):
    python train/train.py --task sr --model srcnn --scale 2 \
        --data_root data --epochs 100 --batch_size 16 --lr 1e-3

Super-Resolution (advanced SRResNet generator, x4, perceptual loss):
    python train/train.py --task sr --model generator --scale 4 \
        --data_root data --epochs 200 --batch_size 8 --lr 1e-4 --perceptual

Low-Light Enhancement:
    python train/train.py --task lowlight --data_root data \
        --epochs 200 --batch_size 8 --lr 2e-4

Outputs
-------
  models/<task>_<model>_scale<scale>_best.pth   (best validation checkpoint)
  results/train_log_<task>.csv                  (epoch, train_loss, val_psnr, val_ssim)
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torchvision import models

from models import build_model
from datasets import get_dataloader
from metrics import evaluate_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 feature maps (relu5_4)."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:30].eval().to(DEVICE)
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(self.vgg(pred), self.vgg(target))


def l1_charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3):
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def train(args):
    torch.manual_seed(42)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    model = build_model(args.task, scale=args.scale,
                        advanced=(args.model == "generator")).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.task}/{args.model}] params={n_params/1e6:.2f}M device={DEVICE}")

    train_loader = get_dataloader(args.task, args.data_root, "train",
                                  batch_size=args.batch_size, scale=args.scale)
    val_loader = get_dataloader(args.task, args.data_root, "val",
                                batch_size=args.batch_size, scale=args.scale)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = GradScaler(enabled=(DEVICE == "cuda"))
    percep = VGGPerceptualLoss() if args.perceptual else None

    best_psnr = -1.0
    log_path = f"results/train_log_{args.task}_{args.model}.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_psnr", "val_ssim", "time_s"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        t0 = time.time()
        for lr, hr in train_loader:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            optimizer.zero_grad()
            with autocast(enabled=(DEVICE == "cuda")):
                out = model(lr)
                loss = l1_charbonnier(out, hr)
                if percep is not None:
                    loss = 0.01 * loss + percep(out, hr)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * lr.size(0)
        scheduler.step()

        # Validation.
        model.eval()
        psnrs, ssims = [], []
        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(DEVICE), hr.to(DEVICE)
                out = model(lr)
                m = evaluate_batch(out.cpu(), hr.cpu())
                psnrs.append(m["psnr"])
                ssims.append(m["ssim"])
        val_psnr, val_ssim = sum(psnrs) / len(psnrs), sum(ssims) / len(ssims)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d} | loss={running/len(train_loader.dataset):.4f} "
              f"| val PSNR={val_psnr:.2f} SSIM={val_ssim:.4f} | {elapsed:.1f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{running/len(train_loader.dataset):.4f}",
                                    f"{val_psnr:.2f}", f"{val_ssim:.4f}", f"{elapsed:.1f}"])

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            ckpt = f"models/{args.task}_{args.model}_scale{args.scale}_best.pth"
            torch.save({"state_dict": model.state_dict(), "scale": args.scale,
                        "model": args.model}, ckpt)
            print(f"  -> saved best checkpoint: {ckpt}")

    print("Training finished. Best val PSNR: {:.2f}".format(best_psnr))


def parse_args():
    p = argparse.ArgumentParser(description="Train SR / Low-Light models")
    p.add_argument("--task", choices=["sr", "lowlight"], required=True)
    p.add_argument("--model", choices=["srcnn", "generator"], default="srcnn")
    p.add_argument("--scale", type=int, default=2, choices=[2, 4])
    p.add_argument("--data_root", default="data")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--perceptual", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
