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
  results/train_log_<task>_<model>.csv          (epoch, train_loss, val_psnr, val_ssim)

Note: for low-light, ``<model>`` is whatever ``--model`` you passed (default
``srcnn``), and ``<scale>`` is ``--scale`` (default 2), e.g.
``models/lowlight_srcnn_scale2_best.pth``.
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


# ImageNet normalization stats — REQUIRED for ImageNet-pretrained VGG features.
# Without this, feeding raw [0,1] tensors drives >90% of VGG activations to zero
# and the perceptual loss becomes meaningless (see DIAGNOSIS_ROUND3/11/13/15).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 features.

    Uses ``features[:30]``, which ends at ``ReLU5_3`` (the SRGAN "VGG54" choice,
    Ledig et al. 2017) — not ``relu5_4`` as an earlier docstring claimed.
    Inputs are ImageNet-normalized before feature extraction; skipping this step
    was a real defect (94% of activations collapsed to zero).
    """

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:30].eval().to(DEVICE)
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.register_buffer(
            "mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer(
            "std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def _feat(self, x: torch.Tensor) -> torch.Tensor:
        # Align every operand to the VGG backbone's device. The backbone is
        # created on DEVICE (cuda when available), but callers may pass CPU
        # tensors (e.g. unit tests, or pre-to(DEVICE) inputs). Without this the
        # loss only works when inputs already live on DEVICE — a latent bug that
        # passed CPU-only CI but broke on the training GPU (see the audit rounds
        # covering the per-device alignment fix).
        dev = next(self.vgg.parameters()).device
        x = x.to(dev)
        mean = self.mean.to(dev)
        std = self.std.to(dev)
        return self.vgg((x - mean) / std)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(self._feat(pred), self._feat(target))


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
    # F5: record the training objective and the best-checkpoint criterion in the
    # same artifact, so the two can never silently diverge (DIAGNOSIS_ROUND13).
    best_criterion = "val_psnr"  # best checkpoint is selected by validation PSNR
    objective = (f"{args.w_pixel}*charbonnier + {args.w_percep}*vgg_perceptual"
                 if args.perceptual else "charbonnier")
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# objective={objective}"])
        w.writerow([f"# best_criterion={best_criterion}"])
        w.writerow([f"# perceptual_normalized={bool(args.perceptual)}"])
        w.writerow(["epoch", "train_loss", "val_psnr", "val_ssim", "time_s"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        t0 = time.time()
        for lr, hr in train_loader:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            optimizer.zero_grad()
            with autocast(enabled=(DEVICE == "cuda")):
                out = model(lr)
                pix = l1_charbonnier(out, hr)
                if percep is not None:
                    # Explicit, readable weighting. The old code used
                    # ``0.01 * loss + percep`` which silently erased the pixel
                    # term (see the audit rounds covering loss-weighting). Weights
                    # are logged so the
                    # two terms' magnitudes can be inspected during training.
                    per = percep(out, hr)
                    loss = args.w_pixel * pix + args.w_percep * per
                else:
                    loss = pix
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

    print(f"Training finished. Best val PSNR: {best_psnr:.2f}")


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
    # Explicit loss weights (only used when --perceptual is on). Johnson et al.
    # 2016 use a ~1:0.006 pixel:perceptual ratio after proper normalization.
    p.add_argument("--w_pixel", type=float, default=1.0,
                   help="weight for the pixel (Charbonnier) term")
    p.add_argument("--w_percep", type=float, default=0.006,
                   help="weight for the VGG perceptual term")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
