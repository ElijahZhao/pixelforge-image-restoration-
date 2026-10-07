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
  results/train_log_<task>_<model>_scale<n>.csv (epoch, train_loss, val_psnr, val_ssim)
  models/ckpt/<task>_<model>_scale<n>_last.pt   (resume point, written every --save_every)

Note: for low-light, ``<model>`` is whatever ``--model`` you passed (default
``srcnn``), and ``<scale>`` is ``--scale`` (default 2), e.g.
``models/lowlight_srcnn_scale2_best.pth``.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torchvision import models
from torchvision.models import VGG16_Weights

from models import build_model
from datasets import get_dataloader
from metrics import evaluate_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Single seed for the whole run. `torch.manual_seed` alone is NOT enough: the
# dataset's random crop / flip augmentation calls `torch.rand` inside DataLoader
# worker processes (whose seeds are derived per worker), and numpy/random are
# used elsewhere. Seeding all four sources + wiring a `generator`/`worker_init_fn`
# into the loaders (see `seed_everything` + `get_dataloader`'s `seed` arg) is what
# makes a run reproducible. Without this, two runs with identical flags diverge.
SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ImageNet normalization stats: REQUIRED for ImageNet-pretrained VGG features.
# Without this, feeding raw [0,1] tensors drives >90% of VGG activations to zero
# and the perceptual loss becomes meaningless.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 features.

    Uses ``features[:30]``, which ends at ``ReLU5_3`` (the SRGAN "VGG54" choice,
    Ledig et al. 2017). Inputs are ImageNet-normalized before feature extraction;
    skipping this step was a real defect (94% of activations collapsed to zero).
    """

    def __init__(self):
        super().__init__()
        # `pretrained=True` is the legacy API (deprecated since torchvision 0.13,
        # removed in 0.15+). Use the explicit weights enum, which is the supported
        # interface and pins the exact ImageNet checkpoint we normalize for.
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:30].eval().to(DEVICE)
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
        # loss only works when inputs already live on DEVICE; a latent bug that
        # passed CPU-only CI but broke on the training GPU.
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
    seed_everything(SEED)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("models/ckpt", exist_ok=True)

    model = build_model(args.task, scale=args.scale,
                        advanced=(args.model == "generator")).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.task}/{args.model}] params={n_params/1e6:.2f}M device={DEVICE}")

    # `seed=SEED` gives the shuffle order and every augmentation RNG a fixed
    # starting point, so re-running with the same flags reproduces the run.
    train_loader = get_dataloader(args.task, args.data_root, "train",
                                  batch_size=args.batch_size, scale=args.scale,
                                  seed=SEED)
    # Validation reuses the training loader's deterministic path (`seed=SEED`):
    # the SR/LOL val splits have fewer images than one batch, so `shuffle` and
    # `drop_last` never apply, and what remains is exactly the guarantee we want
    # -- `_seed_worker` fixes every worker RNG, so the metric cannot drift
    # between epochs for reasons unrelated to the weights.
    val_loader = get_dataloader(args.task, args.data_root, "val",
                                batch_size=args.batch_size, scale=args.scale,
                                seed=SEED)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = GradScaler(enabled=(DEVICE == "cuda"))
    percep = VGGPerceptualLoss() if args.perceptual else None

    best_psnr = -1.0
    start_epoch = 1
    # The scale is part of the filename: SR x2 and x4 are different models with
    # different objectives, and without it the second run would silently append
    # its epochs onto the first one's CSV, producing a log whose numbers come
    # from two unrelated trainings.
    log_path = f"results/train_log_{args.task}_{args.model}_scale{args.scale}.csv"
    # Record the training objective and the best-checkpoint criterion in the
    # same artifact, so the two can never silently diverge.
    best_criterion = "val_psnr"  # best checkpoint is selected by validation PSNR
    objective = (f"{args.w_pixel}*charbonnier + {args.w_percep}*vgg_perceptual"
                 if args.perceptual else "charbonnier")
    resume_path = f"models/ckpt/{args.task}_{args.model}_scale{args.scale}_last.pt"

    # ---- Resume support ---------------------------------------------------- #
    # GPU boxes (AutoDL and friends) get reclaimed, and a 200-epoch run that
    # restarts from zero each time may never finish. The optimizer/scheduler
    # state is what makes a resume actually *resume* rather than restart with a
    # warm-looking but wrong LR (the cosine schedule is a function of the epoch,
    # so restoring only the weights would silently give the model a high LR at
    # epoch 150 and undo its convergence).
    if args.resume:
        # `--resume` with no checkpoint is almost always a typo'd task/model/scale
        # (the path is derived from them), and silently starting a fresh run would
        # look like a successful resume while actually training a new model from
        # scratch. Fail loudly instead.
        if not os.path.exists(resume_path):
            raise SystemExit(
                f"--resume: no checkpoint at {resume_path}. "
                f"Check --task/--model/--scale, or drop --resume to start fresh."
            )
        ckpt = torch.load(resume_path, map_location=DEVICE)
        # A checkpoint silently loaded into a DIFFERENT configuration produces a
        # plausible-looking but meaningless run, so validate the identity first.
        if (ckpt.get("task"), ckpt.get("model"), ckpt.get("scale")) != (
                args.task, args.model, args.scale):
            raise SystemExit(
                f"--resume: {resume_path} is for "
                f"{ckpt.get('task')}/{ckpt.get('model')}/x{ckpt.get('scale')}, "
                f"not {args.task}/{args.model}/x{args.scale}."
            )
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt["best_psnr"]
        print(f"[resume] 从 {resume_path} 恢复：第 {start_epoch} 轮继续，"
              f"best PSNR {best_psnr:.2f}")

    # Append when resuming so the log stays one continuous series; truncate only
    # on a fresh start.
    log_mode = "a" if start_epoch > 1 else "w"
    with open(log_path, log_mode, newline="") as f:
        if log_mode == "w":
            w = csv.writer(f)
            w.writerow([f"# objective={objective}"])
            w.writerow([f"# best_criterion={best_criterion}"])
            w.writerow([f"# perceptual_normalized={bool(args.perceptual)}"])
            w.writerow(["epoch", "train_loss", "val_psnr", "val_ssim", "time_s"])

    for epoch in range(start_epoch, args.epochs + 1):
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
                    # term. Weights are logged so the two terms' magnitudes can
                    # be inspected during training.
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

        # Rolling resume point, written every epoch right after validation. Kept
        # separate from the best checkpoint (which holds weights only): this one
        # carries the full optimizer/scheduler/scaler state needed to continue.
        if args.save_every and epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch, "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if DEVICE == "cuda" else None,
                "best_psnr": best_psnr, "task": args.task,
                "model": args.model, "scale": args.scale,
            }, resume_path)
            print(f"  -> resume point saved: {resume_path} (epoch {epoch})")

    print(f"Training finished. Best val PSNR: {best_psnr:.2f}")


# Default loss weights (only used when --perceptual is on). Johnson et al. 2016
# use a ~1:0.006 pixel:perceptual ratio after proper normalization.
#
# These live as module constants (not inline argparse defaults) so the unit
# test that checks the pixel/perceptual balance can import the REAL values
# instead of re-typing them. Previously the test hard-coded 1.0/0.006, so it
# kept passing even if the defaults here were changed (or broken).
DEFAULT_W_PIXEL = 1.0
DEFAULT_W_PERCEP = 0.006


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
    p.add_argument("--resume", action="store_true",
                   help="continue from models/ckpt/<task>_<model>_scale<n>_last.pt")
    p.add_argument("--save_every", type=int, default=10,
                   help="write a resume point every N epochs (0 = never)")
    p.add_argument("--w_pixel", type=float, default=DEFAULT_W_PIXEL,
                   help="weight for the pixel (Charbonnier) term")
    p.add_argument("--w_percep", type=float, default=DEFAULT_W_PERCEP,
                   help="weight for the VGG perceptual term")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
