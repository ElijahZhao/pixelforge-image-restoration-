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

Seed matrix (one run per seed, isolated under --tag so nothing overwrites):
    for s in 42 43 44 45 46; do
      python train/train.py --task sr --model generator --scale 4 \
          --data_root data --epochs 200 --batch_size 8 --lr 1e-4 \
          --seed $s --tag seeds/s$s
    done

Outputs
-------
  models/[<tag>/]<task>_<model>_scale<scale>_best.pth   (best validation checkpoint)
  results/[<tag>/]train_log_<task>_<model>_scale<n>.csv (epoch, train_loss, val_psnr, ...)
  results/[<tag>/]config_<task>_<model>_scale<n>.json   (full args + environment)
  models/[<tag>/]ckpt/<task>_<model>_scale<n>_last.pt   (resume point, every --save_every)

Note: for low-light, ``<model>`` is whatever ``--model`` you passed (default
``srcnn``), and ``<scale>`` is ``--scale`` (default 2), e.g.
``models/lowlight_srcnn_scale2_best.pth``.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torchvision import models
from torchvision.models import VGG16_Weights

from models import build_model
from datasets import DEFAULT_CROP, get_dataloader
from metrics import aggregate, evaluate_batch
from runenv import dumps as json_dumps, env_snapshot

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The seed used when `--seed` is not given. Until this run it was a module
# constant, which made the seed un-overridable: a seed matrix (5 seeds) was
# impossible without editing source, and editing source changes the code that
# produced the number.
DEFAULT_SEED = 42
# Alias kept for anything written against the old constant. It now means
# "default", not "the seed".
SEED = DEFAULT_SEED

# Training patch size per task, mirroring the dataset classes' own defaults
# (SR crops the HR image to 96 px; the low-light U-Net downsamples by 32 five
# times, so its patch must be a multiple of 32 -> 128). Imported rather than
# restated: the loader, the class signature and --crop_size's help text all read
# from `datasets.DEFAULT_CROP`, so they cannot drift apart.
TASK_CROP_DEFAULT = DEFAULT_CROP


def seed_everything(seed: int = DEFAULT_SEED, deterministic: bool = True,
                    strict: bool = False) -> None:
    """Seed every RNG a run touches, and pin cuDNN's algorithm choice.

    `torch.manual_seed` alone is NOT enough: the dataset's random crop / flip
    augmentation calls `torch.rand` inside DataLoader worker processes (whose
    seeds are derived per worker), and numpy/random are used elsewhere. Seeding
    all four sources + wiring a `generator`/`worker_init_fn` into the loaders
    (see `get_dataloader`'s `seed` arg) is what makes a run reproducible.
    Without this, two runs with identical flags diverge.

    `deterministic` additionally fixes cuDNN's autotuner. `benchmark=True` lets
    cuDNN pick a different convolution algorithm per input shape *by timing
    them*, so two runs of the same script can accumulate in different orders
    and drift in the last digits -- which is exactly the noise a seed matrix is
    trying to measure. `strict` goes further and bans any op whose kernel is
    not deterministic; it can raise at run time (antialiased resize backward on
    CUDA is one such op), which is why it is opt-in.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if strict:
        torch.use_deterministic_algorithms(True)


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


def _atomic_save(obj, path: str) -> None:
    """Save ``obj`` to ``path`` so a crash can never leave a partial file.

    ``torch.save(obj, path)`` opens ``path`` and writes into it. If the process
    dies mid-write (AutoDL reclaiming the instance, OOM-kill, Ctrl-C), the file
    left behind is truncated -- and that is precisely the file the next
    ``--auto-resume`` run tries to read, turning one interruption into a
    permanent crash loop. Writing to a sibling temp file and ``os.replace``-ing
    it into place is atomic on POSIX: readers see either the old complete file
    or the new complete one, never a half-written one.
    """
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _run_dir(base: str, tag: str | None) -> str:
    """Resolve an output directory, honouring ``--tag``.

    ``--tag seeds/s43`` puts a run under ``models/seeds/s43/`` instead of
    ``models/``. The seed matrix produces five runs whose filenames are
    otherwise identical, so without isolation they overwrite each other and the
    survivor looks like a finished experiment. Nesting under a tag also makes
    the whole matrix one ``rsync`` away from the archive.
    """
    tag = (tag or "").strip().strip("/")
    return os.path.join(base, tag) if tag else base


def _seed_from_log(path: str) -> int | None:
    """Read ``# seed=N`` from an existing training CSV; ``None`` if absent.

    Used to refuse a run that would overwrite a *different* seed's log. Logs
    written before ``--seed`` existed have no such line, hence the ``None``
    (unknown, not "wrong").
    """
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for _ in range(60):
                line = f.readline()
                if not line:
                    break
                if line.startswith("# seed="):
                    return int(line.strip().split("=", 1)[1])
    except (OSError, ValueError):
        return None
    return None


def train(args):
    seed = args.seed
    seed_everything(seed, deterministic=not args.no_determinism,
                    strict=args.strict_determinism)

    model_dir = _run_dir("models", args.tag)
    res_dir = _run_dir("results", args.tag)
    ckpt_dir = os.path.join(model_dir, "ckpt")
    for d in (model_dir, res_dir, ckpt_dir):
        os.makedirs(d, exist_ok=True)

    # Log the patch size the loader will actually use, not the flag's value:
    # `--crop_size` defaults to None so that each task keeps its own sane
    # default (96 / 128), and a config that recorded `null` would be useless.
    crop_size = args.crop_size or TASK_CROP_DEFAULT[args.task]

    model = build_model(args.task, scale=args.scale,
                        advanced=(args.model == "generator")).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.task}/{args.model}] params={n_params/1e6:.2f}M device={DEVICE} "
          f"seed={seed} crop={crop_size}")

    # `seed=seed` gives the shuffle order and every augmentation RNG a fixed
    # starting point, so re-running with the same flags reproduces the run.
    # `crop_size` is threaded through explicitly: the loader used to always fall
    # back to the dataset class's own default, so the patch size was a magic
    # constant rather than a recorded experimental variable.
    train_loader = get_dataloader(args.task, args.data_root, "train",
                                  batch_size=args.batch_size, scale=args.scale,
                                  crop_size=crop_size, seed=seed)
    # Validation reuses the training loader's deterministic path (`seed=seed`):
    # the SR/LOL val splits have fewer images than one batch, so `shuffle` and
    # `drop_last` never apply, and what remains is exactly the guarantee we want
    # -- `_seed_worker` fixes every worker RNG, so the metric cannot drift
    # between epochs for reasons unrelated to the weights.
    #
    # The validation batch size is a SEPARATE knob from the training one. It used
    # to silently follow `--batch_size`, which made the reported metric a function
    # of a training hyper-parameter: changing the training batch regrouped the
    # validation images and moved the score by up to 11.65 dB on this project's
    # own checkpoints (measured under the old batch-wide-MSE aggregation; under
    # the per-image one the value is invariant by construction).
    # Default 1 keeps each image scored on its own; even at N>1 the per-image
    # metrics are unchanged, and `tests/test_protocol.py` holds that invariant.
    #
    # The validation crop is the CENTRE patch of size `crop_size` (see
    # datasets.SuperResolutionDataset._load), so it must be the same patch the
    # training validation used -- passing the training crop keeps the two
    # consistent and keeps the number comparable across epochs.
    val_loader = get_dataloader(args.task, args.data_root, "val",
                                batch_size=args.val_batch_size, scale=args.scale,
                                crop_size=crop_size, seed=seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = GradScaler(enabled=(DEVICE == "cuda"))
    percep = VGGPerceptualLoss() if args.perceptual else None

    best_psnr = -1.0
    start_epoch = 1
    # The scale is part of the filename: SR x2 and x4 are different models with
    # different objectives, and without it the second run would silently append
    # its epochs onto the first one's CSV, producing a log whose numbers come
    # from two unrelated trainings. (The SEED is not part of the filename; that
    # is what --tag is for, and the guard below stops a cross-seed overwrite.)
    stem = f"{args.task}_{args.model}_scale{args.scale}"
    log_path = os.path.join(res_dir, f"train_log_{stem}.csv")
    config_path = os.path.join(res_dir, f"config_{stem}.json")
    best_path = os.path.join(model_dir, f"{stem}_best.pth")
    resume_path = os.path.join(ckpt_dir, f"{stem}_last.pt")
    # Record the training objective and the best-checkpoint criterion in the
    # same artifact, so the two can never silently diverge.
    best_criterion = "val_psnr"  # best checkpoint is selected by validation PSNR
    objective = (f"{args.w_pixel}*charbonnier + {args.w_percep}*vgg_perceptual"
                 if args.perceptual else "charbonnier")

    # ---- Refuse a cross-seed overwrite ------------------------------------- #
    # `--seed 43` without `--tag` would truncate the CSV produced by seed 42,
    # leaving one file that reads like a complete experiment but is a different
    # one -- and the seed matrix is precisely the experiment whose whole point
    # is that the runs differ. Re-running the SAME seed is a legitimate
    # overwrite (that is how you check determinism), so only a mismatch stops.
    if not (args.resume or args.auto_resume):
        if os.path.exists(log_path):
            prev_seed = _seed_from_log(log_path)
            if prev_seed is not None and prev_seed != seed:
                raise SystemExit(
                    f"{log_path} already holds seed={prev_seed}. Refusing to "
                    f"overwrite it with seed={seed}.\n"
                    f"Pass --tag <name> to keep both (e.g. --tag seeds/s{seed}), "
                    f"or delete the file if the old run is disposable."
                )

    # ---- Config artifact ---------------------------------------------------- #
    # Everything needed to rebuild this exact run: the arguments, the resolved
    # knobs (crop, aggregation), the data counts and the software stack. Written
    # next to the CSV so a number can never travel without its conditions.
    env = env_snapshot()
    config = {
        "run": {
            "task": args.task, "model": args.model, "scale": args.scale,
            "tag": args.tag or None, "seed": seed,
            "crop_size": crop_size,
            "batch_size": args.batch_size, "val_batch_size": args.val_batch_size,
            "epochs": args.epochs, "lr": args.lr,
            "perceptual": bool(args.perceptual),
            "w_pixel": args.w_pixel, "w_percep": args.w_percep,
            "objective": objective, "best_criterion": best_criterion,
            "save_every": args.save_every,
            "params_millions": round(n_params / 1e6, 4),
        },
        "protocol": {
            # The four knobs this project measured to matter. Spelling them out
            # is the point: each one moved the reported dB by >= 0.37 on this
            # repo (see results/README.md), so a table without them is not
            # comparable with any other table.
            "train_crop": "random_position",
            # `val_crop` differs per task and was written as "centre" for both
            # until 2026-11-04. SuperResolutionDataset scores a fixed centre
            # 96x96 patch; LowLightDataset keeps the FULL image so its number
            # stays comparable with `scripts/eval_baseline.py`. The old value was
            # not just imprecise, it was falsifiable: the low-light seed matrix
            # reports 18.510 (best-of-200) and protocol P3 (full image) reports
            # 18.509, while P2 (centre 96) reports 20.569 -- the training log
            # matches the full-image protocol, not the one this field named.
            "val_crop": "centre_96" if args.task == "sr" else "full_native",
            "val_aggregation": "per_image_mean",
            # Spatial extent is `val_crop`; this field only names the channel
            # set and the border treatment, so it must not repeat "full image".
            "metric_space": "RGB, no shave",
            "lr_degradation": "bicubic(antialias=True)",
        },
        "data": {
            "data_root": args.data_root,
            "train_images": len(train_loader.dataset),
            "val_images": len(val_loader.dataset),
            "train_batches_per_epoch": len(train_loader),
        },
        "determinism": {
            "seeded": ["random", "numpy", "torch", "torch.cuda",
                       "dataloader.worker"],
            "cudnn_deterministic": not args.no_determinism,
            "cudnn_benchmark": False,
            "strict_determinism": bool(args.strict_determinism),
        },
        "argv": sys.argv,
        "environment": env,
    }

    # ---- Resume support ---------------------------------------------------- #
    # GPU boxes (AutoDL and friends) get reclaimed, and a 200-epoch run that
    # restarts from zero each time may never finish. The optimizer/scheduler
    # state is what makes a resume actually *resume* rather than restart with a
    # warm-looking but wrong LR (the cosine schedule is a function of the epoch,
    # so restoring only the weights would silently give the model a high LR at
    # epoch 150 and undo its convergence).
    if args.resume or args.auto_resume:
        # `--resume` means "continue, and fail if there is nothing to continue
        # from" -- a missing checkpoint usually means a typo'd task/model/scale,
        # and silently starting fresh would look like a successful resume while
        # actually training a new model.
        #
        # `--auto-resume` means "continue if a checkpoint exists, otherwise
        # start fresh". This is what the retraining script wants: it runs the
        # same command every time an instance gets reclaimed, and on the very
        # first invocation there is legitimately nothing to resume from.
        if not os.path.exists(resume_path):
            if args.resume:
                raise SystemExit(
                    f"--resume: no checkpoint at {resume_path}. "
                    f"Check --task/--model/--scale, or drop --resume to start fresh."
                )
            print(f"[auto-resume] 无断点，从头开始训练")
        else:
            # A resume point can be TRUNCATED: `torch.save` writes in place, so a
            # kill (instance reclaimed / OOM / Ctrl-C) mid-write leaves a partial
            # file. On AutoDL that is exactly the moment we then try to resume --
            # an unhandled load error here would crash the unattended script on
            # every restart, never making progress. Under --auto-resume we treat
            # an unreadable checkpoint as "no checkpoint"; under --resume (manual,
            # explicit) we stop and report, because silently discarding a file the
            # operator pointed at is worse than failing.
            try:
                ckpt = torch.load(resume_path, map_location=DEVICE)
                if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
                    raise ValueError("not a training checkpoint")
            except Exception as exc:  # noqa: BLE001
                if args.resume:
                    raise SystemExit(
                        f"--resume: {resume_path} is unreadable "
                        f"({type(exc).__name__}: {exc}). Delete it to start fresh."
                    ) from exc
                print(f"[auto-resume] {resume_path} 损坏（{type(exc).__name__}），"
                      f"忽略并从头开始训练")
                ckpt = None

            if ckpt is not None:
                # A checkpoint silently loaded into a DIFFERENT configuration
                # produces a plausible-looking but meaningless run, so validate
                # identity first. Checked even under --auto-resume: starting over
                # silently is worse than stopping, because the operator believes
                # the run continued.
                #
                # Seed and crop are part of that identity: resuming seed 42's
                # state under --seed 43 would produce a run that is neither
                # seed, and the seed matrix would quietly contain a duplicate
                # masquerading as an independent sample. Checkpoints written
                # before those fields existed default to the values that
                # produced them (seed 42, the loader's own crop default).
                ck_seed = ckpt.get("seed", DEFAULT_SEED)
                ck_crop = ckpt.get("crop_size", crop_size)
                if (ckpt.get("task"), ckpt.get("model"), ckpt.get("scale"),
                        ck_seed, ck_crop) != (
                        args.task, args.model, args.scale, seed, crop_size):
                    raise SystemExit(
                        f"--resume: {resume_path} is for "
                        f"{ckpt.get('task')}/{ckpt.get('model')}/x{ckpt.get('scale')}"
                        f" seed={ck_seed} crop={ck_crop}, not "
                        f"{args.task}/{args.model}/x{args.scale} "
                        f"seed={seed} crop={crop_size}."
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

    # Written only AFTER the resume identity check above: a rejected resume must
    # not leave a config file that describes a run the log does not belong to.
    # (Observed in testing: failing the seed check still overwrote the JSON, so
    # the CSV said seed=7 and its config said seed=8.)
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(json_dumps(config) + "\n")
    print(f"[config] {config_path}")

    # Append when resuming so the log stays one continuous series; truncate only
    # on a fresh start.
    log_mode = "a" if start_epoch > 1 else "w"
    with open(log_path, log_mode, newline="") as f:
        if log_mode == "w":
            w = csv.writer(f)
            w.writerow([f"# objective={objective}"])
            w.writerow([f"# best_criterion={best_criterion}"])
            w.writerow([f"# perceptual_normalized={bool(args.perceptual)}"])
            # The aggregation rule belongs in the log, not only in the code: a
            # CSV that does not say HOW its metric was averaged cannot be
            # compared with another CSV, and this project has already been bitten
            # by that (11.85 dB between two aggregation choices).
            w.writerow(["# val_aggregation=per_image_mean"])
            w.writerow([f"# val_batch_size={args.val_batch_size}"])
            # ---- Run identity -------------------------------------------- #
            # A log whose CSV does not name its seed, patch size and commit is
            # not evidence: two tables built from "the same" training run can
            # differ by more than the effect being claimed. `# seed=` is also
            # what the cross-seed overwrite guard reads back.
            w.writerow([f"# seed={seed}"])
            w.writerow([f"# crop_size={crop_size}"])
            w.writerow([f"# batch_size={args.batch_size}"])
            w.writerow([f"# lr={args.lr}"])
            w.writerow([f"# epochs={args.epochs}"])
            w.writerow([f"# tag={args.tag or '-'}"])
            git = env.get("git") or {}
            w.writerow([f"# git_commit={git.get('short') or 'unknown'}"
                        f"{' (dirty)' if git.get('dirty') else ''}"])
            w.writerow([f"# torch={env.get('torch')} "
                        f"cuda={env.get('cuda_version')} "
                        f"device={env.get('device')}"])
            w.writerow([f"# config={config_path}"])
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
        #
        # Every image is scored on its own and only then averaged, so the number
        # cannot depend on how images are grouped into batches. The old code took
        # one MSE over the whole batch and averaged those per-batch scalars --
        # a quantity that moved with `--batch_size` (measured spread on this
        # project: 11.85 dB), i.e. the metric measured the dataloader, not the
        # model. See metrics.py's module docstring.
        model.eval()
        psnrs, ssims = [], []
        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(DEVICE), hr.to(DEVICE)
                out = model(lr)
                m = evaluate_batch(out.cpu(), hr.cpu(), per_image=True)
                psnrs.extend(m["psnr_per_image"])
                ssims.extend(m["ssim_per_image"])
        val_psnr, val_ssim = aggregate(psnrs), aggregate(ssims)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d} | loss={running/len(train_loader.dataset):.4f} "
              f"| val PSNR={val_psnr:.2f} SSIM={val_ssim:.4f} | {elapsed:.1f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{running/len(train_loader.dataset):.4f}",
                                    f"{val_psnr:.2f}", f"{val_ssim:.4f}", f"{elapsed:.1f}"])

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            # The best checkpoint carries the conditions that produced it, so a
            # weight file found on disk months later can still be traced back to
            # a seed and a patch size instead of "some run".
            _atomic_save({"state_dict": model.state_dict(), "scale": args.scale,
                          "model": args.model, "task": args.task,
                          "seed": seed, "crop_size": crop_size,
                          "val_psnr": val_psnr, "epoch": epoch,
                          "tag": args.tag or None}, best_path)
            print(f"  -> saved best checkpoint: {best_path}")

        # Rolling resume point, written every epoch right after validation. Kept
        # separate from the best checkpoint (which holds weights only): this one
        # carries the full optimizer/scheduler/scaler state needed to continue.
        if args.save_every and epoch % args.save_every == 0:
            _atomic_save({
                "epoch": epoch, "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if DEVICE == "cuda" else None,
                "best_psnr": best_psnr, "task": args.task,
                "model": args.model, "scale": args.scale,
                # Part of the resume identity check: see the block above.
                "seed": seed, "crop_size": crop_size, "tag": args.tag or None,
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
    p.add_argument("--val_batch_size", type=int, default=1,
                   help="batch size for validation only, independent of --batch_size. "
                        "Default 1: metrics are per-image, so this changes speed but "
                        "must never change the reported value")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--perceptual", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="continue from [models/<tag>/]ckpt/<task>_<model>_scale<n>_last.pt; "
                        "error out if it does not exist")
    p.add_argument("--auto-resume", dest="auto_resume", action="store_true",
                   help="continue if a resume point exists, otherwise start fresh")
    p.add_argument("--save_every", type=int, default=10,
                   help="write a resume point every N epochs (0 = never)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"RNG seed for this run (default {DEFAULT_SEED}). Seeds "
                        f"python/numpy/torch + DataLoader workers. Use --tag too "
                        f"when running several seeds, or they share filenames")
    p.add_argument("--crop_size", type=int, default=None,
                   help="training patch size in pixels (and the size of the "
                        "centre crop used at validation). Defaults per task: "
                        f"sr={TASK_CROP_DEFAULT['sr']}, "
                        f"lowlight={TASK_CROP_DEFAULT['lowlight']}. SR needs a "
                        f"multiple of --scale; low-light needs a multiple of 32")
    p.add_argument("--tag", default="",
                   help="experiment label; outputs go to models/<tag>/ and "
                        "results/<tag>/ instead of the repo root. Use one tag "
                        "per run of a seed matrix (e.g. --tag seeds/s43)")
    p.add_argument("--no-determinism", dest="no_determinism", action="store_true",
                   help="leave cuDNN's autotuner free (benchmark=True). Faster, "
                        "but repeated runs can differ in the last digits")
    p.add_argument("--strict-determinism", dest="strict_determinism",
                   action="store_true",
                   help="also call torch.use_deterministic_algorithms(True). May "
                        "raise on ops without a deterministic kernel (e.g. "
                        "antialiased resize backward on CUDA)")
    p.add_argument("--w_pixel", type=float, default=DEFAULT_W_PIXEL,
                   help="weight for the pixel (Charbonnier) term")
    p.add_argument("--w_percep", type=float, default=DEFAULT_W_PERCEP,
                   help="weight for the VGG perceptual term")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
