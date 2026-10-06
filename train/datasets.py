"""Dataset loaders for Super-Resolution (DIV2K) and Low-Light Enhancement (LOL).

These loaders expect the standard public dataset layouts:

  Super-Resolution (DIV2K):
    data/div2k/
      train/  (HR images, e.g. 0801.png ...)
      val/
    We synthesise LR by bicubic downscaling on the fly (scale x2/x4).

  Low-Light (LOL-v1):
    data/lol/
      train/
        low/   <low-light images>
        high/  <normal-light images, same filenames>
      val/
        low/
        high/

You can download these datasets for free:
  DIV2K : https://data.vision.ee.ethz.ch/cvl/DIV2K/  (use DIV2K_train_HR)
  LOL   : https://github.com/weichen582/RetinexNet  (LOLdataset_v1)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


class SuperResolutionDataset(Dataset):
    """HR images; LR is produced by bicubic downscaling at load time."""

    def __init__(self, root: str, split: str = "train", scale: int = 2,
                 crop_size: int = 96, augment: bool = True):
        self.scale = scale
        self.crop_size = crop_size
        self.augment = augment and split == "train"
        base = Path(root) / split
        if not base.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {base}. Download DIV2K (SR) or "
                f"LOL-v1 (low-light) and place it under {root}; see data/README.md."
            )
        entries = sorted(p for p in base.iterdir() if p.is_file())
        if entries and entries[0].is_dir():
            # Nested layout: e.g. split contains sub-directories of images.
            entries = sorted(base.rglob("*"))
        self.files = [p for p in entries if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        if not self.files:
            raise RuntimeError(
                f"No images found in {base} (checked .png/.jpg/.jpeg)."
            )

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, idx: int) -> Image.Image:
        img = Image.open(self.files[idx]).convert("RGB")
        w, h = img.size
        if self.augment:
            # Training: a fixed-size patch at a RANDOM position, plus flips.
            if min(w, h) < self.crop_size:
                img = TF.resize(img, [self.crop_size, self.crop_size])
                w, h = img.size
            i = torch.randint(0, w - self.crop_size + 1, (1,)).item() if w > self.crop_size else 0
            j = torch.randint(0, h - self.crop_size + 1, (1,)).item() if h > self.crop_size else 0
            img = TF.crop(img, j, i, self.crop_size, self.crop_size)
            if torch.rand(1) < 0.5:
                img = TF.hflip(img)
            if torch.rand(1) < 0.5:
                img = TF.vflip(img)
        else:
            # Evaluation: a fixed-size patch at a FIXED (centre) position.
            #
            # The crop position used to be drawn from `torch.randint` regardless
            # of `augment`, so every validation epoch scored a different crop of
            # every image. Measured on the DIV2K val split, that alone moved the
            # reported PSNR by 7.35 dB peak-to-peak, while 200 epochs of real
            # training moved it +0.46 dB -- the metric was ~16:1 noise-to-signal,
            # so the "best" checkpoint was largely chosen by crop luck. Centre
            # cropping keeps the patch size (and therefore the compute) identical
            # while making the number comparable across epochs.
            img = TF.crop(img, (h - self.crop_size) // 2, (w - self.crop_size) // 2,
                          self.crop_size, self.crop_size)
        return img

    def __getitem__(self, idx: int):
        hr = TF.to_tensor(self._load(idx))
        lr_size = [max(1, hr.shape[-2] // self.scale),
                   max(1, hr.shape[-1] // self.scale)]
        lr = TF.resize(hr, lr_size, interpolation=TF.InterpolationMode.BICUBIC)
        return lr, hr


class LowLightDataset(Dataset):
    """Paired low/high images for low-light enhancement.

    Robustness: the loader validates that low/high are paired by *filename stem*
    (not just sorted order) and, crucially, that the two images share the same
    size. Mismatched sizes used to load fine but then crop low and high from
    *different spatial regions*: a silent misalignment that poisons the
    supervision signal. Such samples are skipped with a warning rather than
    silently corrupting training.
    """

    def __init__(self, root: str, split: str = "train", crop_size: int = 128,
                 augment: bool = True):
        self.crop_size = crop_size
        self.augment = augment and split == "train"
        low_dir = Path(root) / split / "low"
        high_dir = Path(root) / split / "high"
        if not low_dir.exists() or not high_dir.exists():
            raise FileNotFoundError(
                f"Expected paired dirs {low_dir} and {high_dir}. "
                f"Download LOL-v1; see data/README.md."
            )

        # Pair by filename stem, not by sorted position.
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        high_by_stem = {p.stem: p for p in sorted(high_dir.glob("*"))
                        if p.suffix.lower() in exts}
        pairs, skipped = [], 0
        for lp in sorted(low_dir.glob("*")):
            if lp.suffix.lower() not in exts:
                continue
            hp = high_by_stem.get(lp.stem)
            if hp is None:
                skipped += 1
                continue
            pairs.append((lp, hp))
        if skipped:
            print(f"[LowLightDataset/{split}] WARNING: skipped {skipped} low "
                  f"image(s) with no matching high filename.")

        self.low_files, self.high_files = [], []
        size_mismatch = 0
        for lp, hp in pairs:
            try:
                with Image.open(lp) as li, Image.open(hp) as hi:
                    lsz, hsz = li.size, hi.size
            except Exception as e:  # noqa: BLE001
                print(f"[LowLightDataset/{split}] WARNING: unreadable pair "
                      f"{lp.name}: {type(e).__name__}; skipped.")
                continue
            if lsz != hsz:
                size_mismatch += 1
                print(f"[LowLightDataset/{split}] WARNING: size mismatch "
                      f"{lp.name} {lsz} vs {hp.name} {hsz}; skipped (would "
                      f"misalign the crop).")
                continue
            self.low_files.append(lp)
            self.high_files.append(hp)
        if not self.low_files:
            raise RuntimeError(
                f"No valid low/high pairs in {root}/{split}. "
                f"({len(pairs)} pairs, {size_mismatch} skipped for size mismatch.)"
            )

    def __len__(self) -> int:
        return len(self.low_files)

    def __getitem__(self, idx: int):
        low = Image.open(self.low_files[idx]).convert("RGB")
        high = Image.open(self.high_files[idx]).convert("RGB")
        w, h = low.size
        if self.augment:
            # Random crop only for training.
            if min(w, h) > self.crop_size:
                i = torch.randint(0, w - self.crop_size + 1, (1,)).item()
                j = torch.randint(0, h - self.crop_size + 1, (1,)).item()
            else:
                i = j = 0
                low = TF.resize(low, [self.crop_size, self.crop_size])
                high = TF.resize(high, [self.crop_size, self.crop_size])
                w = h = self.crop_size
            # A square patch is fine HERE: the crop is random, so no spatial
            # region is systematically favoured, and low/high stay aligned.
            low = TF.crop(low, j, i, self.crop_size, self.crop_size)
            high = TF.crop(high, j, i, self.crop_size, self.crop_size)
            if torch.rand(1) < 0.5:
                low, high = TF.hflip(low), TF.hflip(high)
        else:
            # Deterministic evaluation: keep the FULL image at its native aspect
            # ratio, so the metric is reproducible and measured on the same
            # pixels as `scripts/eval_baseline.py`.
            #
            # This branch used to squash every image to a crop_size square. LOL-v1
            # is 400x600, so that compressed width to 21% and height to 32% --
            # a different picture, not a smaller one. Training-time validation
            # and the standalone evaluator were therefore measuring different
            # things and their numbers could not be cross-checked. Only images
            # smaller than crop_size are scaled, and then uniformly.
            if min(w, h) < self.crop_size:
                ratio = self.crop_size / min(w, h)
                low = TF.resize(low, [round(h * ratio), round(w * ratio)])
                high = TF.resize(high, [round(h * ratio), round(w * ratio)])
        return TF.to_tensor(low), TF.to_tensor(high)


def _seed_worker(worker_id: int) -> None:
    """Give every DataLoader worker a deterministic, distinct RNG state.

    PyTorch seeds each worker by deriving from the base seed, but ``numpy`` and
    the stdlib ``random`` module inside the worker are NOT reseeded by default,
    and our augmentation uses ``torch.rand``. Seeding all three here (plus making
    the base seed come from a fixed ``generator``) is what actually removes the
    run-to-run divergence.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(task: str, data_root: str, split: str, batch_size: int = 16,
                   scale: int = 2, num_workers: int = 4, seed: int | None = None):
    if task == "sr":
        ds = SuperResolutionDataset(f"{data_root}/div2k", split, scale=scale)
    elif task == "lowlight":
        ds = LowLightDataset(f"{data_root}/lol", split)
    else:
        raise ValueError(task)

    # When a seed is given, the shuffle order and worker RNGs become reproducible.
    # `pin_memory` is only meaningful (and only legal without warnings) when CUDA
    # is actually available; on a CPU-only box it just prints a warning per epoch.
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=(split == "train"),
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
        generator=generator,
        worker_init_fn=_seed_worker if seed is not None else None,
    )
