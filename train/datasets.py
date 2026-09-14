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

import os
from pathlib import Path
from typing import Callable, Optional

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
                f"LOL-v1 (low-light) and place it under {root} — see data/README.md."
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
        # Crop a fixed HR patch.
        w, h = img.size
        if min(w, h) < self.crop_size:
            img = TF.resize(img, [self.crop_size, self.crop_size])
            w, h = img.size
        i = torch.randint(0, w - self.crop_size + 1, (1,)).item() if w > self.crop_size else 0
        j = torch.randint(0, h - self.crop_size + 1, (1,)).item() if h > self.crop_size else 0
        img = TF.crop(img, j, i, self.crop_size, self.crop_size)
        if self.augment:
            if torch.rand(1) < 0.5:
                img = TF.hflip(img)
            if torch.rand(1) < 0.5:
                img = TF.vflip(img)
        return img

    def __getitem__(self, idx: int):
        hr = TF.to_tensor(self._load(idx))
        lr = TF.resize(hr, [self.crop_size // self.scale, self.crop_size // self.scale],
                      interpolation=TF.InterpolationMode.BICUBIC)
        return lr, hr


class LowLightDataset(Dataset):
    """Paired low/high images for low-light enhancement."""

    def __init__(self, root: str, split: str = "train", crop_size: int = 128,
                 augment: bool = True):
        self.crop_size = crop_size
        self.augment = augment and split == "train"
        low_dir = Path(root) / split / "low"
        high_dir = Path(root) / split / "high"
        self.low_files = sorted(low_dir.glob("*"))
        self.high_files = sorted(high_dir.glob("*"))

    def __len__(self) -> int:
        return len(self.low_files)

    def __getitem__(self, idx: int):
        low = Image.open(self.low_files[idx]).convert("RGB")
        high = Image.open(self.high_files[idx]).convert("RGB")
        # Random crop.
        w, h = low.size
        if min(w, h) > self.crop_size:
            i = torch.randint(0, w - self.crop_size + 1, (1,)).item()
            j = torch.randint(0, h - self.crop_size + 1, (1,)).item()
            low = TF.crop(low, j, i, self.crop_size, self.crop_size)
            high = TF.crop(high, j, i, self.crop_size, self.crop_size)
        else:
            low = TF.resize(low, [self.crop_size, self.crop_size])
            high = TF.resize(high, [self.crop_size, self.crop_size])
        if self.augment:
            if torch.rand(1) < 0.5:
                low, high = TF.hflip(low), TF.hflip(high)
        return TF.to_tensor(low), TF.to_tensor(high)


def get_dataloader(task: str, data_root: str, split: str, batch_size: int = 16,
                   scale: int = 2, num_workers: int = 4):
    if task == "sr":
        ds = SuperResolutionDataset(f"{data_root}/div2k", split, scale=scale)
    elif task == "lowlight":
        ds = LowLightDataset(f"{data_root}/lol", split)
    else:
        raise ValueError(task)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=(split == "train"),
        num_workers=num_workers, pin_memory=True, drop_last=(split == "train"),
    )
