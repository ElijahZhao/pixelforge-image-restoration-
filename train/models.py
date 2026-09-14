"""Model definitions for the image restoration / enhancement project.

Two tasks are supported:
  1. Single Image Super-Resolution (SR)  -> SRCNN (baseline) and a lightweight
     SRResNet/ESRGAN-style generator (advanced).
  2. Low-Light Image Enhancement (LLIE)  -> a small U-Net style Retinex network.

All models are plain PyTorch (``torch.nn.Module``) so they can be trained on
Kaggle/Colab free GPUs and exported for the FastAPI inference service.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Super-Resolution
# ---------------------------------------------------------------------------

class SRCNN(nn.Module):
    """SRCNN (Dong et al., ECCV 2014) — the classic 3-layer SR network.

    A deliberately simple, fully explainable baseline. Great for showing the
    fundamentals: feature extraction -> non-linear mapping -> reconstruction.
    """

    def __init__(self, scale: int = 2, num_channels: int = 3):
        super().__init__()
        self.scale = scale
        # Up-sample the low-res input with bicubic before the conv layers.
        self.upscale = nn.Upsample(scale_factor=scale, mode="bicubic", align_corners=False)
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upscale(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return torch.clamp(x, 0.0, 1.0)


class ResidualBlock(nn.Module):
    """A basic residual block used by SRResNet/ESRGAN-style generators."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class SRGenerator(nn.Module):
    """Lightweight SRResNet-style generator (Ledig et al., CVPR 2017).

    Deeper than SRCNN and trained with perceptual loss for sharper results.
    Used as the "advanced" SR model in this project.
    """

    def __init__(self, scale: int = 2, num_channels: int = 3, num_blocks: int = 8):
        super().__init__()
        self.scale = scale
        self.entry = nn.Conv2d(num_channels, 64, 3, 1, 1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])
        self.mid = nn.Conv2d(64, 64, 3, 1, 1)
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * scale * scale, 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.PReLU(),
        )
        self.exit = nn.Conv2d(64, num_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.entry(x)
        out = self.res_blocks(feats)
        out = self.mid(out) + feats
        out = self.upsample(out)
        out = self.exit(out)
        return torch.clamp(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 2. Low-Light Image Enhancement (LLIE)
# ---------------------------------------------------------------------------

class LowLightUNet(nn.Module):
    """A small U-Net style network for low-light image enhancement.

    Inspired by Retinex-theory decomposition (illumination + reflectance),
    but implemented as a direct learning-based mapping from a low-light image
    to its enhanced version. Lightweight enough to train on a single free GPU.
    """

    def __init__(self, num_channels: int = 3):
        super().__init__()

        def down(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, 1, 1),
                nn.ReLU(inplace=True),
            )

        def up(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )

        self.enc1 = down(num_channels, 32)
        self.enc2 = down(32, 64)
        self.enc3 = down(64, 128)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True)
        )
        self.dec3 = up(128, 64)
        self.dec2 = up(64 + 64, 32)
        self.dec1 = up(32 + 32, 32)
        # After dec1 (32ch) we concatenate the original input (num_channels),
        # so the final conv sees 32 + num_channels channels.
        self.final = nn.Conv2d(32 + num_channels, num_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)   # 32
        e2 = self.enc2(e1)  # 64
        e3 = self.enc3(e2)  # 128
        b = self.bottleneck(e3)
        d3 = self.dec3(b)              # 64
        d3 = torch.cat([d3, e2], dim=1)  # 128
        d2 = self.dec2(d3)             # 32
        d2 = torch.cat([d2, e1], dim=1)  # 64
        d1 = self.dec1(d2)             # 32
        d1 = torch.cat([d1, x], dim=1)  # 64
        out = self.final(d1)
        # Residual learning: predict an enhancement residual added to input.
        return torch.clamp(x + out, 0.0, 1.0)


def build_model(task: str, scale: int = 2, advanced: bool = False) -> nn.Module:
    """Factory: build a model for ``task`` ('sr' or 'lowlight')."""
    if task == "sr":
        return SRGenerator(scale=scale) if advanced else SRCNN(scale=scale)
    if task == "lowlight":
        return LowLightUNet()
    raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    # Quick shape sanity check.
    x = torch.rand(2, 3, 64, 64)
    for task, model in [("sr", SRCNN(2)), ("sr", SRGenerator(2)), ("lowlight", LowLightUNet())]:
        y = model(x if task == "lowlight" else F.interpolate(x, scale_factor=0.5, mode="bilinear"))
        print(f"{task}: {list(y.shape)}")
    print("All model shapes OK.")
