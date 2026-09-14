"""Image quality metrics: PSNR and SSIM (PyTorch, batched).

Used both during validation and to build the comparison table in the README /
website Method page.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio (dB), averaged over the batch."""
    mse = F.mse_loss(pred, target, reduction="mean")
    if mse == 0:
        return torch.tensor(float("inf"))
    return 10.0 * torch.log10(max_val * max_val / mse)


def _gaussian_window(size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    import math
    coords = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) @ g.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0)
    return window.expand(channels, 1, size, size).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0,
         window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Structural Similarity Index, averaged over the batch."""
    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma, channels, pred.device, pred.dtype)
    pad = window_size // 2
    mu_pred = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_tgt = F.conv2d(target, window, padding=pad, groups=channels)
    mu_pred_sq = mu_pred.pow(2)
    mu_tgt_sq = mu_tgt.pow(2)
    mu_pred_tgt = mu_pred * mu_tgt
    sigma_pred_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_pred_sq
    sigma_tgt_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu_tgt_sq
    sigma_pred_tgt = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_pred_tgt

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    ssim_map = ((2 * mu_pred_tgt + c1) * (2 * sigma_pred_tgt + c2)) / (
        (mu_pred_sq + mu_tgt_sq + c1) * (sigma_pred_sq + sigma_tgt_sq + c2)
    )
    return ssim_map.mean()


@torch.no_grad()
def evaluate_batch(pred: torch.Tensor, target: torch.Tensor) -> dict:
    return {
        "psnr": psnr(pred, target).item(),
        "ssim": ssim(pred, target).item(),
    }


if __name__ == "__main__":
    a = torch.rand(4, 3, 64, 64)
    b = a + 0.01 * torch.rand(4, 3, 64, 64)
    print(evaluate_batch(a, b))
