"""Image quality metrics: PSNR and SSIM (PyTorch, batched).

Used both during validation and to build the comparison table in the README /
website Method page.

Aggregation policy (important -- read before changing anything)
---------------------------------------------------------------
Every metric in this module is computed **per image** first and only then
averaged across images, with each image weighted equally.

This matters more than it sounds. The previous implementation built one MSE over
the whole batch::

    mse = F.mse_loss(pred, target, reduction="mean")   # over B*C*H*W

and ``train.py`` then averaged those per-batch scalars over the validation
loader. With that design the reported value is a function of the validation
batch size -- the same weights and the same images score differently depending
on how they happen to be grouped. Measured on this project's own checkpoints the
spread reached **11.85 dB**, i.e. larger than any improvement the model makes.

Per-image aggregation removes that dependency structurally: the value no longer
has any way to see the batch. ``tests/test_protocol.py::test_psnr_is_batch_invariant``
and ``::test_ssim_is_batch_invariant`` lock it in.

When reporting in a paper, state the policy explicitly, e.g. "PSNR is the
arithmetic mean of per-image PSNR over the validation split (RGB, full channel,
no border shave)".
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Below this MSE the log term is dominated by float rounding, and the true limit
# is +inf anyway, so we saturate to inf rather than emit a meaningless 10*log10.
_ZERO_MSE_EPS = 1e-12


def psnr_per_image(pred: torch.Tensor, target: torch.Tensor,
                   max_val: float = 1.0) -> torch.Tensor:
    """PSNR for each image in the batch, shape ``(B,)``.

    Each image's MSE is taken over its own channels and pixels, so the result
    does not depend on what else is in the batch.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.dim() < 2:
        raise ValueError(f"expected at least (B, C, ...) tensors, got {tuple(pred.shape)}")

    diff = pred - target
    mse = diff.pow(2).flatten(1).mean(dim=1)  # (B,) -- reduce C*H*W, keep B separate
    finite = 10.0 * torch.log10(torch.full_like(mse, max_val * max_val) / mse.clamp_min(_ZERO_MSE_EPS))
    # An exactly matching image has MSE == 0 and PSNR == +inf; that is the
    # mathematically correct limit and callers can still compare it.
    return torch.where(mse <= _ZERO_MSE_EPS, torch.full_like(mse, float("inf")), finite)


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0,
         reduction: str = "mean"):
    """PSNR in dB, averaged over the batch by default.

    Returns ``+inf`` for a perfect match (see ``psnr_per_image``).

    ``reduction="none"`` returns the per-image ``(B,)`` tensor; use that when the
    caller wants to aggregate over several batches itself (``train.py`` does).
    """
    per_image = psnr_per_image(pred, target, max_val=max_val)
    if reduction == "none":
        return per_image
    if reduction != "mean":
        raise ValueError(f"unknown reduction {reduction!r}; use 'mean' or 'none'")
    return per_image.mean()


def _gaussian_window(size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) @ g.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0)
    return window.expand(channels, 1, size, size).contiguous()


def ssim_per_image(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0,
                   window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """SSIM for each image in the batch, shape ``(B,)``."""
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
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
    # Per image: average over channels and pixels, keep the batch axis separate.
    return ssim_map.mean(dim=(1, 2, 3))


def ssim(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0,
         window_size: int = 11, sigma: float = 1.5, reduction: str = "mean"):
    """SSIM, averaged over the batch by default (``reduction="none"`` for per-image)."""
    per_image = ssim_per_image(pred, target, max_val=max_val,
                               window_size=window_size, sigma=sigma)
    if reduction == "none":
        return per_image
    if reduction != "mean":
        raise ValueError(f"unknown reduction {reduction!r}; use 'mean' or 'none'")
    return per_image.mean()


def aggregate(values) -> float:
    """Mean of a sequence of per-image metric values, weighting each image equally.

    Kept as a named function (rather than an inline ``sum(...)/len(...)``) so the
    aggregation policy has exactly one definition in the codebase and can be
    referenced from the paper's protocol description.
    """
    values = list(values)
    if not values:
        raise ValueError("aggregate() got no values")
    return float(sum(values) / len(values))


@torch.no_grad()
def evaluate_batch(pred: torch.Tensor, target: torch.Tensor,
                   per_image: bool = False) -> dict:
    """PSNR/SSIM for a batch.

    ``per_image=False`` (default) returns floats averaged over the batch.
    ``per_image=True`` also returns the per-image lists, for callers that
    accumulate across batches (``train.py`` validation loop).
    """
    p = psnr_per_image(pred, target)
    s = ssim_per_image(pred, target)
    p_list = [float(v) for v in p]
    s_list = [float(v) for v in s]
    # Average through the same `aggregate` used by the validation loop, so the
    # batch-level value and the accumulated-over-batches value are identical
    # rather than differing by float32 accumulation order (~2e-6 dB).
    out = {"psnr": aggregate(p_list), "ssim": aggregate(s_list)}
    if per_image:
        out["psnr_per_image"] = p_list
        out["ssim_per_image"] = s_list
    return out


if __name__ == "__main__":
    a = torch.rand(4, 3, 64, 64)
    b = a + 0.01 * torch.rand(4, 3, 64, 64)
    print(evaluate_batch(a, b))
    print(evaluate_batch(a, b, per_image=True))
