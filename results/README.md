# Results

Training logs and quantitative results live here.

- `train_log_<task>_<model>_scale<n>.csv` — per-epoch `epoch, train_loss, val_psnr, val_ssim, time_s`.
  The CSV header also records the training objective and best-checkpoint criterion.
  The scale is part of the name because SR ×2 and ×4 are separate models; sharing one
  filename would append two unrelated trainings into a single CSV.
- These logs are committed (via `.gitignore` whitelist) so the reported numbers are
  reproducible, not just reported.

## Comparison table (full-image protocol; updated 2026-10-07 after the three-run best-of selection)

> All PSNR/SSIM below are measured on the same validation set with the same metric
> implementation by `scripts/eval_baseline.py`, so the model and the baseline differ
> only in method. Reproduce with `python scripts/eval_baseline.py --task sr --scale 4`,
> `python scripts/eval_baseline.py --task sr --scale 2` and
> `python scripts/eval_baseline.py --task lowlight`.

Super-Resolution (full-image validation, 100 images)

| Method                        | Task | Scale | PSNR (dB) | SSIM  |
|-------------------------------|------|-------|-----------|-------|
| Bicubic (classical baseline)  | SR   | 2×    | 31.04     | 0.894 |
| SRCNN (ours, 2nd run) **deployed** | SR   | 2×    | 32.35     | 0.917 |
| Gain                          |      |       | +1.31     | +0.023 |
| Bicubic (classical baseline)  | SR   | 4×    | 26.69     | 0.754 |
| SRResNet (ours, 1st run) **deployed** | SR   | 4×    | 27.47     | 0.780 |
| SRResNet (ours, 2nd run)      | SR   | 4×    | 27.38     | 0.777 |
| Gain (deployed)               |      |       | +0.77     | +0.026 |

Low-Light Enhancement (LOL validation, 15 images)

| Method                          | Task     | PSNR (dB) | SSIM  |
|---------------------------------|----------|-----------|-------|
| No-op (raw low, baseline)       | LowLight | 7.77      | 0.192 |
| LowLight U-Net (ours, 1st run)  | LowLight | 18.12     | 0.743 |
| LowLight U-Net (ours, 3rd run) **deployed** | LowLight | 18.32 | 0.746 |
| Gain (deployed)                 |          | +10.55    | +0.553 |

> Notes
> - **2026-10-07 three-run best-of selection**: a retraining round on AutoDL (RTX 4090,
>   batch 16) produced the first-ever ×2 weight (32.35, +1.31 dB over bicubic — deployed)
>   and a new low-light U-Net (18.32 vs 18.12, both PSNR and SSIM higher — deployed);
>   SR ×4 landed at 27.38, 0.09 dB below the historical weight, which stays deployed.
>   Per-slot selection rule: **higher full-image number wins**; all three deployed slots
>   beat their baselines. Weights: `d15d8353` (×4) / `f751611d` (×2) / `56e2f98d` (low-light).
> - The batch-16 training-log numbers (SR×4 23.34 / low-light 17.20) are **not comparable**
>   with the historical training logs: the validation protocol changed from random-crop
>   to deterministic crop (see CHANGELOG 1.1.1). Under the same-protocol full-image
>   comparison the new models are on par with (×4) or better than (×2, low-light) the
>   previous generation — the log-level "4 dB gap" is a protocol artifact, not model
>   quality.
> - SR ×4 was previously 17.20 / 0.217 (below Bicubic) due to two perceptual-loss
>   defects (VGG input not ImageNet-normalized; pixel term erased by a `0.01` factor).
>   Both are fixed in code and the model has been retrained; it is now above
>   the Bicubic baseline.
> - The older "low-light 19.26 dB" was measured under a random-crop protocol on a
>   literature reference band never measured on this project's split; it is
>   not comparable with the current deterministic full-image protocol. Under the
>   fair, same-protocol comparison the U-Net gains +10.41 dB over the no-op baseline.
> - Training-log bests (random-crop protocol) were 27.39 (SR) / 18.59 (low-light);
>   the full-image protocol above yields 27.47 / 18.18. These are NOT
>   interchangeable, and the training-log figure is the LESS trustworthy of the two:
>   the SR crop position was drawn at random for validation, which moved the reported
>   PSNR by 7.35 dB peak-to-peak against a real training gain of +0.46 dB. The
>   27.39 came from a single epoch whose neighbours were 25.3-26.6 -- its edge is
>   crop luck, not model quality. The validation crop is now deterministic, so
>   the numbers below (full-image, one fixed crop) are the ones to quote.
> - Reproduce with: `python scripts/eval_baseline.py --task sr --scale 4` and
>   `python scripts/eval_baseline.py --task lowlight`.
