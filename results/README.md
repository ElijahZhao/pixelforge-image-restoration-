# Results

Training logs and quantitative results live here.

- `train_log_<task>_<model>_scale<n>.csv` — per-epoch `epoch, train_loss, val_psnr, val_ssim, time_s`.
  The CSV header also records the training objective and best-checkpoint criterion.
  The scale is part of the name because SR ×2 and ×4 are separate models; sharing one
  filename would append two unrelated trainings into a single CSV.
- These logs are committed (via `.gitignore` whitelist) so the reported numbers are
  reproducible, not just reported.

## Comparison table (after fixes + retrain, 2026-10-03)

> All PSNR/SSIM below are measured on the same validation set with the same metric
> implementation by `scripts/eval_baseline.py`, so the model and the baseline differ
> only in method. Reproduce with `python scripts/eval_baseline.py --task sr --scale 4`
> and `python scripts/eval_baseline.py --task lowlight`.

Super-Resolution ×4 (full-image validation)

| Method                        | Task | Scale | PSNR (dB) | SSIM  |
|-------------------------------|------|-------|-----------|-------|
| Bicubic (classical baseline)  | SR   | 4×    | 26.69     | 0.754 |
| SRResNet (ours, retrained)    | SR   | 4×    | 27.47     | 0.780 |
| Gain                          |      |       | +0.77     | +0.026 |
| SRCNN (ours)                  | SR   | 2×    | TBD (not trained) | TBD |

Low-Light Enhancement (LOL validation, 15 images)

| Method                          | Task     | PSNR (dB) | SSIM  |
|---------------------------------|----------|-----------|-------|
| No-op (raw low, baseline)       | LowLight | 7.77      | 0.192 |
| LowLight U-Net (ours, retrained) | LowLight | 18.18     | 0.739 |
| Gain                            |          | +10.41    | +0.547 |

> Notes
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
