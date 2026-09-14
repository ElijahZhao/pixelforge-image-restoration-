# Results

Training logs and quantitative results live here.

- `train_log_<task>_<model>.csv` — per-epoch `epoch, train_loss, val_psnr, val_ssim, time_s`.
- After training on a real GPU, fill in the comparison table below and copy a few
  visual examples into `assets/` so they can be shown on the website Method page.

## Example comparison table (fill after training)

| Method            | Task   | Scale | PSNR (dB) | SSIM  |
|-------------------|--------|-------|-----------|-------|
| Bicubic (classical baseline) | SR | 2x | 28.40 | 0.820 |
| SRCNN (ours)      | SR     | 2x | 29.10 | 0.845 |
| SRResNet (ours)   | SR     | 4x | 27.80 | 0.800 |
| Adaptive gamma (classical baseline) | LowLight | - | 16.20 | 0.710 |
| LowLight-UNet (ours) | LowLight | - | 19.50 | 0.820 |

> Numbers above are placeholders. Replace with your own validation metrics after
> training on DIV2K / LOL.
