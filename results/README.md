# Results

Training logs and quantitative results live here.

- `train_log_<task>_<model>.csv` — per-epoch `epoch, train_loss, val_psnr, val_ssim, time_s`.
- After training on a real GPU, fill in the comparison table below and copy a few
  visual examples into `assets/` so they can be shown on the website Method page.

## Comparison table

> ⚠️ **Baseline rows are reference values; all "(ours)" rows are NOT yet measured.**
> The `ours` models have not been trained (no GPU weights in this repo), so their
> cells are intentionally left as `TBD`. Do **not** fill in numbers until you have
> actually trained on DIV2K / LOL and evaluated on the test sets.

**Super-Resolution (Set5, Y-channel)**

| Method                          | Task | Scale | PSNR (dB) | SSIM  |
|---------------------------------|------|-------|-----------|-------|
| Bicubic (classical baseline)    | SR   | 2×    | 33.66 *   | 0.9299 * |
| SRCNN (ours)                    | SR   | 2×    | TBD       | TBD   |
| SRResNet + perceptual (ours)    | SR   | 4×    | TBD       | TBD   |

**Low-Light Enhancement (LOL-test)**

| Method                          | Task     | Scale | PSNR (dB) | SSIM  |
|---------------------------------|----------|-------|-----------|-------|
| Adaptive gamma (classical baseline) | LowLight | —  | TBD       | TBD   |
| LowLight U-Net (ours)           | LowLight | —     | TBD       | TBD   |

> `*` = widely-cited reference value for Bicubic ×2 on Set5 (Dong et al., 2016).
> Baseline low-light numbers are also left as `TBD` because they should be
> re-measured on the exact same test split as your trained model for a fair
> comparison. Replace every `TBD` with your own measured numbers after training.
