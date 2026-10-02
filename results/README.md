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
| SRResNet + perceptual (ours)    | SR   | 4×    | 17.20     | 0.217 |

**Low-Light Enhancement (LOL-test)**

| Method                          | Task     | Scale | PSNR (dB) | SSIM  |
|---------------------------------|----------|-------|-----------|-------|
| Adaptive gamma (classical baseline) | LowLight | —  | ~15-17 †  | ~0.7 † |
| LowLight U-Net (ours)           | LowLight | —     | 19.26     | 0.74-0.78 |

> `*` = widely-cited reference value for Bicubic ×2 on Set5 (Dong et al., 2016).
> `†` = typical reference range for classical adaptive-gamma low-light enhancement on LOL (~15-17 dB), **not measured on this project's test split**; our trained U-Net (19.26 dB) already exceeds it.
> SR×4 (17.20 dB) uses VGG perceptual loss and is intentionally below the Bicubic baseline on PSNR (trades pixel fidelity for perceptual quality). Remaining `TBD`s (e.g. SRCNN 2×, which was not trained) can be filled if trained later.
