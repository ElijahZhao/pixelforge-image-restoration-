# Results

Training logs and quantitative results live here.

- `train_log_<task>_<model>.csv` — per-epoch `epoch, train_loss, val_psnr, val_ssim, time_s`.
- After training on a real GPU, fill in the comparison table below and copy a few
  visual examples into `assets/` so they can be shown on the website Method page.

## Comparison table

> ⚠️ **These numbers are reported by the training runs, but are NOT yet
> independently reproducible from this repo** — no training log CSV or
> measurement script is committed here. Treat them as *pending verification*
> until an `eval_on_testset.py` + logs are added (see `PIXELFORGE_FIX_PLAN.md`,
> item F6). The TorchScript weights themselves are real (trained on AutoDL);
> only the *numbers* lack a reproducible path.
> Baseline rows are reference values from the literature.

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
> `†` = typical reference range for classical adaptive-gamma low-light enhancement on LOL (~15-17 dB), **not measured on this project's test split**.
> ⚠️ **Corrections (see DIAGNOSIS_ROUND1/3/10/11/18):**
> - SR×4 (17.20 dB) being below Bicubic is **not** an intentional trade-off. The perceptual loss had two implementation defects (VGG input not ImageNet-normalized; pixel term erased by a `0.01` factor). It is a *bug, now identified and fixed in code* — a re-train is needed to realize the improvement.
> - The U-Net's 19.26 dB is compared against a **literature reference band that was never measured on this project's own split**, and a controlled comparison on this repo's own sample showed the classical gamma baseline closer to ground truth. **Do not claim "beats baseline"** until F6 (reproducible eval) is done.
> - Remaining `TBD`s (e.g. SRCNN 2×, which was not trained) can be filled if trained later.
