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
| SRResNet (ours, 2nd run) **deployed** | SR   | 2×    | 32.35     | 0.917 |
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

## Literature-comparable protocol (Y channel + border shave)

> The tables above are **RGB, all channels, no shave**. Published SR tables
> (SRCNN/SRResNet/EDSR/RCAN, and the NTIRE challenge protocol) instead report
> PSNR/SSIM on the **Y channel of YCbCr with `scale` border pixels shaved**.
> The table below is the same weights, same 100 images, same torch pipeline —
> only the coordinate system changes. Reproduce with
> `python scripts/eval_baseline.py --task sr --scale 4 --protocol y_shave`
> (and `--scale 2`; low-light uses `--protocol y`, since it has no upscale
> factor and therefore no shave).

| Slot                    | RGB (repo)     | Y + shave (literature) | Δ PSNR  |
|-------------------------|----------------|------------------------|---------|
| SR ×4 — bicubic         | 26.69 / 0.7541 | 28.10 / 0.7762         | +1.41   |
| SR ×4 — **ours**        | 27.47 / 0.7798 | **28.91 / 0.8026**     | +1.44   |
| SR ×2 — bicubic         | 31.04 / 0.8935 | 32.43 / 0.9046         | +1.39   |
| SR ×2 — **ours**        | 32.35 / 0.9165 | **33.82 / 0.9275**     | +1.47   |
| LowLight — no-op        | 7.77 / 0.1924  | 9.28 / 0.3928          | +1.51   |
| LowLight — **ours**     | 18.32 / 0.7457 | **20.59 / 0.8444**     | +2.27   |

> - **Definition of Y**: BT.601 studio swing, identical to MATLAB `rgb2ycbcr`
>   (`Y = (16 + 65.481R + 128.553G + 24.966B)/255`); shave = `scale` px per side
>   (4 px for ×4, which equals the NTIRE fixed-4 convention); SSIM uses a
>   Gaussian 11×11 / σ=1.5 window on Y (MATLAB `ssim` defaults). Implemented in
>   `train/eval_protocol.py`, pinned by 10 golden tests.
> - **The gain barely moves, the absolute value does.** The model's edge over
>   bicubic is ≈+0.78 dB → +0.81 dB (×4) and +1.31 → +1.39 dB (×2), i.e. within
>   0.1 dB; but a bare PSNR number is a property of the coordinate system and
>   cannot be moved between tables.
> - **An earlier estimate was wrong by 1 dB**: the 8th audit round extrapolated
>   "RGB → Y+shave ≈ 2.4–2.8 dB"; measured on the same weights it is 1.44 dB
>   (×4) / 1.47 dB (×2) for the model and 1.41 / 1.39 for the baseline. Quote the
>   measured column, not the extrapolation.
> - **Low-light is a different story**: the Y-channel shift is 2.27 dB, so the
>   RGB low-light number (18.32) is *not* directly comparable to a luma-channel
>   report even though it is within the paper's own convention.

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
> - Training-log bests (crop protocol) were 27.39 (SR) / 18.59 (low-light);
>   the full-image protocol above yields 27.47 for the same ×4 weight. These
>   are NOT interchangeable, and the training-log figure is the LESS
>   trustworthy of the two: the SR crop position used to be drawn at random for
>   validation, which moved the reported PSNR by 7.35 dB peak-to-peak against a
>   real training gain of +0.46 dB. The 27.39 came from a single epoch whose
>   neighbours were 25.3-26.6 -- its edge is crop luck, not model quality. The
>   validation crop is now deterministic (96×96 centre patch, see
>   `train/datasets.py`), but it still measures 96×96 pixels of each image,
>   while the table above measures the FULL image via `scripts/eval_baseline.py`.
>   Measured on the deployed weights over DIV2K val-100 (2026-11-02), the
>   centre-patch protocol reads 1.17 dB (×4) / 1.42 dB (×2) LOWER than
>   full-image -- same model, same images, different window. The full-image
>   numbers above are the ones to quote.
> - Reproduce with: `python scripts/eval_baseline.py --task sr --scale 4` and
>   `python scripts/eval_baseline.py --task lowlight`.
> - **Aggregation (added 2026-11-01, CHANGELOG 1.3.0)**: every PSNR/SSIM in the tables
>   above is the arithmetic mean of **per-image** values. `scripts/eval_baseline.py`
>   has always evaluated one image at a time, so these numbers are unaffected by the
>   1.3.0 change. The training logs are a different story: `train.py` used to build one
>   MSE over the whole validation batch, which is dominated by the hardest image in the
>   batch and therefore reads **lower** than the per-image mean (measured: 1.16 dB on
>   8×256² patches; more when image difficulty is mixed). Logs written before 1.3.0 and
>   logs written after it are therefore **not directly subtractable** — compare across
>   protocols only through `eval_baseline.py`, which is the same code path for both.

## Protocol matrix (Figure 1 source; added 2026-11-04)

> The two tables above each pin **one** protocol. This section holds all eight at once,
> for all five seeds — because the single biggest number in the whole project is not a
> model result, it is a **protocol choice**.

Files: `protocol_matrix_fig1_long.csv` (105 per-checkpoint rows),
`protocol_matrix_fig1_wide.csv` (21 seed-averaged rows), `model_vs_bicubic_gain.csv`
(21 model-vs-baseline rows). Produced by `scripts/protocol_matrix.py`, which groups the
eight protocols by `(degrade, sampling)` so each image needs only **three** full-image
forwards instead of eight.

| # | Protocol | Degrade | Metric | Sampling |
|---|----------|---------|--------|----------|
| P1 | random crop, best-of-N | `tv_bicubic_aa` | RGB | random 96², max over N draws |
| P2 | centre crop (training val) | `tv_bicubic_aa` | RGB | centre 96² |
| P3 | full image, RGB | `tv_bicubic_aa` | RGB | full |
| P4 | full image, RGB + shave | `tv_bicubic_aa` | RGB + shave | full |
| P5 | full image, Y + shave | `tv_bicubic_aa` | Y + shave | full |
| P6 | same as P5, **no antialias** | `tv_bicubic_noaa` | Y + shave | full |
| P7 | same as P5, **PIL bicubic** | `pil_bicubic` | Y + shave | full |
| P8 | same as P3, **PIL bicubic** | `pil_bicubic` | RGB | full |

**SR ×4 — SRResNet, 5 seeds (DIV2K val, 100 images)**

| Protocol | Model | Bicubic | Gain |
|---|---|---|---|
| P1 best-of-60 | 44.958 ± 0.236 | 44.783 *(n=95)* | **−0.579** |
| P2 centre 96² | 26.421 ± 0.036 | 25.808 | +0.614 |
| **P3 full RGB** | **27.614 ± 0.035** | 26.832 | **+0.782** |
| P4 full RGB+shave | 27.602 ± 0.035 | 26.819 | +0.784 |
| **P5 full Y+shave** | **29.052 ± 0.036** | 28.234 | **+0.818** |
| **P6 Y+shave, no AA** | **25.163 ± 0.019** | 26.916 | **−1.753** |
| P7 Y+shave, PIL | 29.043 ± 0.036 | 28.230 | +0.814 |
| P8 full RGB, PIL | 27.601 ± 0.035 | 26.825 | +0.775 |

**SR ×2 — SRCNN, 5 seeds (DIV2K val, 100 images)**

| Protocol | Model | Bicubic | Gain |
|---|---|---|---|
| P1 best-of-60 | 46.334 ± 0.558 | 48.107 *(n=95)* | **−2.312** |
| P2 centre 96² | 30.686 ± 0.069 | 30.042 | +0.644 |
| **P3 full RGB** | **32.163 ± 0.067** | 31.300 | **+0.863** |
| P4 full RGB+shave | 32.162 ± 0.067 | 31.295 | +0.868 |
| **P5 full Y+shave** | **33.829 ± 0.027** | 32.683 | **+1.146** |
| P6 Y+shave, no AA | 31.105 ± 0.094 | 32.994 | **−1.889** |
| P7 Y+shave, PIL | 33.809 ± 0.027 | 32.672 | +1.137 |
| P8 full RGB, PIL | 32.142 ± 0.066 | 31.282 | +0.859 |

> This baseline was **missing until 2026-11-04**. The stage-5 kernel ran
> `bicubic_sr_x4` and `bicubic_ll_x2` but never `bicubic_sr_x2`, so for two days
> ×2 had model numbers with no same-protocol reference. It was filled on a local
> CPU (no GPU, 13.5 min, 100 images) — bicubic baselines need no model, so this
> never had to cost GPU quota. The run reuses the same `protocol_matrix.py` code
> path as the ×4 baseline, and the two agree on the one cross-check available:
> P5 vs P7 (torchvision vs PIL bicubic) differ by 0.004 dB on ×4 and 0.011 dB on
> ×2, the same order of magnitude, so the CPU and T4 degradation paths have not
> drifted.
>
> **The sign flip reproduces, and is larger.** P5 → P6 moves from +1.146 to
> −1.889 (a swing of **3.035 dB**, against ×4's 2.571). ×2 also flips at P1:
> −1.773, because the bicubic best-of-60 reaches 48.107 and no model output does.
> That second flip is a property of the protocol, not of the model — see the note
> under "Three things this table says".

**Low-light — U-Net, 5 seeds (LOL eval15, 15 pairs, paired input)**

| Protocol | Model | Identity | Gain |
|---|---|---|---|
| P1 best-of-60 | 24.577 ± 0.415 | 12.174 | +12.403 |
| P2 centre 96² | 20.569 ± 0.530 | 7.632 | +12.937 |
| **P3 full RGB** | **18.509 ± 0.036** | 7.773 | **+10.736** |
| P4 full RGB+shave | 18.520 ± 0.035 | 7.771 | +10.748 |
| **P5 full Y+shave** | **20.759 ± 0.081** | 9.279 | **+11.480** |

> Low-light has no upscale factor, so there is nothing to shave and no degrade operator
> axis: P6–P8 do not exist for it. P1–P5 run with `scale=1` on the **real captured
> `low/`** input, paired with `high/` as target — the model is a residual U-Net
> (`clamp(x + out)`), not a super-resolution network, despite the historical
> `scale2` in its filename.

### Four things this table says

1. **P6 flips the sign — on both super-resolution tasks.** On the exact same weights
   and images, switching the degrade operator from antialiased to naive bicubic drops
   the model from **+0.818 dB above bicubic to −1.753 dB below it** on ×4, and from
   **+1.146 to −1.889** on ×2. Neither number is "wrong": antialiased downsampling
   removes the aliasing the network was trained to invert, naive downsampling creates
   it. A paper that reports only one of these is reporting an artifact of its own
   resize call. That the flip reproduces on a second task, a second architecture and a
   second scale is the strongest evidence here — it is not a quirk of one model.
2. **P1 flips too, and for a reason that makes the whole protocol unusable.** P1's
   baseline is averaged over **95 of 100 images**, because bicubic reconstructs a flat
   patch *exactly* under best-of-60 — five images (`0839/0843/0846/0847/0868.png`)
   score `inf` and are excluded from the aggregate (`n_inf=5`). Comparing that against a
   100-image model mean mixes two samples, so both sides are re-averaged on the same 95
   images (`scripts/align_p1_inf.py` → `results/p1_inf_alignment.csv`): the model then
   reads **−0.579 dB (×4)** and **−2.312 dB (×2)**. Nothing is wrong with the networks:
   on those five images the model gets 50–68 dB, which is excellent and still loses to a
   perfect reconstruction. **The protocol manufactures a baseline nobody can beat** — and
   the effect is not even signed consistently with model quality, which is why P1 is
   drawn off-scale in Figure 1 and never used in a claim.
3. **Protocol choice moves the number 18 dB.** For the same ×4 weight, the eight
   protocols span **25.16 → 44.96 dB**. Restricted to full-image protocols the span is
   still **3.889 dB** (P5 29.052 vs P6 25.163), against a model improvement of
   **0.782 dB** — a ratio of 5×, or 25× if P1 is included.
4. **The seed spread is small, the protocol spread is large.** Across 5 seeds the
   standard deviation is 0.019–0.036 dB on full-image protocols; across protocols it is
   3.889 dB. Seed variance is *not* the uncertainty that matters here — protocol is.

### Anchor gate (how these numbers are protected)

`scripts/protocol_matrix.py` refuses to run the matrix unless four known anchors
reproduce exactly: **27.4977 / 32.4243 / 28.9371 / 33.8931** (P8/P7 on ×4 and ×2).
Every kernel run in stage 5 printed `d=+0.0000` on all four before touching the matrix.
This caught a real bug on the first attempt: `pil_bicubic` returned a CPU tensor, the
GPU forward raised a device mismatch, and the gate stopped the run at 1.07 minutes
instead of publishing eight wrong protocols.

### How these three CSVs are produced

The tables used to be assembled by hand from the kernel outputs, so a reader could
only check them by trusting us. They are now generated:

```bash
# P1 only: re-average both sides on the 95 images where bicubic is finite
python scripts/align_p1_inf.py --out results/p1_inf_alignment.csv
# 6 raw matrix JSONs (+ the alignment table) -> long 105 / wide 21 / gain 21 rows
python scripts/aggregate_matrix.py --runs-root results/matrix_runs --out results
# the paper figure, straight from those CSVs
python scripts/make_figure1.py --out paper/figures --fmt pdf,png
```

`results/matrix_runs/` holds one `protocol_matrix.json` per kernel job — 64 KB in
total, so the raw evidence travels with the repo. Per-image CSVs (105 files) live in
the private archive instead; `aggregate_matrix.py` needs only the JSONs.

Two conventions the script pins down, because both have been silently wrong before:

- `psnr_std` is the **sample** standard deviation (ddof=1) over the 5 seeds. With n=5
  it is ~10% larger than the population value, which is bigger than every seed effect
  in this project.
- the baseline's **name** comes from `paired`, not from the checkpoint key. Paired mode
  performs no upscaling, so its reference is the capture itself; the archived
  `identity_ll` JSON still calls that key `bicubic` because the string was hardcoded
  before the paired path existed.

A third one was wrong until 2026-11-04 and is now pinned by `align_p1_inf.py`:

- **a gain is only defined on a common image set.** `protocol_matrix.py` drops `inf`
  PSNR from an aggregate and records `n_inf`, which is right for the baseline's own
  mean but silently changes the sample. P1 is the only protocol affected (every other
  baseline protocol has `n_inf = 0`, and so does the paired `identity` reference).
  The gain CSV therefore carries `n_aligned` and `gain_naive_mixed`: when the two
  differ, **`gain` is the like-for-like number** and `gain_naive_mixed` exists only so
  the size of the correction stays auditable.

```bash
# reproduce the matrix (needs the 15 seed weights)
# SR: 10 checkpoints x 8 protocols, 0.587 h on a T4 (kernel v2)
python scripts/protocol_matrix.py --task sr --scale 4 --datasets div2k/val --align 8
python scripts/protocol_matrix.py --task sr --scale 2 --datasets div2k/val --align 8
# low-light: 5 checkpoints x P1-P5 on paired real captures, 0.116 h (kernel v4)
python scripts/protocol_matrix.py --task lowlight --scale 1 --datasets lol/val/high --align 8

# the reference lines, through the same code path (no model, so CPU is enough:
# the 100-image x2 baseline takes 13.5 min on 16 threads)
python scripts/protocol_matrix.py --task sr --scale 4 --baseline --datasets div2k/val
python scripts/protocol_matrix.py --task sr --scale 2 --baseline --datasets div2k/val
python scripts/protocol_matrix.py --task lowlight --scale 1 --baseline --datasets lol/val/high --align 8
# in paired mode --baseline returns the capture unchanged, i.e. "do nothing";
# the CSV will name that checkpoint `identity`, not `bicubic`.
```

`--align 8` is the U-Net downsampling multiple; it reflect-pads the HR side to a
multiple of `align × scale` before the forward pass and crops back, so metrics stay on
original pixels. Without it a 600×400 input loses one pixel per skip connection
(`Expected size 76 but got size 75`). Plain conv stacks are unaffected by the flag.


