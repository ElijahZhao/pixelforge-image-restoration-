# Per-image evaluation values

Every protocol cell behind the paper's tables, released as the
per-image PSNR/SSIM values that were averaged to produce it.

## What is here

126 CSV files, one per (task, seed, protocol) cell -- 126 distinct cells in total.

- `generator_x4_x4/` -- SR x4 (SRGenerator, 5 seeds x 8 protocols = 40 cells)

- `srcnn_x2_x2/` -- SR x2 (SRCNN, 5 seeds x 8 protocols = 40 cells)

- `lowlight_unet_x1/` -- Low-light (LowLightUNet, 5 seeds x 5 protocols = 25 cells)

- `bicubic_sr_x4/` -- SR x4 bicubic baseline (8 protocols = 8 cells)

- `bicubic_sr_x2/` -- SR x2 bicubic baseline (8 protocols = 8 cells)

- `identity_ll/` -- Low-light identity baseline (5 protocols = 5 cells)

## File naming

    <job>__<checkpoint>__<protocol>.csv

for example

    generator_x4_x4__sr_generator_scale4_best__s42__P3.csv

means: the x4 super-resolution generator, seed 42, protocol P3
(full image, RGB channel, bicubic downsampling with antialiasing).

## File format

    image,psnr,ssim

One row per test image. `psnr` is in dB, `ssim` is the standard
per-image SSIM.

**Non-finite PSNR values.** A bicubic upsampler can reconstruct a
flat patch exactly, so MSE is 0 and PSNR is `+inf`. Such rows are
kept in the file (so the file is a faithful record) but must be
dropped before averaging: the reported means for the affected
protocols are taken over the finite subset only. This is the
reason P1's bicubic baseline is an average over 95 images while
the model side is an average over 100 -- see `paper/FACTS.md`
section 10, entry E21.

## How to reproduce the paper's tables

Re-average each file to get the cell value, then average the
five seeds per (task, protocol) to get the row in the results
table, and take the sample standard deviation across those five
seed means for the spread column.

Two scripts in the repository do this and check the result
against the released aggregates:

    python3.11 scripts/verify_public_release.py   # re-derive every cell
    python3.11 scripts/make_tables.py --check      # re-derive every table

## Provenance

Each file was copied from the private evaluation archive after
verifying that re-averaging it reproduces the corresponding
`psnr_mean` in `results/matrix_runs/<job>/protocol_matrix.json`.
Files from superseded or failed runs were excluded; the check is
reproducible with `scripts/verify_per_image_mapping.py`.

The aggregate matrix itself (`results/matrix_runs/`) and the
derived tables (`results/*.csv`) are in the same repository.
