# Data

This folder holds the public datasets used for training and evaluation.
They are **not** committed to the repo (too large). Download them once:

## Super-Resolution (DIV2K)
- Homepage: https://data.vision.ee.ethz.ch/cvl/DIV2K/
- Download `DIV2K_train_HR` and `DIV2K_valid_HR`, then arrange as:
  ```
  data/div2k/train/   <HR png files>
  data/div2k/val/     <HR png files>
  ```
- LR images are synthesised on the fly by bicubic downscaling (see `train/datasets.py`).

## Low-Light (LOL-v1)
- Source: https://github.com/weichen582/RetinexNet (LOLdataset_v1)
- Arrange as:
  ```
  data/lol/train/low/   <low-light images>
  data/lol/train/high/  <normal-light images, same filenames>
  data/lol/val/low/
  data/lol/val/high/
  ```

## Quick start without data
The inference service (`serve/`) works **without** any trained weights: it falls
back to classical baselines (bicubic + unsharp for SR, adaptive gamma correction
for low-light). Train the real models on a free GPU to replace these with learned
models — the service picks up exported weights automatically.
