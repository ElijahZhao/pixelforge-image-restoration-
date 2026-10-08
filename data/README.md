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

## Running the service without datasets
You don't need the training datasets to run inference: `serve/models/` already ships
the exported TorchScript weights for all three tasks, so the service starts in ML mode
out of the box. The datasets are only needed if you want to (re)train. To swap or
retrain a model, export its weights into `serve/models/` and restart — the service
picks them up automatically. If a weight file is missing, that slot falls back to the
classical baseline (bicubic + unsharp for SR, adaptive gamma for low-light).
