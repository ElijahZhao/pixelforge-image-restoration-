**Setup: models, data and environments.**

|  | SR $\times$4 | SR $\times$2 | Low-light |
|---|---|---|---|
| Architecture | SRResNet (SRCNN-style) | SRCNN | U-Net |
| Trainable parameters | 1,224,204 | 69,251 | 564,692 |
| Degradation | torchvision bicubic (AA) | torchvision bicubic (AA) | none (paired) |
| Training images | 800 (DIV2K train) | 800 (DIV2K train) | 485 pairs (LOL train) |
| Validation images | 100 (DIV2K val) | 100 (DIV2K val) | 15 pairs (LOL eval) |
| Seeds | 42-46 | 42-46 | 42-46 |
| Epochs | 200 | 100 | 200 |
| Loss | Charbonnier + 0.006 x VGG (inert) | Charbonnier | Charbonnier |
| Environment | torch 2.11+cu128 / CUDA 12.8 | torch 2.11+cu128 / CUDA 12.8 | torch 2.11+cu130 / CUDA 13.0 |

1. Parameter counts are measured `parameters()` values, not state-dict element counts: the $\times$4 model has 2,064 additional BN buffers (1,226,268 total), which are not trainable weights.
2. The perceptual term is reported because it is in the configuration, and reported as inert because the paired ablation gives on 27.61 / off 27.60 dB ($-0.01$ dB, sharpness difference 0.25%).
3. The two low-light-adjacent rows are two containers on one T4, not one: the low-light branch runs CUDA 13.0 while both SR branches run CUDA 12.8.
