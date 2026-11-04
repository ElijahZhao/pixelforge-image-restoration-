**Protocol definitions (P1-P8). Each protocol differs from its neighbour in one dimension.**

|  | Description | Degradation | Metric | Sampling | Aggregation |
|---|---|---|---|---|---|
| P1 | Random 96x96 crops, best-of-60 | torchvision bicubic, AA on | RGB | random crop | best-of-N |
| P2 | Centre 96x96 crop (training val) | torchvision bicubic, AA on | RGB | centre crop | per-image mean |
| P3 | Full image, RGB | torchvision bicubic, AA on | RGB | full image | per-image mean |
| P4 | Full image, RGB + shave | torchvision bicubic, AA on | RGB + shave | full image | per-image mean |
| P5 | Full image, Y + shave | torchvision bicubic, AA on | Y + shave | full image | per-image mean |
| P6 | Full image, Y + shave, **no antialias** | torchvision bicubic, **AA off** | Y + shave | full image | per-image mean |
| P7 | Full image, Y + shave, PIL bicubic | PIL bicubic | Y + shave | full image | per-image mean |
| P8 | Full image, RGB, PIL bicubic | PIL bicubic | RGB | full image | per-image mean |

1. P5 and P6 differ in exactly one boolean: the antialias flag of the downsampling operator. Every other cell is identical, and the two rows are the paper's sign-flip experiment.
2. The matrix covers four of the five dimensions (aggregation, sampling, metric/channel, degradation). It has no axis for validation batch size: after per-image aggregation the metric is structurally independent of the batch split, which `tests/test_protocol.py` locks in.
3. Low-light supports P1-P5 only: it is a paired task with no synthetic downsampling, so the PIL/anti-alias rows P6-P8 do not apply.
