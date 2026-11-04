**The sign reversal: identical weights, one boolean.**

|  | SR $\times$4 (SRResNet) | SR $\times$2 (SRCNN) | Low-light (U-Net) |
|---|---|---|---|
| P5 model (dB) | 29.0523 | 33.8290 | 20.7590 |
| P5 reference (dB) | 28.2344 | 32.6834 | 9.2787 |
| P5 gain (dB) | 0.8179 | 1.1456 | 11.4803 |
| P6 model (dB) | 25.1631 | 31.1048 | -- |
| P6 reference (dB) | 26.9162 | 32.9937 | -- |
| P6 gain (dB) | -1.7531 | -1.8889 | -- |
| Swing P5 -> P6 (dB) | 2.571 | 3.034 | -- |
| P1 gain (best-of-60, dB) | -0.5793 | -2.3123 | 12.4027 |

1. P5 and P6 share every setting but the antialias flag of the downsampling operator. The same weights, the same images and the same code path give a positive gain under P5 and a negative one under P6. Low-light has no P6, so its column shows the P5 row only.
2. P1's negative gain on SR is a definitional effect, not a weak model: under best-of-60 the bicubic reference reconstructs flat blocks exactly (PSNR = +inf on 5 images), so its aggregate is computed on the 95-image intersection with the model. No network output can beat an exact reconstruction. Low-light's P1 gain is positive (+12.40 dB) because its reference is the identity map, which best-of-60 also lifts; the effect there is that P1 sits 6.07 dB above P3 for the same model.
3. The $\times$2 swing is larger than the $\times$4 swing because the two sides move in opposite directions: on $\times$2 the model loses 2.72 dB while the baseline gains 0.31 dB.
