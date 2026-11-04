**Nearest-neighbour work on evaluation protocols, with the part of this paper that addresses what each left open.**

| Work | Venue | What it established | What it left open | What this paper adds |
|---|---|---|---|---|
| `keles2021psnr` | PCS 2021 | PSNR over a set is not the mean of per-image PSNR; gives the bound for SISR (2.07 dB measured, 2.507 dB theoretical) | Isolates aggregation only; no protocol matrix, and no seed variance | Protocol P1 (best-of-N) is the extreme case of the same effect; we quantify it against seven other protocol dimensions |
| `parmar2022cleanfid` | CVPR 2022 | Antialiased vs aliased resizing changes FID rankings in GAN evaluation | Resize semantics only, in the perceptual/GAN metric setting | P6 vs P5 isolates antialiasing in the distortion metric; the effect is 2.72 dB, not a ranking detail |
| `zhang2024seal` | ICLR 2024 | Random-crop evaluation is unreliable and flips rankings among SR methods | Sampling randomness only; degradation, channel and aggregation are held fixed | P1 vs P2 vs P3 separates crop position from the other five dimensions, with 5 seeds per cell |
| `srinivasan2026lol` | arXiv 2026 | LOL benchmarks carry hidden label shifts: LOLv1 test labels render differently, and LOLv2-Real is 14% darker than the training labels | Dataset-side label shift; the evaluation protocol itself is not varied | The low-light task is our counterexample: protocol does not dominate there (ratio 0.21), so the claim is conditional |
| `shah2026hyperbench` | arXiv 2026 | Protocol implementation differences in the Wald protocol prevent hyper-resolution rankings from holding | One task family (hyperspectral), one protocol dimension (degradation) | Three tasks and eight protocols; the protocol span exceeds the training gain on two of them |
| `su2025bridging` | arXiv 2025 | Audits perceptual quality metrics for SR and finds they diverge from distortion metrics | Metric choice is orthogonal to the protocol dimensions this paper varies | Holds the metric fixed and varies sampling, channel, aggregation and degradation instead |
| `cawley2010overfitting` | JMLR 2010 | Selection bias: repeated model selection on a validation set inflates the reported performance of the selected model | Gives the theory; does not measure how large the bias is under a stated protocol | Measures the same bias at 1.74 dB under P1, which is why the reporting checklist requires the selection budget N |

1. The seven entries are the works that vary an *evaluation* protocol rather than propose a method. Five were verified against arXiv metadata; two (Cawley & Talbot, Keleş et al.) against the publisher record.
2. No row claims that the cited work is wrong. Each entry records what it established and the scope it left open.
3. `refs.bib` is the single source for every citation in this table; the keys in the first column are verified to exist by `scripts/make_table1.py --check`.
