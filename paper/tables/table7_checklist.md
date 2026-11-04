**A six-item minimum reporting checklist, with the literature rate for each item.**

| # | Item to report | Why it matters (measured here) | Literature rate |
|---|---|---|---|
| 1 | Aggregation: per-image mean or global MSE | Moves the metric by 1.51-2.20 dB (full image) and up to 3.47 dB (single crop) | 1/63 |
| 2 | Validation set and whether it is the test set | 0/63 state it; deciding it needs code or appendix for 48 of 63 | 0/63 |
| 3 | Sampling window and position | One 96x96 crop carries 1.60 dB of noise; 23.10 dB spread across 100 images | n/a |
| 4 | Channel and shave boundary | 41/63 do not state the channel; 0/44 give the shave boundary | 0/44 (shave) |
| 5 | Degradation operator and its defaults | Antialias alone flips the sign of the reported gain | 5/44 |
| 6 | Seed count and best-of-N selection | Best-of-60 raises the metric by 6.07 dB on low-light alone | n/a |

1. This table has two halves: the left two columns are the checklist, the right column gives the literature rate. Section 5.4 uses the rates, Section 6.2 uses the checklist; they are one table.
2. Item 2 merges 'validation batch size' with 'validation/test overlap' -- both are answers to the same question, 'what set was the number computed on?'. Item 5 merges the degradation operator with the seed count for the same reason: both are 'what choices produced this number?'
