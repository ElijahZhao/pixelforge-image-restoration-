**Reporting rates in 63 recent papers.**

| Reported item | Reported | Denominator | Rate | Wilson 95% CI |
|---|---|---|---|---|
| Aggregation method | 1/63 | 63 | 1.6% | [0.3%, 8.5%] |
| Validation batch size | 0/63 | 63 | 0.0% | [0.0%, 5.7%] |
| SR degradation operator reproducible | 5/44 | 44 | 11.4% | [5.0%, 24.0%] |
| Channel selection | 41/63 | 63 | 65.1% | [52.8%, 75.7%] |
| Shave boundary (SR subset only) | 0/44 | 44 | 0.0% | [0.0%, 8.0%] |
| val / test same set | 0/63 | 63 | 0.0% | [0.0%, 5.7%] |
| Code repository | 35/63 | 63 | 55.6% | [43.3%, 67.2%] |

1. Sample: N = 63 papers (SR 44 + LLIE 19), 2019-2025, all from arXiv.
2. Shave is counted on the SR subset only: the 5 papers marked N/A are LLIE, where a paired task has no synthetic downsampling and the concept does not apply. Reporting 0/63 would wrongly lower the upper bound of the interval.
3. For val/test, 48 of 63 papers were never coded (deciding it needs the code or appendix), so 15 is the count of papers that can be judged: 0 of 15 state it. Zero reported is not zero occurrence -- that inference is exactly what this paper criticises.
4. The degradation-operator count is 5/44 for the bicubic protocol (4 named MATLAB implementations plus 1 explicit antialias setting). Two further papers name a Gaussian/blur kernel with parameters: named degradations, but not the bicubic protocol, so they are excluded here (7/44 under the looser rule).
5. Known bias: the sample is drawn from arXiv, so it over-represents open-source work. The true rate in the journal literature can only be lower.
