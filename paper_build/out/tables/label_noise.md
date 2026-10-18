**Propagation of reference-label noise (Monte Carlo, ViT3D headline)**

| Label noise σ | Noise floor MAE | Model MAE | MSE skill (95% interval) | Runs beating constant |
|---|---|---|---|---|
| 0.0025 | 0.002 | 0.0135 | -0.101 to -0.020 | 0.0% |
| 0.0050 | 0.004 | 0.0139 | -0.136 to +0.018 | 4.5% |
| 0.0063 | 0.005 | 0.0142 | -0.152 to +0.035 | 9.9% |

_σ = 0.0025: illustrative small test-retest error; σ = 0.005: illustrative moderate error; σ = 0.0063: SD of SynthSeg(64 mT T2w) minus FastSurfer(3T MPRAGE) nWBV, n=23 — an UPPER bound on FastSurfer label noise because it includes the modality gap and the definition difference_
