**Table IX. Summary of statistical comparisons**

| # | Comparison | Test | n | Result |
|---|---|---|---|---|
| T1 | ViT3D vs CNN3D, OASIS-1 | paired Wilcoxon (normal approx.) | 38 | MAE 0.0584 vs 0.0243, p = 0.0037 (CNN3D better) |
| T2 | Physics vs blur, image fidelity | paired Wilcoxon | 23 | NCC 0.497 vs 0.516, p = 0.02 (blur better); SNR error p = 2e-07 (blur closer) |
| T3 | Physics vs blur, downstream nWBV, real 64 mT (matched, pooled over 3 seeds) | paired Wilcoxon | 23 | ΔMAE -0.0032, p = 0.85 (n.s.) |
| T4 | Proxy-regression vs DINO pretraining, real 64 mT LOOCV | paired Wilcoxon (normal approx.) | 23 | MAE 0.0134 vs 0.0220, p = 0.070 (n.s.); bootstrap ΔMAE -0.0086 [-0.016, -0.001] |
| P1 | Inter-session ICC, single frozen model, vs permutation null | permutation, 20,000 | 23 | ICC(C,1) 0.508, p = 0.012 (same-fold test, ICC 0.6146, withdrawn) |
| P2 | Prediction vs true nWBV | permutation, 20,000 | 23 | r = -0.119, p = 0.590 (n.s.) |
| S1 | Swin-UNETR frozen + ridge vs adapted ViT3D, real 64 mT | paired Wilcoxon | 23 | MAE 0.0075 vs 0.0134, p = 0.016 |
| S2 | Swin-UNETR readout, labels permuted through the full nested procedure | permutation, 1,000 | 23 | p = 0.001 |
| S3 | Adapted ViT3D vs leave-one-out constant mean | paired Wilcoxon | 23 | MAE 0.0134 vs 0.0128, p = 0.58 (n.s.) |

_T1 and T4 use the normal approximation with continuity correction (the method of the original analysis); the exact test gives smaller p._
