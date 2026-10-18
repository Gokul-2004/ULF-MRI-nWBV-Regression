**ViT3D LN+head adapter, five seeds, cross-session LOOCV (n = 23)**

| Seed | MAE | MSE skill | r | ICC(C,1) same fold | n < 0.020 |
|---|---|---|---|---|---|
| 42 | 0.0133 | +0.01 | +0.03 | 0.629 | 21/23 |
| 1 | 0.0134 | -0.06 | -0.12 | 0.629 | 18/23 |
| 7 | 0.0129 | -0.03 | -0.07 | 0.563 | 19/23 |
| 123 | 0.0131 | +0.02 | +0.07 | 0.714 | 19/23 |
| 2024 | 0.0123 | +0.09 | +0.19 | 0.684 | 20/23 |

_Same-fold ICC shares the fold-level offset and is not evidence of input dependence (see reproducibility table)._
