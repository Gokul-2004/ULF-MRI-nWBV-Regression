**Matched physics-simulation vs Gaussian-blur pretraining (identical pipeline, splits, budgets and seeds)**

|  | Real 64 mT MAE, physics | Real 64 mT MAE, blur | Δ (physics − blur) [95% CI] | Wilcoxon p | OASIS Δ on physics inputs | OASIS Δ on blur inputs |
|---|---|---|---|---|---|---|
| seed 42 | 0.0170 | 0.0268 | -0.0098 [-0.0185, -0.0013] | 0.045 | -0.0319 | +0.0634 |
| seed 1 | 0.0192 | 0.0238 | -0.0047 [-0.0155, +0.0054] | 0.87 | -0.0377 | +0.0515 |
| seed 7 | 0.0249 | 0.0201 | +0.0048 [-0.0054, +0.0146] | 0.26 | -0.0349 | +0.0573 |
| pooled | 0.0203 | 0.0236 | -0.0032 [-0.0121, +0.0046] | 0.85 | -0.0348 | +0.0574 |

_Real 64 mT: cross-session LOOCV, n = 23, paired per subject. OASIS: 38 test subjects with one fixed noise realisation for both arms._
