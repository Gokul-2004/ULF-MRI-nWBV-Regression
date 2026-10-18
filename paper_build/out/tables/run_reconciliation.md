**Three runs of the identical LN+head recipe (not pooled)**

| Run | MAE | Bias | n < 0.020 | r |
|---|---|---|---|---|
| headline_loocv_cross_session | 0.0134 | -0.0022 | 19 | -0.119 |
| ablation_adapter_ln_head | 0.0137 | -0.0022 | 19 | -0.047 |
| multiseed_seed42 | 0.0133 | -0.0021 | 21 | +0.032 |

_Identical recipe (oasis_finetuned.pt, 80 epochs, lr 1e-4, wd 1e-4, batch 4, 5 augmentations, best-training-loss state). Only the RNG state differs: headline seeds 42 at import then consumes torch RNG building the base model; the ablation seeds once and runs 23 head-only folds before ln_head; multiseed re-seeds 42 after model construction, so its numpy augmentation stream matches the headline but its torch randperm stream does not._
