**Session reproducibility of predictions (n = 23)**

| Design | ICC(C,1) | ICC(A,1) |
|---|---|---|
| Headline, same fold model for both sessions | 0.615 | 0.612 |
| Leave-one-out constant mean, same fold | 1.000 | 1.000 |
| Frozen Stage-2 model (one model, no folds) | 0.508 | 0.507 |
| Cross-seed (HFC from seed a, HFE from seed b; mean of 20 pairs) | 0.289 | — |

_When one fold model predicts both sessions of its held-out subject, a fold-level offset is shared by both predictions; a constant predictor then scores ICC = 1. Only the frozen-model and cross-seed rows are free of that shared component._
