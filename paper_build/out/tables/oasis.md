**OASIS-1 test set (n = 38), physics-simulated 64 mT inputs**

| Model | Parameters | MAE | r | Bias | Test inputs |
|---|---|---|---|---|---|
| ViT3D (proxy-regression pretraining, headline) | 4.23 M | 0.0582 ± 0.0001 | 0.760 ± 0.063 | +0.0531 | fixed, 10 draws |
| CNN3D | 8.22 M | 0.0270 ± 0.0017 | 0.863 ± 0.015 | -0.0082 | fixed, 10 draws |
| Swin-UNETR | 62.19 M | 0.0154 ± 0.0008 | 0.960 ± 0.004 | +0.0062 | fixed, 10 draws |
| ViT3D (DINO pretraining) | 4.23 M | 0.0256 ± 0.0015 | 0.907 ± 0.012 | +0.0185 | fixed, 10 draws |
| UNETR | 92.69 M | 0.0198 | 0.923 | -0.0002 | single draw (weights not saved) |
| ViT3D (MAE pretraining) | 4.23 M | 0.0319 | 0.756 | +0.0082 | single draw (weights not saved) |
| ViT3D (SimMIM pretraining) | 4.23 M | 0.0538 | 0.907 | +0.0538 | single draw (weights not saved) |
| ViT3D (SimCLR pretraining) | 4.23 M | 0.0589 | 0.744 | -0.0295 | single draw (weights not saved) |

_Test inputs are simulated from the 1.5 T scans. Rows marked 'fixed' are mean ± SD over 10 seeded noise realisations of the same 38 inputs; the other rows are one unseeded realisation, as originally reported, because their fine-tuned weights were not saved._
