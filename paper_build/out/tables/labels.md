**Reference labels by cohort**

| Cohort | Target | Source | Acquired on |
|---|---|---|---|
| IXI (Stage 1) | 4 proxy targets (BTF, TCR, VBR, MCI) | FreeSurfer label maps shipped with the preprocessed IXI release (Chen et al.) | 1.5 T / 3 T T1w |
| OASIS-1 (Stage 2) | nWBV | OASIS-supplied spreadsheet: FSL FAST segmentation of the atlas-registered, masked image (Fotenos et al. 2005) | 1.5 T T1w |
| ds006557 (Stage 3) | nWBV = BrainSegVol / MaskVol | FastSurfer --seg_only on paired 3 T MPRAGE | 3 T (labels); 64 mT T2w (inputs) |
| van den Broek (external) | nWBV = BrainSegVol / MaskVol | FastSurfer --seg_only on paired 3 T | 3 T (labels); 64 mT T2w (inputs) |
| SynthSeg+ comparator | nWBV = (GM + WM) / TIV | FreeSurfer 7.4.1 mri_synthseg --robust on 64 mT T2w | 64 mT T2w |

_ds006557 and external label files were re-checked row by row against their stated formula during the build._
