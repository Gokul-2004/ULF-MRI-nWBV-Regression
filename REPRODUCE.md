# Reproducing the results

Release `v3.0-access-r3` accompanies the IEEE Access resubmission of
Access-2026-45125, *Direct nWBV Regression from 64 mT Ultra-Low-Field MRI: A
Reproducible Feasibility and Failure-Analysis Baseline*.

Everything in the manuscript can be checked at three levels of effort.

| Level | Command | Needs | Time (CPU) |
|---|---|---|---|
| 1. Rebuild the paper | `python -m paper_build` | this repository only | minutes |
| 2. Re-run the evaluations | `scripts/reproduce_all.sh experiments` | public datasets + released checkpoints | ~2 h |
| 3. Retrain | `scripts/reproduce_all.sh retrain` | public datasets | ~9 h |

All results were produced on one machine: Intel i7-8750H (6 cores), 31 GB RAM,
Ubuntu, Python 3.12.3, the pinned versions in `requirements.txt`, **CPU only**.

## Level 1 — every table, figure and number from the committed result files

```bash
pip install -r paper_build/requirements.txt     # numpy, scipy, matplotlib
python -m paper_build --strict
```

Writes `paper_build/out/`:

- `tables/` — every table in the manuscript (Markdown and CSV)
- `figures/` — every data figure (600 dpi PNG and SVG)
- `numbers.json` — every number quoted in the text, each with the result file it comes from
- `BUILD_REPORT.md` — git commit, the list of input files, and cross-checks in
  which key results are recomputed independently and compared with the value
  the experiment script saved

`--strict` fails if any input file is not tracked by git or the working tree is
modified. The build also refuses to run if any of its code draws random numbers
outside the seeded bootstrap (seed 42, 10,000 resamples) or contains a typed-in
result. Bootstrap confidence intervals are therefore identical on every run.

Not regenerated: Figure 1 (example MRI slices; needs the image data).

## Level 2 — re-run the evaluations

Inputs (none of the data can be redistributed here; all are public):

| Input | Where to put it | Source |
|---|---|---|
| ds006557 (real 64 mT, n = 23, paired 3 T) | `data/ds006557_data/` | OpenNeuro ds006557 |
| van den Broek et al. (external, n = 10 usable) | `Website_dat/extracted/` | Zenodo 10.5281/zenodo.15471394 |
| OASIS-1 (for E8 only) | `data/oasis_processed/`, `data/oasis_raw/` | oasis-brains.org |
| Stage-2 ViT3D `oasis_finetuned.pt` | `checkpoints/` | in git |
| Old 15-subject adapter `real64mt_finetuned.pt` | `checkpoints/` | in git |
| All-23 headline adapters `r3_adapter_all23_seed*.pt` | `checkpoints/` | in git (retrained by E3 if absent) |
| CNN3D `r3_cnn3d_oasis.pt` | `checkpoints/` | in git |
| Swin-UNETR `swin_oasis.pt` (256 MB) | `checkpoints/` | **release asset** (exceeds GitHub's file limit) |

Reference labels are committed (`experiments/fastsurfer_output/nwbv_ground_truth.csv`,
`experiments/fastsurfer_zenodo/nwbv_ground_truth_zenodo.csv`, SynthSeg+ volumes in
`experiments/synthseg_output/`), so FastSurfer and SynthSeg+ need not be re-run.

`scripts/reproduce_all.sh experiments` runs, in order:

| Script (`scripts/round3/`) | Writes `experiments/…` | Answers |
|---|---|---|
| `e1_fold_independent_reproducibility.py` | `r3_e1_reproducibility/`, `r3_frozen_predictions/` | R2.3, R4.1 |
| `e2_nonneural_baselines.py` | `r3_e2_nonneural_baselines/` | R4.2, R2.8 |
| `e2b_swin_readout_verification.py` | `r3_e2b_swin_verification/` | R4.2, R2.8 |
| `a_existing_data_analyses.py` | `r3_existing_data_analyses/` | R2.2, R2.4, R2.11, R4.3 |
| `e3_external_headline_adapter.py` | `r3_e3_external/` | R2.6, R4.5, R2.9 |
| `e7_conformal_per_cohort.py` | `r3_e7_conformal/` | R3.1, R4.4, R2.9 |
| `e6_time_budget.py` | `r3_e6_time_budget/` | R2.12 |
| `e2c_cnn3d_target_domain.py` | `r3_e2c_cnn3d_target/` | R2.8, R4.2 |
| `e8_oasis_fixed_inputs.py` | `r3_e8_oasis_fixed_inputs/` | R3.3 |

Each round-3 `results.json` records the script, git commit, whether the code tree
was modified, library versions and a timestamp.

## Level 3 — retrain

`scripts/reproduce_all.sh retrain` runs the matched physics-vs-blur experiment
E4 (`e4_matched_physics_vs_blur.py`; both arms × seeds 42, 1, 7 through Stage 1,
Stage 2 and the cross-session LOOCV, ~75 min per run) and, if its checkpoint is
absent, retrains CNN3D with the Table-II recipe. Additionally needs
`data/high_field/`, `data/low_field/` (Stage-1 volumes; `scripts/download_ixi_data.py`)
and OASIS-1. The 12 E4 checkpoints (194 MB) are not in git; they are attached to
the release.

## Known limits of reproducibility (stated in the manuscript)

1. **The headline Stage-1/Stage-2 checkpoints predate version control.**
   `checkpoints/best_model.pt` and `oasis_finetuned.pt` were trained in March 2026;
   the git history starts on 22 July 2026. The headline numbers are properties of
   these released checkpoints. Re-running the documented recipe from scratch (E4
   physics arm, three seeds) gives OASIS-1 test MAE 0.022–0.026 and real-64 mT
   cross-session MAE 0.017–0.025, not the headline values. Both are reported.
2. **The 64 mT simulator draws unseeded noise.** OASIS-1 numbers computed with
   on-the-fly simulation change from one evaluation to the next. E8 re-evaluates
   every saved OASIS-1 checkpoint on fixed, seeded inputs (10 noise realisations);
   the manuscript reports those.
3. **CPU training is not bit-for-bit deterministic across thread counts.** Three
   runs of the identical adapter recipe give MAE 0.0133–0.0137 with per-subject
   agreement r ≈ 0.5 (`r3_existing_data_analyses → r2_11`).
4. **Results produced before round 3** (`loocv_cross_session`, `ablation_*`,
   `multiseed_loocv`, `ssl_comparator_*`, `arch_comparator_*`) have no provenance
   block. Fine-tuned weights for UNETR and the MAE, SimMIM and contrastive
   comparators were not saved, so those OASIS-1 numbers come from a single
   on-the-fly evaluation.
5. `experiments/transfer_probe_64mt/` used features computed without the CLS
   token and positional embedding and a non-nested ridge penalty; it is
   superseded by `r3_e2_nonneural_baselines/` and not used in the manuscript.

## Legacy code

`scripts/legacy_figures/` holds the figure scripts of earlier versions. Some use
synthetic placeholders; none is used for the current manuscript.
