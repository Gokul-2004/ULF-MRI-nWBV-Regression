# CLAUDE.md — Repository Guide

This file documents the structure of this repository for contributors and for
AI coding assistants (Claude Code). It explains what every folder contains, how
the pipeline fits together, and how to reproduce the results.

## Project

**A Reproducible Feasibility Baseline for Segmentation-Free nWBV Regression
from 64 mT Ultra-Low-Field MRI Using Physics-Constrained Deep Learning.**

The goal is to predict **normalized whole-brain volume (nWBV)** directly from a
raw 64 mT ultra-low-field (ULF) MRI scan — with **no segmentation and no
super-resolution** — using a compact 3D Vision Transformer trained via a
three-stage pipeline (physics-simulated pre-training → OASIS-1 fine-tuning →
cross-session LOOCV adaptation on real 64 mT hardware).

Target venue: IEEE Access (under revision). See `standalone_paper/`.

## The three-stage pipeline (mental model)

1. **Stage 1 — Physics pre-training (IXI, n=156).** Public 3T volumes are
   degraded to synthetic 64 mT using a physics simulator (T1/T2 relaxation
   remapping, Rician noise, B0 inhomogeneity). A denoising objective
   reconstructs the high-field volume from its simulated low-field version.
2. **Stage 2 — Supervised fine-tuning (OASIS-1, n=375).** The encoder is
   fine-tuned to regress nWBV (FreeSurfer/FastSurfer ground truth).
3. **Stage 3 — Cross-session LOOCV adaptation (ds006557, n=23).** A lightweight
   adapter (final LayerNorm + head, 769 params) is trained on one 64 mT session
   (HFC) of the non-held-out subjects and tested on the other session (HFE) of
   the held-out subject. Fully subject- and session-independent.

## Top-level folders

| Folder | Contents |
|---|---|
| `models/` | Model definitions. `baselines.py` holds `BaselineViT3D` (4.23M params, the primary model) and `BaselineCNN3D` (matched-param comparator). `transformer_gnn/`, `uncertainty/` hold auxiliary/experimental modules. |
| `scripts/` | All runnable experiment/data scripts (36 files). Data download, physics simulation, training, LOOCV, ablations, figure generation, statistics. See "Key scripts" below. |
| `utils/` | Shared utilities. `field_conversion.py` = **the physics 64 mT simulator** (`FieldConverter`), the core scientific contribution. |
| `training/`, `evaluation/`, `inference/` | Reusable train/eval/inference helpers imported by scripts. |
| `configs/` | YAML configuration files for the pipeline. |
| `experiments/` | **Experiment outputs** — one subfolder per experiment, each with a `results.json`. This is the record of every result in the paper. (Raw FastSurfer/SynthSeg NIfTI volumes are gitignored as regenerable; the derived `nwbv_ground_truth.csv` is kept.) |
| `checkpoints/` | Trained model weights (~16 MB each). `oasis_finetuned.pt` is the Stage-2 checkpoint used by LOOCV; `real64mt_finetuned.pt`, `synthseg_finetuned.pt` are variants. |
| `standalone_paper/` | The manuscript (`manuscript.tex`), the reviewer **`response_to_reviewers.md`**, and `paper_figures/` (final figures). This is the submission package. |
| `paper_figures/`, `Pictures/`, `figures_for_sharing/` | Generated figures (PNG/PDF). `Pictures/` is the upload-ready set for Overleaf. |
| `Research_Papers/` | Reference PDFs (related literature). For personal reference only — respect copyright; do not redistribute. |
| `logs/` | Training logs. |
| `data/` | **NOT in git.** Raw MRI datasets (~98 GB): IXI, OASIS-1, ds006557. All public — re-download with the scripts in `scripts/` (see "Getting the data"). |
| `Website_dat/` | **NOT in git.** Website/demo assets (~4 GB). |

## Key scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `download_ixi_data.py`, `download_mri_data.py`, `download_real_mri.py` | Fetch the public datasets into `data/`. |
| `convert_high_to_low.py` | Apply the physics 64 mT simulation to high-field volumes. |
| `finetune_oasis.py` | Stage-2 OASIS-1 fine-tuning (produces `oasis_finetuned.pt`). |
| `loocv_cross_session.py` | **The headline experiment** — cross-session LOOCV (HFC→HFE), LN+head adapter. |
| `ablation_adapter_strategy.py` | Adapter ablation: head-only vs LN+head vs full fine-tune. |
| `multiseed_loocv.py` | Multi-seed robustness (5 seeds) — imports and reruns the published LOOCV. |
| `simulation_sensitivity.py` | Simulation-parameter sensitivity (±20% SNR/B0/relaxation) on OASIS test. |
| `validate_simulation.py` | Simulation-fidelity vs real 64 mT (NCC, SNR/CNR). |
| `ablation_gaussianblur.py` | Physics-sim vs Gaussian-blur pre-training ablation. |
| `ablation_vit_vs_cnn_real64mt.py` | ViT3D vs CNN3D comparison. |
| `uncertainty_ci.py` | MC Dropout uncertainty / calibration. |
| `failure_analysis.py`, `paper_statistics.py` | Failure characterization and paper statistics. |
| `generate_all_figures.py` | Regenerate every paper figure from `experiments/*/results.json`. |

## Experiment results (`experiments/`)

Each subfolder has a `results.json`. The ones backing the current revision:

- `loocv_cross_session/` — headline LOOCV (MAE ≈ 0.0134).
- `ablation_adapter/` — adapter strategies (head 0.0133 / LN+head 0.0137 / full-FT 0.0123).
- `multiseed_loocv/` — 5-seed robustness (MAE 0.0130 ± 0.0004, ICC 0.644 ± 0.058).
- `simulation_sensitivity/` — parameter sensitivity (max |ΔMAE| = 0.0062).
- `sim_validation/` — simulation fidelity (NCC vs Gaussian blur).
- `oasis_finetune/`, `oasis_bootstrap/`, `oasis_cnn_comparison/` — OASIS-1 results.
- `simulated_dementia/` — CDR-stratified stress test.

## Getting the data (not in git)

`data/` (~98 GB) holds public datasets, excluded from git. To reproduce:

1. **IXI** (Stage 1): `python scripts/download_ixi_data.py`
2. **OASIS-1** (Stage 2): register at oasis-brains.org; place under `data/oasis_processed/` and `data/oasis_raw/`.
3. **ds006557** (Stage 3, real 64 mT): OpenNeuro dataset ds006557; place under `data/ds006557_data/`.

The nWBV ground-truth labels (`experiments/fastsurfer_output/nwbv_ground_truth.csv`)
are committed, so the LOOCV/ablation scripts can run against the checkpoints
without re-running FastSurfer.

## Reproducing key results (needs `data/` + committed checkpoints)

```bash
python scripts/loocv_cross_session.py        # headline LOOCV
python scripts/ablation_adapter_strategy.py  # adapter ablation
python scripts/multiseed_loocv.py            # multi-seed robustness
python scripts/simulation_sensitivity.py     # sim-parameter sensitivity
python scripts/generate_all_figures.py       # regenerate all figures
```

Environment: Python 3.12, PyTorch, NumPy, SciPy, nibabel, scikit-image,
matplotlib, pandas. (A `requirements.txt` should be added — currently the
dependencies are implied by the imports.) Note: training/eval here run on **CPU**
(the development machine's GPU was CUDA-incompatible); a modern GPU will be far
faster.

## Conventions for contributors / assistants

- New experiments write a `results.json` into their own `experiments/<name>/` folder.
- Figure scripts read from `experiments/*/results.json` — never hardcode numbers.
- The physics simulator constants live in `utils/field_conversion.py`.
- Keep the paper's honest, feasibility-baseline framing: report negative and
  null results plainly (constant-mean baseline comparison, non-significant
  physics-vs-blur, miscalibrated uncertainty).
