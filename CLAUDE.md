# CLAUDE.md — Repository Guide

This file documents the structure of this repository for contributors and for
AI coding assistants (Claude Code). It explains what every folder contains, how
the pipeline fits together, and how to reproduce the results.

## Project

**Direct nWBV Regression from 64 mT Ultra-Low-Field MRI: A Reproducible
Feasibility and Failure-Analysis Baseline** (round-3 title; the round-2 title was
"A Reproducible Feasibility Baseline for Segmentation-Free nWBV Regression from
64 mT Ultra-Low-Field MRI Using Physics-Constrained Deep Learning").

The goal is to predict **normalized whole-brain volume (nWBV)** directly from a
raw 64 mT ultra-low-field (ULF) MRI scan — with **no segmentation and no
super-resolution** — using a compact 3D Vision Transformer trained via a
three-stage pipeline (physics-simulated pre-training → OASIS-1 fine-tuning →
cross-session LOOCV adaptation on real 64 mT hardware).

Target venue: IEEE Access (under revision). See `standalone_paper/`.

## The three-stage pipeline (mental model)

1. **Stage 1 — Proxy-target pre-training (`experiments/stage1/run_stage1.py`).**
   The Stage-1 set is 156 volumes: 150 IXI (Chen et al. preprocessed release,
   1.5 T/3 T, with FreeSurfer label maps) + 6 non-IXI (Haxby, 3 synthetic, 2 OASIS
   VBM maps). Deterministic sorted split: train IXI_001–104, val IXI_105–130,
   test = the rest (the 6 non-IXI volumes are never trained on). Input = physics-
   simulated 64 mT (`FieldConverter 'combined'`); target = 4 segmentation-derived
   proxies (BTF, TCR, VBR, MCI; `utils/biomarker_extraction.py`), MSE loss.
   **It is supervised regression, not a denoising autoencoder.**
2. **Stage 2 — Supervised fine-tuning (OASIS-1, n=375, split 300/37/38).** Full
   fine-tune (Adam, lr 5e-5, no weight decay) on physics-simulated OASIS input
   (`'hyperfine'`) to regress the **OASIS-supplied nWBV** (spreadsheet column;
   Fotenos et al. 2005, FSL FAST atlas-based) — not FreeSurfer.
3. **Stage 3 — Cross-session LOOCV adaptation (ds006557, n=23).** A lightweight
   adapter (final LayerNorm + head, 769 params) is trained on one 64 mT session
   (HFC) of the non-held-out subjects and tested on the other session (HFE) of
   the held-out subject. Fully subject- and session-independent.

## Top-level folders

| Folder | Contents |
|---|---|
| `models/` | Model definitions. `baselines.py` holds `BaselineViT3D` (4.23M params, the primary model) and `BaselineCNN3D` (8.22M-param comparator, ~1.95× the ViT). `transformer_gnn/`, `uncertainty/` hold auxiliary/experimental modules. |
| `scripts/` | All runnable experiment/data scripts (36 files). Data download, physics simulation, training, LOOCV, ablations, figure generation, statistics. See "Key scripts" below. |
| `utils/` | Shared utilities. `field_conversion.py` = the physics 64 mT simulator (`FieldConverter`). Note: the simulator adds unseeded noise; seed explicitly when inputs must be reproducible (see `scripts/round3/e8_oasis_fixed_inputs.py`). |
| `paper_build/` | **Regenerates every table, figure and number in the paper** from `experiments/*/results.json` (`python -m paper_build`; numpy/scipy/matplotlib only). Outputs in `paper_build/out/`, including `BUILD_REPORT.md` with cross-checks. |
| `scripts/round3/` | Round-3 (Access-2026-45125) experiments E1–E8 and existing-data analyses; each writes `experiments/r3_*/results.json`. |
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
| `round3/` | Round-3 experiments E1–E8 (see `REPRODUCE.md`). |
| `legacy_figures/` | Superseded figure scripts from rounds 1–2 (contain synthetic fallbacks). Do not use; figures come from `paper_build/`. |

## Experiment results (`experiments/`)

Each subfolder has a `results.json`. Round-3 files (`r3_*`) carry a provenance
block (script, git commit, library versions, timestamp). Key files:

- `loocv_cross_session/` — headline ViT3D LN+head cross-session LOOCV (MAE 0.0134).
- `ablation_adapter/`, `ablation_lora/`, `multiseed_loocv/`, `dino_headline_loocv/` — adapter variants, seeds, DINO.
- `r3_e1_reproducibility/` — fold-independent session agreement, head-size and image-QC nuisance analysis.
- `r3_e2_nonneural_baselines/`, `r3_e2b_swin_verification/`, `r3_e2c_cnn3d_target/` — every readout under the cross-session design.
- `r3_e3_external/` — external cohort (van den Broek et al.), headline-recipe adapter, shift diagnostics.
- `r3_e4_physics_vs_blur/` — matched, paired physics-vs-blur rerun (3 seeds per arm).
- `r3_e6_time_budget/`, `r3_e7_conformal/`, `r3_e8_oasis_fixed_inputs/`, `r3_existing_data_analyses/`.

## Getting the data (not in git)

`data/` (~98 GB) holds public datasets, excluded from git:

1. **IXI** (Stage 1): `python scripts/download_ixi_data.py`
2. **OASIS-1** (Stage 2): register at oasis-brains.org; place under `data/oasis_processed/` and `data/oasis_raw/`.
3. **ds006557** (Stage 3, real 64 mT): OpenNeuro ds006557; place under `data/ds006557_data/`.
4. **van den Broek et al.** (external, Zenodo 10.5281/zenodo.15471394): place under `Website_dat/extracted/`.

Reference labels are committed (`experiments/fastsurfer_output/`, `experiments/fastsurfer_zenodo/`),
so no segmentation needs to be re-run.

## Reproducing results

See **`REPRODUCE.md`**. In short: `python -m paper_build` regenerates every table,
figure and number from the committed result files (no data needed);
`scripts/reproduce_all.sh` re-runs the experiments (needs `data/` and checkpoints).
Environment: Python 3.12.3, `requirements.txt` (pinned). All runs were on CPU.

## Reference labels (do not confuse)

- ds006557 and the external van den Broek cohort: FastSurfer `--seg_only` on the
  paired 3 T scan, nWBV = BrainSegVol / MaskVol. The CSV column named `etiv` in
  `experiments/fastsurfer_output/nwbv_ground_truth.csv` holds **MaskVol**.
- OASIS-1: OASIS-supplied nWBV (FSL FAST, atlas-based). SynthSeg+: (GM+WM)/TIV.

## Conventions for contributors / assistants

- New experiments write a `results.json` into their own `experiments/<name>/` folder.
- Figures and tables for the paper come only from `paper_build/` — never hardcode numbers;
  `python -m paper_build` refuses to build if it finds RNG use or typed-in results.
- Older figure scripts (`scripts/legacy_figures/`) contain synthetic fallbacks
  and are superseded by `paper_build/`; do not use them for manuscript figures.
- The physics simulator constants live in `utils/field_conversion.py`.
- Keep the paper's honest, feasibility-baseline framing: report negative and
  null results plainly (constant-mean baseline comparison, non-significant
  physics-vs-blur, miscalibrated uncertainty).
