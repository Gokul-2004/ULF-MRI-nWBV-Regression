# Direct nWBV Regression from 64 mT Ultra-Low-Field MRI

Code, result files and paper build for *Direct nWBV Regression from 64 mT
Ultra-Low-Field MRI: A Reproducible Feasibility and Failure-Analysis Baseline*
(Gokul Krishnan, Ganesh Khekare; Vellore Institute of Technology), submitted to
IEEE Access.

The study asks whether normalized whole-brain volume (nWBV) can be read directly
from a 64 mT Hyperfine Swoop T2-weighted scan, without segmenting it, and reports
plainly where this works and where it fails:

- a 3D Vision Transformer pretrained on simulated low-field data and adapted to
  real 64 mT scans does **not** beat a constant-mean predictor on the 23-subject
  ds006557 cohort;
- simple linear readouts of frozen CNN and Swin-UNETR features recover part of
  the nWBV signal on held-out sessions of the same cohort;
- the physics-based low-field simulator gives no measurable benefit over a
  Gaussian-blur degradation in a matched, paired comparison;
- every model fails on an independent 64 mT cohort whose reference nWBV lies
  outside the adaptation range.

All numbers are in the manuscript and in `paper_build/out/` (see below).

## Reproduce

| What | Command |
|---|---|
| Every table, figure and quoted number, from the committed results (minutes, no data) | `python -m paper_build` |
| Re-run the evaluations (needs the public datasets) | `scripts/reproduce_all.sh experiments` |
| Retrain the matched physics-vs-blur experiment and CNN3D | `scripts/reproduce_all.sh retrain` |

Full instructions, inputs, run times and known limits: **[REPRODUCE.md](REPRODUCE.md)**.

## Layout

| Path | Contents |
|---|---|
| `paper_build/` | Builds every table, figure and number of the paper from `experiments/*.json` |
| `experiments/` | One folder per experiment, each with a `results.json` (the record of every result) |
| `scripts/` | Data download, simulation, training and evaluation scripts |
| `scripts/round3/` | Experiments added for the resubmission (E1–E8) |
| `models/`, `utils/` | `BaselineViT3D`, `BaselineCNN3D`; the 64 mT simulator (`utils/field_conversion.py`) |
| `checkpoints/` | Trained weights (larger files are attached to the GitHub release) |
| `scripts/legacy_figures/` | Superseded figure scripts from earlier versions; not used |

`CLAUDE.md` describes the three-stage pipeline and the reference labels in more detail.

## Data

All datasets are public and are not redistributed here: IXI, OASIS-1, OpenNeuro
ds006557, and van den Broek et al. (Zenodo 10.5281/zenodo.15471394). Derived
reference labels (FastSurfer, SynthSeg+) are committed under `experiments/`.

## Citation

See `CITATION.cff`. The archived release has a Zenodo DOI (listed on the release page).
