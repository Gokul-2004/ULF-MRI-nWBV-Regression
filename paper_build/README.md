# paper_build

Regenerates every table, figure and quoted number of the manuscript from the
committed result files in `experiments/`. No MRI data, checkpoints or GPU needed.

```bash
pip install -r paper_build/requirements.txt   # Python 3.12
python -m paper_build            # writes paper_build/out/
python -m paper_build --strict   # also fails if an input file is not tracked by git
```

Outputs (`paper_build/out/`):

- `tables/*.md`, `tables/*.csv` — every table in the paper
- `figures/*.png` (600 dpi) and `figures/*.svg` — every data figure
- `numbers.json` — every number quoted in the text, with the file it comes from
- `BUILD_REPORT.md` — git commit, input files, and cross-checks in which each key
  result is recomputed independently and compared with the value saved by the
  experiment script that produced it

Guards: the build stops if any code here draws random numbers outside the seeded
bootstrap (`stats.py`, seed 42, 10,000 resamples) or contains a number with four
or more decimals (i.e. a typed-in result).

Where the inputs come from: `scripts/round3/` (experiments E1–E8 for the
Access-2026-45125 resubmission) and the earlier experiment scripts listed in the
repository `CLAUDE.md`. Re-running those needs the public datasets (IXI, OASIS-1,
OpenNeuro ds006557, van den Broek et al. on Zenodo) and the checkpoints.

Manuscript figures (`figures/manuscript/Figure_NN_*`, numbered as in the paper):
Figures 2–18 and 20–23 are rebuilt in their manuscript layout by
`kept_figures.py` from `kept.py`; Figure 24 is `figures/fig24_external_validation`.
The `figures/fig_*` files are the round-3 analysis figures that back the new
tables. Tables II, IV, VI, VIII and IX are `tables/kept_table*.md`.

Not regenerated, with the reason:
- Figure 1 — example MRI slices drawn from the raw scans (needs the image data).
- Figure 19 — MC Dropout intervals on OASIS-1: that run saved only its summary
  (`experiments/oasis_mc_dropout/results.json`: 23.7 % coverage, width 0.0284),
  not the per-subject intervals. The figure in the manuscript is from that run.
