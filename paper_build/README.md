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

Not regenerated here: Figure 1 (MRI slices; needs the image data).
