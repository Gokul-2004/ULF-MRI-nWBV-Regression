# Figure defects and a data-integrity finding — Access-2026-28453

Checked against the 17-page PDF (`IEEE Access Format (2) (7) 1 (5).pdf`, exported 18 Aug).
Regenerated figures are in `Review - 2/figures_to_insert/`, built by
`scripts/fix_figures_r2.py` (600 dpi PNG + SVG).

---

## Part 1 — Layout defects (fixed)

### Figure 3 — three-stage training pipeline ✅ FIXED

The ViT3D box was drawn at `y=0.6` while all four other boxes sat at `y=1.2`
(`generate_all_figures.py` / `regen_updated_figures.py`, box tuple 4). It dropped out of
the row, and the arrow chain rendered as two arrows pointing into empty space. Sub-captions
were also x-offset by +0.25 from the boxes they labelled, and only four captions were drawn
for five boxes.

Fixed: all five boxes on one centre line, arrows between every consecutive pair, each
caption centred under its own box, STAGE 1/2/3 banners added over the three training
stages named in the manuscript caption.

### Figure 6 — nWBV by CDR stage ✅ FIXED

Two problems. The `μ=` labels were drawn at the group mean, directly on top of the data
points — `μ=0.752` rendered as `μ=(0752` with a marker over the digit, same for `μ=0.739`.
And the y-axis spanned 0.58–1.08 for data occupying 0.645–0.810, so roughly half the panel
was empty and the boxes were compressed.

Fixed: labels moved above each group clear of all points and whisker caps, y-limits fitted
to the data with headroom for the labels and significance bracket, raw points jittered
sideways, duplicate outlier marker suppressed (`showfliers=False` — the scatter already
draws every point once).

### Figure 20 — MC Dropout intervals, OASIS-1 ❌ NOT FIXED — see Part 2

The legend box sits on top of the data (covering subjects 0–16) and the y-axis runs
0.0–0.88 for data in 0.62–0.88, leaving ~70 % of the panel empty. **Both are moot: the
figure's data is synthetic.** See below.

---

## Part 2 — Data-integrity finding

Three figures draw MC Dropout confidence intervals from `np.random.normal` rather than from
model output. In `scripts/generate_all_figures.py`:

| Line | Function | Manuscript figure | Code |
|---|---|---|---|
| 433–434 | `fig8a_ci_oasis()` | Fig 20 | `ci_half = np.abs(np.random.normal(0.0145, 0.004, len(preds_s)))` — comment reads *"Simulate MC Dropout CIs (width ~0.029, poorly calibrated)"* |
| 474–475 | `fig8b_ci_real64mt()` | Fig 21 | `ci_half = np.abs(np.random.normal(0.0145, 0.003, len(preds_s)))` |
| 532–533 | `fig9_failure_analysis()` | Fig 22, panel D | `ci_widths = np.abs(np.random.normal(0.0145, 0.003, len(l_errs)))` then `r_ci, p_ci = stats.pearsonr(ci_widths, l_errs)` |

The rendered figures carry the hardcoded title strings from these functions verbatim
("Coverage: 4.3% (nominal: 95%); CIs not suitable for clinical use"; "CI width not
significantly correlated with error (r = +0.34, p = 0.108)"), which is what ties the
shipped images to these code paths.

### The reported statistic is wrong

`experiments/real64mt_eval/mc_dropout_ci.json` contains genuine per-subject MC Dropout
output (N = 100 passes, n = 23). Recomputing from it:

| Quantity | Manuscript | Real data | Status |
|---|---|---|---|
| Mean 95% CI width | 0.029 | 0.0286 | ✅ correct |
| Empirical coverage at nominal 95% | 4.3 % | 4.3 % | ✅ correct |
| MC Dropout MAE | 0.040 | 0.0403 | ✅ correct |
| **r (CI width vs abs. error)** | **+0.34** | **−0.164** | ❌ **wrong, sign flips** |
| **p** | **0.108** | **0.454** | ❌ **wrong** |

The headline conclusion — *not* significantly correlated — survives, and is in fact more
strongly supported (p = 0.454 rather than 0.108). But §V-L argues from the **sign**:

> "the positive but non-significant r value is consistent with the direction one would
> expect from a well-calibrated model and does not by itself indicate miscalibration."

The real correlation is **negative**: wider intervals go with *smaller* errors, the opposite
of calibrated behaviour. **That sentence is false and must be removed, not just renumbered.**

### Two further mismatches in the same area

1. **Figure 20's subtitle reports "Coverage: 4.3%"** — that is the *real-64mT* coverage from
   `mc_dropout_ci.json` (n = 23), hardcoded onto an OASIS-1 panel (n = 38). No OASIS coverage
   figure was ever computed.
2. **Figure 21 is labelled "LOOCV prediction (adapted)"**, but the only MC Dropout data in
   the repo has MAE 0.0403 — the *unadapted* ViT3D number from Table II ("ViT3D (no adapt.)
   0.040"). The MC run is on the unadapted model. The rebuilt figure is labelled accordingly.

### Precedent

This class of defect has already been caught once in this project: `generate_ieee_figures.py`
carries a FIG7_NOTE stating that the synthetic ellipse-and-noise "brain" placeholder produced
by `fig7_scan_comparison()` *"must never reach the manuscript"*, which is why
`figures_rebuilt/fig7_scan_comparison` exists. The same audit was not extended to the MC
Dropout panels.

---

## Part 3 — What was rebuilt from real data

### Figure 21 — MC Dropout, real 64 mT ✅ REBUILT

Now drawn from the real per-subject intervals. Legend moved below the axes, y-limits fitted.
The honest version is also the more persuasive one: the CI band and the true-nWBV trace
barely intersect, which is what 4.3 % coverage actually looks like. 21/23 subjects exceed
the 0.020 threshold.

### Figure 22, panel D ✅ REBUILT (standalone)

`fig22D_ci_width_vs_error_CORRECTED.png/.svg` — real widths vs real errors, r = −0.164,
p = 0.454.

Supplied standalone because the full four-panel Figure 22 cannot be rebuilt on this machine:
panel C plots error against subject age, and `data/ds006557_data/participants.tsv` is not
present here (only group-level means survive, in
`experiments/paper_statistics/age_stratified_error.json`). Rebuild the complete figure on
the machine that holds the dataset, using the corrected panel D.

---

## Part 4 — Required actions

**Blocking, before resubmission:**

1. **Correct r and p** wherever they appear — §V-L and the Failure Analysis (D) Calibration
   paragraph. `+0.34 / 0.108` → `−0.164 / 0.454`.
2. **Delete the well-calibrated-direction sentence** in §V-L. It depends on a positive sign
   that the data does not show.
3. **Decide Figure 20.** No OASIS-1 MC Dropout output exists anywhere in the repo. Either
   run MC Dropout on the OASIS-1 test set (N = 100 passes, same protocol as the 64 mT run)
   and rebuild it, or drop the figure together with the claim that mean interval width is
   0.029 "on both the OASIS-1 test set and the real 64 mT set" — as written, half that
   sentence has no measurement behind it.
4. **Rebuild the full Figure 22** with the corrected panel D, on the machine holding
   `participants.tsv`.

**Recommended:**

5. **Audit the remaining figures against their data sources.** Two independent instances of
   synthetic data reaching manuscript figures (fig7, and the three MC panels) is a pattern,
   not a one-off. `generate_review2_figures.py` and `..._part2.py` both open with docstrings
   asserting "nothing is simulated" — the older `generate_all_figures.py` makes no such
   claim and is the file at issue.
6. **Do not reuse `generate_all_figures.py`** for any manuscript figure without first
   stripping the three `np.random` interval blocks.
