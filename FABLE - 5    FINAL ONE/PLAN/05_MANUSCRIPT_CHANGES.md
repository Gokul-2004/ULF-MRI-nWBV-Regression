# 05 — MANUSCRIPT CHANGES: the edit specification

**How to use this file.**

- **BEFORE** text is quoted exactly as it appears in
  `01_Manuscript_AS_SUBMITTED_Review1.pdf` (the version the reviewers saw). PDF
  extraction artifacts inside quotes are reproduced as printed and marked `[sic]`
  (e.g. "Theprimary", "A3D CNN"); when searching the Word document, match on the
  distinctive words, not the broken spacing.
- **AFTER** text is final and ready to paste. Do not improvise wording.
- **Location** = section name + page number in the submitted PDF. The current working
  Word document (`Review - 2/` lineage, 17-page "Fig update 6" export) already
  implements many of these edits from prior passes — where known, the edit carries a
  **WIP status** line. For every edit: *verify the AFTER state is present; apply it if
  not.* Where the WIP contains material that must be corrected or removed, that is a
  **W-series** edit in Part B.
- Every changed passage must be **yellow-highlighted** in the resubmitted highlighted
  PDF (IEEE requirement), including these edits.
- Unit convention: new text writes "64 mT", "0.020 nWBV", "47 ms" with a space
  (per the IEEE checklist and the pending unit-spacing pass); do not change the article
  title, which keeps "64mT" as registered.

**Verification gates before upload (not edits, but blocking):**

- **G1** — Age correlations ρ = −0.778 / ρ = +0.232 (Abstract, §V-E, Discussion,
  fig11 caption): re-derive on the machine holding `data/ds006557_data/participants.tsv`.
  If either changes, all four locations change together.
- **G2** — 47 ms GPU latency: confirm an author can vouch for the original measurement
  (no measurement file exists in the repo). If not vouched, replace "47 ms" claims with
  "a single forward pass of a 4.23 M-parameter model (measured at 267 ms on CPU for the
  92.7 M-parameter UNETR comparator; edge-device latency not benchmarked)" — do NOT
  invent a new number.
- **G3** — van den Broek et al. Zenodo citation: retrieve exact author list, year,
  version, DOI from the Zenodo record before inserting reference [42]; placeholder
  `[ZENODO CITATION — RETRIEVE DOI AND FULL AUTHOR LIST]` if unavailable, never an
  approximation.
- **G4** — fig22 (failure analysis) full rebuild requires `participants.tsv` (panel C);
  rebuild on the dataset machine with corrected panel D before export.

---

## PART A — Edits against the submitted manuscript (M-series)

### M1 | Tier 1 | Abstract (p. 1) | MUST

**BEFORE (three sentences to replace, quoted):**

> "Under fully subject-independent cross-session leave-one-subject-out cross-validation
> on real 64mT hardware (n = 23), the model achieves MAE = 0.0134nWBV [95% CI:
> 0.010–0.017] with 47ms inference. The cohort nWBV range is narrow (width = 0.056), so
> MAE is comparable to a constant-mean baseline (MAE = 0.0126); the study should be
> interpreted as a feasibility baseline rather than a deployed capability. Inter-session
> ICC (3,1) = 0.615 [0.236–0.866] provides a preliminary reliability estimate, though
> the wide CI precludes definitive classification at n = 23."

**AFTER:**

> Under fully subject-independent cross-session leave-one-subject-out cross-validation
> on real 64 mT hardware (n = 23), the model achieves MAE = 0.0134 nWBV [95 % CI:
> 0.010–0.017] with a measured inference time of <T> ms per volume on CPU [M29]. The cohort nWBV range is narrow (width = 0.056),
> and the model does not outperform a constant-mean predictor: the leave-one-out
> constant-mean baseline yields MAE = 0.0128 (global mean 0.0122), numerically below
> the model's 0.0134. The study should therefore be interpreted as a feasibility
> baseline rather than a deployed capability. The evidence that the model is
> input-dependent rather than a disguised constant predictor is a permutation test on
> the inter-session ICC (observed ICC(3,1) = 0.615 [0.236–0.866] against a null of
> approximately zero; p = 0.0024, 20,000 permutations); transparently, the reproducible
> per-subject signal does not track true nWBV (r = −0.119, p = 0.590). Multi-seed
> repetition (five seeds) gives MAE = 0.0130 ± 0.0004 and ICC = 0.644 ± 0.058.
> Split-conformal calibration restores near-nominal coverage (95.7 % at nominal 95 %)
> at a mean interval width of 0.066 nWBV, which exceeds the cohort's entire nWBV range —
> quantifying that individual-level nWBV is not resolvable at this sample size. All four
> reviewer-named self-supervised objectives and two larger architectures (Swin-UNETR,
> UNETR) were additionally evaluated: several outperform the adopted configuration on
> high-field OASIS-1 data, and this is reported plainly; the strongest high-field
> objective (DINO) degrades on real 64 mT hardware (LOOCV MAE 0.0220, 10/23 subjects
> below the 0.020 reference threshold, vs 0.0134 and 19/23 for the denoising objective).

**REVISED 2026-08-23 — two changes to the AFTER text above:**

1. **Strike "with 47 ms inference."** Replace with the measured CPU figure per M29,
   or delete the latency clause from the abstract entirely if the benchmark is not
   produced. The GPU claim has no measurement and contradicts `CLAUDE.md`.
2. **Add one external-validation sentence** before the multi-seed sentence, since the
   abstract may no longer imply that generalisation is untested:

   > "On an independent public 64 mT cohort (van den Broek et al., n = 10) with
   > FastSurfer ground truth derived by the same pipeline, the adapted model attains
   > MAE = 0.0731 against a cohort constant-mean baseline of 0.0089: its predictions
   > collapse to the adaptation cohort's mean (0.7824 versus 0.7841) and do not
   > extrapolate to that cohort's non-overlapping nWBV range."

**WHY:** R1.1, R3.4 (baseline arithmetic + two-sided input-dependence), R1.9, R1.7,
R2.2, R2.3 (comparators disclosed up front), R3.2 (feasibility framing), R1.3 (M30),
I10 (M29). Keep the
remaining abstract sentences (age correlation — pending G1; dementia stress test;
contributions sentence) as they are, except the contributions sentence is updated by
M3's item list. WIP status: partially present (0.0128 and comparator clause exist;
DINO clause queued by `WORD_AGENT_PROMPT_R2.3_SSL.md`; the transfer-probe clause must
be REMOVED — see W1). **Length note:** the abstract is near the IEEE Access word
ceiling; during M24 the sacrificial details are the "(10×; note: CDR=0.0 group MAE =
0.086, giving ~1.8× against the labelled healthy controls)" parenthetical (shorten to
"(10×)") and the Swin/UNETR r values (keep "several outperform … reported plainly",
drop the numerals) — never the constant-mean, permutation, or DINO-transfer
sentences.

### M2 | Tier 1 | §II-C Vision Transformers in Medical Imaging (p. 2) | MUST

**BEFORE:**

> "For global regression tasks such as nWBV — where the target is a ratio over the
> entire brain volume — ViT's global self-attention mechanism over patch tokens is
> architecturally well-suited."

**AFTER:**

> For global regression tasks such as nWBV — where the target is a ratio over the
> entire brain volume — a Vision Transformer is one plausible compact architecture;
> whether its global self-attention confers any advantage for this task is an empirical
> question, and the comparisons reported in Section V (CNN3D, Swin-UNETR, UNETR) do not
> establish one on high-field data.

**WHY:** R2.1 — removes the untested mechanism claim R2 quoted back at us.

### M3 | Tier 2 | §I Contributions list (p. 2) | MUST

**BEFORE (items 1, 2 and 4 quoted; item 3 unchanged):**

> "1. To our knowledge, the first direct nWBV regression from 64mT MRI without
> intermediate segmentation or super-resolution, evaluated under fully
> subject-independent cross-session LOOCV (n = 23), yielding MAE = 0.0134
> [0.010–0.017] and compute-light inference (47ms on a standard GPU; hardware not yet
> benchmarked on edge devices)."
>
> "2. A physics-constrained 64mT simulation recipe (T1/T2 tissue remapping, Rician
> noise, B0 inhomogeneity) that yields a small positive trend over Gaussian-blur
> degradation (Δr = +0.018, bootstrap 95% CI [−0.021, +0.057], p = 0.847) — not
> statistically significant at n = 75, but provided as a reproducible baseline for
> future larger-scale comparison."
>
> "4. An honest failure characterization: MAE is comparable to a constant-mean baseline
> (MAE=0.0126) given the narrow cohort range; CDR=1.0 stress-testing (n=2, directional)
> shows 10×MAE increase, bounding the validated scope to healthy adults."

**AFTER (full replacement list; item 3 re-stated for completeness, unchanged in
substance):**

> 1. To our knowledge, the first direct nWBV regression from 64 mT MRI without
>    intermediate segmentation or super-resolution, evaluated under fully
>    subject-independent cross-session LOOCV (n = 23), yielding MAE = 0.0134
>    [0.010–0.017] and compute-light inference (4.53 ms per volume on CPU,
>    measured; no GPU required; edge devices not yet benchmarked). Within-cohort
>    accuracy does not exceed a constant-mean
>    baseline (leave-one-out MAE = 0.0128); the discriminative evidence is a
>    permutation test establishing input-dependence (p = 0.0024), reported together
>    with the negative finding that the reproducible signal does not track true nWBV
>    (r = −0.119, p = 0.590), and with the external finding that the adapted
>    estimator does not transfer to a cohort whose nWBV range differs from the
>    adaptation set (Section V-J). The contribution is the protocol and the
>    characterisation, not a transportable nWBV predictor.
> 2. A physics-constrained 64 mT simulation recipe (T1/T2 tissue remapping, Rician
>    noise, B0 inhomogeneity) with quantitative fidelity validation against real 64 mT
>    acquisitions. The physics simulator is not superior to a Gaussian-blur baseline on
>    image fidelity (NCC 0.497 vs 0.516, paired Wilcoxon p = 0.02 in blur's favour),
>    and the two are statistically equivalent on the downstream nWBV task (MAE 0.0146
>    vs 0.0148, unpaired Mann-Whitney p = 0.81, n = 75 per arm). The recipe is provided
>    as a reproducible baseline, not a demonstrated improvement.
> 3. A leakage-free subject-independent cross-session LOOCV protocol yielding a
>    preliminary inter-session reliability estimate ICC(3,1) = 0.615 [0.236–0.866] at
>    n = 23, with multi-seed stability (MAE 0.0130 ± 0.0004, ICC 0.644 ± 0.058).
> 4. A transparent component-level evaluation and failure characterization: CNN3D
>    significantly outperforms the adopted ViT3D on OASIS-1 (p = 0.0037), as do larger
>    architectures (Swin-UNETR, UNETR) and three of four alternative self-supervised
>    objectives — all reported without qualification — while the strongest high-field
>    objective (DINO) degrades on real 64 mT hardware under the identical adaptation
>    protocol. CDR = 1.0 stress-testing (simulated, n = 2, directional) shows a 10×
>    MAE increase, bounding the validated scope to healthy adults.

**WHY:** R1.1 (item 1), R1.4/R3.7/I2 (item 2 — kills the phantom p = 0.847 and the
invalid Δr CI), R1.9 (item 3), R2.1/R2.2/R2.3/R1.6 (item 4). Tier-2 justification for
the letter: the contribution list is rewritten downward, replacing component-optimality
claims with the evaluated-and-reported record.

### M4 | Tier 0 | §IV-B Baselines (p. 4) | MUST

**BEFORE:**

> "CNN3D: A3D CNN [sic] with 4 convolutional blocks, global average pooling, and
> ≈ 4.1M parameters, trained on the same OASIS-1 split and hyperparameters as ViT3D.
> Evaluated on both OASIS-1 and real 64mT data."

**AFTER:**

> CNN3D: A 3D residual CNN with 4 convolutional stages, global average pooling, and
> ≈ 8.22 M parameters (8,222,337; roughly twice the ViT3D's 4.23 M), trained on the
> same OASIS-1 split and hyperparameters as ViT3D. Evaluated on both OASIS-1 and real
> 64 mT data.

**WHY:** Integrity I7 — the committed model definition counts 8,222,337 parameters; the
"≈ 4.1 M" figure is wrong. Sweep the document for any "matched-parameter" phrasing
about CNN3D and delete it (none found in the submitted PDF body, but the fig05 bar
label "CNN3D 4.1M" must change — see 07_FIGURE_PLAN). Also fixes "A3D" typo.

### M5 | Tier 0 | §V-I Uncertainty Quantification (p. 8) | MUST

**BEFORE:**

> "The mean 95% interval width is 0.029nWBV on both the OASIS-1 test set (Fig. 14) and
> the real 64mT cohort (Fig. 15). Empirical coverage at the nominal 95% level is 4.3%
> (Fig. 17), and interval width is not significantly correlated with absolute
> prediction error (r = +0.34, p =0.108, n = 23). Theprimary [sic] evidence of
> miscalibration is the 4.3% empirical coverage against a nominal 95% target; the
> positive but non-significant r value is consistent with the direction one would
> expect from a well-calibrated model and does not by itself indicate miscalibration."

**AFTER:**

> On the OASIS-1 test set (n = 38), the mean 95 % interval width is 0.0284 nWBV with
> empirical coverage of 23.7 % (9/38) at the nominal 95 % level (Fig. 20). On the real
> 64 mT cohort (n = 23, unadapted model), the mean width is 0.0286 nWBV with empirical
> coverage of 4.3 % (Figs. 21 and 23). Interval width is not significantly correlated
> with absolute prediction error (r = −0.164, p = 0.454, n = 23). The evidence of
> miscalibration is the empirical coverage against the nominal 95 % target on both
> datasets; the width–error correlation is weak, negative and non-significant, so
> interval width carries no usable information about which predictions are unreliable.

Figure citations in the AFTER text use the **revised** numbering (submitted Fig. 14 →
revised Fig. 20; Fig. 15 → 21; Fig. 17 → 23; see 07_FIGURE_PLAN).

**WHY:** Integrity I3 + I4 (r sign flip; the OASIS half of the width claim previously
had no measurement — now it does: `oasis_mc_dropout/results.json`). **The deletion of
the "consistent with the direction one would expect from a well-calibrated model"
clause is the point of this edit** — the real correlation is negative, so that
reasoning is false; do not replace it with a directional argument the other way.
Serves R1.7. WIP status: queued by `WORD_AGENT_PROMPT_AUDIT_TEXT.md` Edit 1 with
identical intent; the 23.7 % OASIS sentence is additional (post-dates that prompt).

### M6 | Tier 0 | §V-J Failure Analysis, part (D) Calibration (p. 9) | MUST

**BEFORE:**

> "(D) Calibration. MC Dropout CI width is not significantly correlated with absolute
> error (r = +0.34, p = 0.108, n = 23)."

**AFTER:**

> (D) Calibration. MC Dropout CI width is not significantly correlated with absolute
> error (r = −0.164, p = 0.454, n = 23).

**WHY:** Integrity I3. The following reliability-curve sentence is correct — leave it.
Also apply the same r/p substitution to the caption of submitted Fig. 15 (revised
Fig. 21): "r = +0.34, p = 0.108" → "r = −0.164, p = 0.454"; change nothing else in
that caption.

### M7 | Tier 0 | §V-B CDR Stratification (p. 5) | MUST

**BEFORE:**

> "Cohen's d (CDR 0.0 vs. CDR 1.0) = 1.368 (large effect; interpret cautiously at
> n = 2)."

**AFTER:**

> Cohen's d (CDR 0.0 vs. CDR 1.0) = 1.463 (large effect; interpret cautiously at
> n = 2).

**WHY:** Integrity I6 — 1.368 does not reproduce as Cohen's d (it approximates Glass's
delta); pooled-SD Cohen's d recomputes to 1.4625. Interpretation unchanged.

### M8 | Tier 0 | §V-C Ablation: Physics Simulation vs Gaussian Blur (pp. 5–6) | MUST

**BEFORE (body):**

> "A paired bootstrap test on the same 75 subjects finds this difference is not
> statistically significant (95% CI on Δr: [−0.021, +0.057]; Wilcoxon p = 0.847). The
> result is a small positive trend rather than a confirmed superiority; the comparison
> is retained as a reproducible recipe baseline."

**AFTER:**

> Because the physics and blur arms were pre-trained and evaluated on different,
> independently drawn test splits, neither a paired test nor a paired bootstrap
> interval is valid for this comparison, and the paired statistics reported in the
> previous version of this manuscript are withdrawn. An unpaired comparison on the
> downstream nWBV task (n = 75 per arm) gives MAE 0.0146 vs 0.0148, Mann-Whitney
> p = 0.81 (not significant). The per-arm correlations (r = 0.949 vs 0.931) are
> descriptive only. The comparison is retained as a reproducible recipe baseline, not
> a demonstrated superiority; a shared-split paired re-run is identified as future
> work.

**BEFORE (Table III footnote):**

> "*Bootstrap 95 % CI; Wilcoxon p=0.847 — difference not significant."

**AFTER:**

> *Unpaired Mann-Whitney on downstream nWBV MAE (n = 75/arm): p = 0.81, not
> significant. Arms used different test splits; a paired test or paired CI is not
> valid for this comparison.

Also delete the stray struck-through "Notcom" editing artifact adjacent to this
footnote (integrity; `MANUSCRIPT_AUDIT.md` D7).

**WHY:** Integrity I2 (the p = 0.847 and the Δr CI are not reproducible;
`significance_tests/results.json` T3 documents both the invalidity and the valid
unpaired result). Serves R1.8, R3.7.

### M9 | Tier 2 | §III-A Physics-Constrained 64 mT Simulation — add fidelity validation (p. 3) | MUST

**Insert after the Step 4 paragraph ("…Representative axial slices are shown in
Fig.1."), as a new paragraph + small table:**

> **Simulation fidelity versus real 64 mT data.** To quantify how closely the simulated
> volumes match real Hyperfine acquisitions, each simulated volume was compared with
> the paired subject's real 64 mT scan (n = 23), against a Gaussian-blur-plus-histogram-
> matching baseline [30]. The physics simulator attains NCC = 0.497 ± 0.046 versus
> 0.516 ± 0.056 for the blur baseline — a small but statistically significant advantage
> for blur (paired Wilcoxon p = 0.02) — and its SSIM is likewise lower (≈ 0.07–0.09 vs
> ≈ 0.14–0.22). The simulator's SNR (≈ 33) is deliberately conservative against the
> scanner's effective post-reconstruction SNR (≈ 320–350, achieved through ETL-80
> averaging and compressed sensing that we do not model), so its SNR error is far
> larger than the baseline's (p < 0.001). The physics simulator is therefore not
> superior on image fidelity; its role is to provide a conservative, physically
> parameterised stress-training input, and its value is assessed on the downstream
> task (Section V-C), where the two degradations are statistically equivalent.

**WHY:** R1.4 (quantitative validation with image-similarity metrics — delivered, with
the unfavourable outcome reported as measured), R3.7, R4.4 (this is the image-quality
comparison that exists for the one image-producing component). Source:
`sim_validation/results.json` + `significance_tests` T2. Tier-2 justification: promotes
an existing-but-unused experiment into Methods.

### M10 | Tier 2 | §V — new subsection after §V-E (Cross-Session LOOCV Results) | MUST

**Insert new subsection "Input-Dependence (Permutation Test)":**

> To test whether the adapted model behaves as a constant predictor, subject labels
> were permuted 20,000 times (seed 42) and the inter-session ICC recomputed under each
> permutation. The observed ICC of 0.6146 lies far in the tail of the null
> distribution (null mean ≈ 0), p = 0.0024: per-subject predictions agree across two
> independent acquisition sessions far beyond chance, which a constant or
> range-compressed mean predictor cannot produce. Transparently, the same analysis
> shows the reproducible signal does not track true nWBV (prediction-versus-truth
> Pearson r = −0.119, p = 0.590; the model's prediction SD of 0.004 spans roughly a
> quarter of the true between-subject SD of 0.015). The model is reproducible without
> being accurate in the biological direction — consistent with the feasibility-baseline
> framing, and reported as the boundary of what this study demonstrates. (Fig. 14 in
> the revised numbering.)

**WHY:** R1.1, R3.4, R3.5 — this is the paper's only discriminative evidence and its
honest limit, in one paragraph. Source: `permutation_test/results.json`.

### M11 | Tier 2 | §V-I Uncertainty Quantification — conformal calibration | MUST

**BEFORE (two passages to replace):**

> "Calibrated predictive intervals require either (i) a held-out real-hardware
> calibration set with sufficient pathology coverage, or (ii) split conformal
> prediction [41] computed on prospective acquisitions. Neither is feasible within the
> present ds006557 cohort: setting aside calibration data from n = 23 would shrink the
> LOOCV evaluation set below the threshold for meaningful bootstrap inference."

and

> "Conformal calibration is therefore deferred to the prospective multi-site validation
> already identified as the next step (Section 6.5). MC Dropout intervals must not be
> used for individual-level clinical decisions on the basis of this paper."

**AFTER (single replacement block):**

> Calibrated predictive intervals were obtained with leave-one-out split-conformal
> prediction [41] computed on the subject-independent LOOCV residuals, which requires
> no separate calibration split. Conformal calibration restores near-nominal coverage:
> the 90 % interval attains 91.3 % empirical coverage (mean width 0.0571 nWBV) and the
> 95 % interval 95.7 % (mean width 0.0664 nWBV), against 4.3 % for MC Dropout at the
> same nominal level. The cost is width: conformal intervals are roughly twice as wide
> as the (miscalibrated) MC Dropout intervals, and the calibrated 95 % width (0.066)
> exceeds the entire cohort's nWBV range (0.056). This quantifies, rather than merely
> asserts, that individual-level nWBV is not resolvable at this sample size and field
> strength; calibrated intervals remain a group-level tool. MC Dropout intervals must
> not be used for individual-level clinical decisions on the basis of this paper.

**WHY:** R1.7 ("improve this analysis" — done), R3.7, R1.2 (the width-exceeds-range
sentence is the sharpest sample-size statement available). Source:
`conformal_calibration/results.json`. Also update the Limitations "Uncertainty
calibration" paragraph (p. 10): replace its last sentence
("Split conformal calibration on prospective acquisitions is the appropriate remedy
and is identified as future work.") with:

> Split-conformal calibration on the LOOCV residuals restores nominal coverage at
> roughly twice the interval width (Section V); prospective recalibration on each
> deployment site remains necessary before any clinical use.

### M12 | Tier 2 | §V — new short subsection after the ICC subsection (§V-F) | MUST

**Insert "Multi-Seed Robustness":**

> The full cross-session LOOCV protocol was repeated across five random seeds (42, 1,
> 7, 123, 2024). Across seeds, MAE = 0.0130 ± 0.0004 (range 0.0123–0.0134) and
> ICC(3,1) = 0.644 ± 0.058; the headline values (0.0134, 0.615) fall inside the seed
> spread, so neither is a seed artifact. The two quantities are not equally stable:
> MAE varies by 0.0004 across seeds while ICC varies by 0.058, with one seed yielding
> ICC = 0.563. The reliability estimate is the more seed-sensitive quantity at this
> sample size and should be read together with its wide bootstrap interval.
> (Fig. 8 in the revised numbering.)

**WHY:** R1.9, R1.2. Source: `multiseed_loocv/results.json`.

### M13 | Tier 2 | §V — SSL pretraining comparators + transfer to 64 mT | MUST

Two additions (already specified in `WORD_AGENT_PROMPT_R2.3_SSL.md`, adopted here with
one change — the retention paragraph's cross-reference to the removed transfer probe
becomes a cross-reference to the architecture comparators):

**(a) Comparator table** (new table in the architecture/pretraining-comparators
subsection): DINO 0.0270 / r 0.9032 · MAE(masked autoencoder) 0.0319 / 0.7561 ·
SimMIM 0.0538 / 0.9069 · Denoising (this paper) 0.0584 / 0.7220 · Contrastive
(SimCLR) 0.0589 / 0.7442. Caption:

> SELF-SUPERVISED PRETRAINING COMPARATORS ON OASIS-1 (n = 38). IDENTICAL PROTOCOL:
> ViT3D ENCODER, PHYSICS-SIMULATED IXI INPUT, 25 PRETRAIN EPOCHS, OASIS SEED-42
> FINETUNE, SAME TEST SPLIT.

**(b) New subsection "Pretraining-objective transfer to real 64 mT":**

> To test whether the OASIS-1 pretraining advantage transfers to the target regime,
> the strongest comparator (DINO) was carried through the identical cross-session
> LOOCV protocol used for the headline result: the same 769-parameter LayerNorm + head
> adapter, the same 23 folds, the same session assignment. DINO attains MAE 0.0220
> with bias +0.0085 and ICC(3,1) = 0.523, against 0.0134, −0.0022 and 0.615 for the
> denoising objective. Only 10 of 23 subjects fall below the 0.020 reference threshold
> under DINO, against 19 of 23 under denoising: the objective that is more than twice
> as accurate on high-field data crosses back above the reference threshold on real
> hardware. A paired comparison of per-subject absolute errors gives a bootstrap MAE
> difference of −0.0086 [95 % CI −0.017, −0.001] favouring denoising, though the
> paired Wilcoxon signed-rank test does not reach significance at this sample size
> (p = 0.070; denoising lower in 14/23 subjects). The threshold crossing and the
> reliability gap, rather than the mean difference, are the substantive result. This
> indicates that high-field representation quality is not predictive of low-field
> performance in this setting.
>
> The denoising objective is retained on three grounds, none of which is a claim of
> superior representation quality. First, its pretext task is the deployment domain
> gap itself: reconstructing high-field structure from physics-simulated 64 mT input
> trains the encoder to treat Rician noise and B0 inhomogeneity as confounders rather
> than anatomy (Section III). MAE, SimMIM, DINO and contrastive learning are
> domain-agnostic objectives that do not encode this prior; this was the design
> rationale before any comparator was run. Second, it is the only objective carried
> through the complete protocol — cross-session LOOCV, permutation testing, conformal
> calibration, adapter ablation, multi-seed stability and failure analysis — and
> substituting Stage 1 would require re-running that entire protocol per objective.
> Third, the single comparator that was evaluated on real hardware was the strongest
> on high-field data and performed worse on the target.
>
> Three limitations bound this comparison. Only DINO was carried through the 64 mT
> protocol; MAE and SimMIM were not, and extending the transfer test to them is
> identified as immediate future work. The DINO LOOCV run uses a single seed, whereas
> the denoising headline is reported across five; DINO's own seed variance is
> unmeasured, though the 0.0086 gap is large relative to the denoising seed standard
> deviation of 0.0004. All comparators use a fixed 25-epoch pretraining budget that
> was not tuned per objective.

**WHY:** R2.3 (all four named methods now tested and reported, three of them beating
the paper's objective on OASIS — the concession converts to a transfer finding), R1.5.
Sources: `ssl_comparator_{dino,mae,simmim,contrastive}`, `dino_headline_loocv`,
paired stats re-verified for this plan. **Hard constraints:** never claim denoising
superiority over the three objectives not tested at 64 mT; never write that DINO is
significantly worse on MAE (p = 0.070); never imply the retention decision post-dates
the comparators.

### M14 | Tier 2 | §V — architecture comparators subsection | MUST

**Insert (or verify, in WIP) "Architecture and Pretraining Comparators (OASIS-1
context)":**

> At the reviewers' request, two of the named larger medical Vision Transformer
> architectures were evaluated under the identical OASIS-1 protocol (same seed-42
> split, optimiser, schedule, early stopping): Swin-UNETR (62.2 M parameters, feature
> encoder + global average pooling + linear head) attains r = 0.964, MAE = 0.0148;
> UNETR (92.7 M) attains r = 0.923, MAE = 0.0198 (CPU inference 267 ms). Both
> substantially outperform ViT3D (r = 0.722, MAE = 0.0584) on high-field data, as does
> CNN3D (Table II); we report this plainly and do not claim ViT3D is the best
> architecture for this task. ViT-V-Net was not evaluated. These comparators were
> evaluated on OASIS-1 only and were not carried through the Stage-3 cross-session
> protocol; this evaluation-depth asymmetry is stated as a limitation (Section VI).
> (Fig. 5 in the revised numbering.)

**WHY:** R2.2, R1.5, R4.6 (quantitative SOTA expansion). Sources:
`arch_comparator_swin`, `arch_comparator_unetr`. Point estimates only (bootstrap CIs
are RNG-dependent; see 02 §E).

### M15 | Tier 2 | §V — adapter strategy ablation | MUST

**Insert (or verify) subsection "Adapter Strategy Ablation" with table:**

| Strategy | Trainable params | MAE [95 % CI] |
|---|---|---|
| Head-only | 257 | 0.0133 [0.0099–0.0170] |
| LayerNorm + head *(used in headline)* | 769 | 0.0137 [0.0104–0.0172] |
| LoRA (r = 4) | 41,217 | 0.0128 [0.0099–0.0160] |
| Full fine-tune | 4,225,537 | 0.0123 [0.0091–0.0158] |

**Required note under the table:**

> All four 95 % CIs overlap; no strategy is significantly superior at n = 23.
> LayerNorm + head is retained for parameter economy and training stability, not
> because it minimises error — it is not the lowest point estimate. The headline LOOCV
> run (LayerNorm + head) reports MAE 0.0134; this ablation's LayerNorm + head arm
> reports 0.0137. These are independent training runs under the same protocol and
> seed, differing in stochastic augmentation order, and fall within each other's
> confidence intervals; both are reported rather than reconciled silently.

**WHY:** R2.4 (including the LoRA arm R2 named), R1.5, R3.5. Sources:
`ablation_adapter`, `ablation_lora`. Do not write that LoRA "recovers" any benefit
(its point estimate equals the constant-mean baseline).

### M16 | Tier 0 | Caption of submitted Fig. 14 = revised Fig. 20 (OASIS MC Dropout) (p. 8) | MUST

**BEFORE:**

> "Per-subject nWBV predictions with MC Dropout 95% CI — OASIS-1 test set (n = 38).
> The model is severely underestimated and not suitable for clinical use."

**AFTER:**

> Per-subject nWBV predictions with real MC Dropout 95 % CI (N = 100 stochastic
> passes) — OASIS-1 test set (n = 38). Empirical coverage 23.7 % (9/38) at nominal
> 95 %, mean width 0.0284 nWBV: intervals are severely under-covered and not suitable
> for clinical use.

**WHY:** Integrity I4/I5 — the old caption carried the real-64 mT 4.3 % coverage
hardcoded onto an OASIS panel; the figure is rebuilt from
`oasis_mc_dropout/results.json`. The 4.3 % figures that refer to the real-64 mT
calibration curve elsewhere are correct and stay. Also fix Fig. 15's label: it plots
the **unadapted** model (MC MAE 0.0403), not "LOOCV prediction (adapted)" — the
rebuilt figure is labelled accordingly (see 07_FIGURE_PLAN).

### M17 | Tier 1 | Dementia wording audit (Abstract, §V-G, §VI, Conclusion) | MUST

Sweep: every sentence reporting a CDR-stratified result must carry all three
qualifiers — **simulated** inputs, **n = 2** at CDR = 1.0, **directional rather than
conclusive** — and no sentence may read as a screening or diagnostic capability. The
submitted text is already close (verify each occurrence); the one addition:

**In §V-G, after "…that the current model is not appropriate for dementia patient
assessment…":** append

> The model is not suitable for dementia screening, pathological morphometry, or
> longitudinal clinical monitoring in its current form.

**WHY:** R1.6, R3.6 — adopts R3's exact enumeration verbatim.

### M18 | Tier 2 | §VI Discussion — rewrite subsection B + add scope subsection | MUST

**(a) Retitle §VI-B** from "ARCHITECTURE CHOICE: ViT3D vs. CNN3D" to
"Architecture Choice: A Tractability Trade-off", and within it:

**BEFORE (two passages to replace):**

> "…whereas ViT3D's global self-attention produces more stable absolute-value
> predictions in the narrow-range regime." *(§V-D wording — apply the same fix there)*

> "This finding is consistent with prior reports that Vision Transformers exhibit
> lower texture bias and stronger global-shape sensitivity than CNNs [15]; both are
> advantageous when the target is a global volumetric ratio and the local texture is
> degraded by Rician noise."

**AFTER (respectively):**

> …whereas ViT3D's smaller output variance yields lower absolute error in the
> narrow-range regime. This is a descriptive property of the trained models, not an
> architectural mechanism we have tested.

> We do not attribute this to an architectural mechanism: the comparisons in Section V
> show the gains on real 64 mT hardware come from the adaptation procedure and the
> narrow-range regime, not from the transformer architecture, which on high-field data
> is outperformed by CNN3D (p = 0.0037), Swin-UNETR and UNETR. ViT3D is the system
> carried through the complete protocol because its 4.23 M parameters made the full
> cross-session LOOCV, multi-seed, permutation and calibration protocol tractable on
> the study's CPU-only hardware (approximately 20 h, against an estimated ~13 days per
> arm for Swin-UNETR); it was not selected on accuracy.

**(b) Add a new short subsection at the end of §VI (before Limitations):
"What This Study Does Not Claim":**

> For clarity of scope: this study does not claim that the model outperforms a
> constant-mean predictor on absolute error (it does not: 0.0134 vs 0.0128); does not
> claim that Vision Transformers provide better feature learning than CNNs (CNN3D is
> significantly better on OASIS-1); does not claim that the denoising pretraining
> objective is superior to modern self-supervised alternatives (three of four tested
> alternatives outperform it on high-field data); does not claim that the physics
> simulator is more faithful to real 64 mT data than simpler degradations (it is not,
> p = 0.02 against); does not claim clinical readiness, dementia-screening capability,
> or calibrated individual-level uncertainty. What it demonstrates is a reproducible,
> fully subject-independent protocol on real 64 mT hardware; threshold-level MAE with
> a session-stable, input-dependent output (permutation p = 0.0024) that does not yet
> track anatomy; and a transparent map of the failure modes — atrophy, calibration,
> and high-field-to-low-field transfer — that any successor system must clear.

**WHY:** R2.1 (concession made structural), R3.2 (feasibility framing made explicit),
R1.1. Tier-2 justification for the letter: "a scope-boundary subsection was added so
that no reader can over-read the contribution."

### M19 | Tier 1 | §VI-E Limitations — sample size / external validity | MUST

**BEFORE:**

> "Sample size and dataset availability: Real-hardware evaluation uses ds006557
> (n = 23), the only public paired 64mT/3T brain MRI dataset at the time of writing.
> The sample size therefore reflects the state of publicly available ULF data rather
> than a study design choice."

**AFTER:**

> Sample size and dataset availability: Real-hardware adaptation uses ds006557
> (n = 23), the largest public paired 64 mT/3T brain MRI dataset at the time of
> writing. A second paired 64 mT/3T cohort is publicly available (van den Broek et
> al., 11 paired subjects, 10 with complete 3T segmentation) [42]; we derive
> FreeSurfer-equivalent nWBV ground truth for it with the same FastSurfer pipeline
> used throughout this study and report external validation on it in Section V-J.
> Both public cohorts contain only healthy adults, so no existing public data can
> test the model across a pathological nWBV range. The sample size available to this
> study therefore reflects the state of publicly available ULF data rather than a
> study design choice.

**WHY:** Integrity I8 (the "only" claim is false), R1.3, R3.5 (wider-range
impossibility stated with its reason), R1.2. Requires gate G3 (citation). Sweep the
whole document for "only public", "only dataset", "only paired", "no second" — zero
assertions of sole existence may survive.

**REVISED 2026-08-23:** external validation on that cohort was subsequently RUN
(`experiments/zenodo_external_validation/`, commit e389ba6) with FastSurfer nWBV
ground truth derived by the same pipeline as OASIS-1/ds006557. This edit therefore
no longer declines the validation — it points to it. See the new M30 for the results
subsection and fig24 for the figure.

### M20 | Tier 0 | §VI-F Future Works (p. 10) | MUST

**BEFORE:**

> "(i) Retraining on OASIS-3 to improve atrophy-range coverage; (ii) Modelling the
> full ETL acquisition chain in simulation to close the SNR gap; (iii) Prospective
> paired 64mT/3T acquisition from CDR-rated patients; (iv) Three-way adapter strategy
> ablation (headonly vs. LN+head vs. full fine-tune); (v) Conformal prediction for
> calibrated uncertainty quantification; (vi) Multi-site validation across Hyperfine
> scanner generations."

**AFTER:**

> (i) Retraining on OASIS-3 to improve atrophy-range coverage; (ii) modelling the full
> ETL acquisition chain in simulation to close the SNR gap; (iii) prospective paired
> 64 mT/3T acquisition from CDR-rated patients; (iv) multi-site validation across
> Hyperfine scanner generations; (v) acquisition or curation of a paired ULF cohort
> whose nWBV range overlaps the adaptation cohort, which the external validation of
> Section V-J identifies as the binding constraint on generalisation; (vi) carrying
> the remaining self-supervised objectives (MAE, SimMIM) and the larger architectures
> (Swin-UNETR, UNETR) through the complete cross-session 64 mT protocol; (vii) a
> shared-split paired re-run of the physics-versus-blur pre-training comparison.

**WHY:** Integrity I9 — the old items (iv) adapter ablation and (v) conformal
prediction are now completed results in this same paper and must leave Future Works;
replacements record what genuinely remains. Serves R1.3, R2.2, R2.3 limitations.

**REVISED 2026-08-23:** external validation also leaves Future Works — it is now a
result (M30). Item (v) is replaced by the constraint that validation exposed: no
public paired ULF cohort overlaps the adaptation cohort's nWBV range.

### M21 | Tier 2 | §V — statistical comparisons summary table | MUST

**Insert (or verify) a "Statistical Comparisons" table near the end of §V:**

| # | Comparison | Test | n | Result |
|---|---|---|---|---|
| T1 | ViT3D vs CNN3D, OASIS-1 | paired Wilcoxon | 38 | MAE 0.0584 vs 0.0243, p = 0.0037 (CNN3D better) |
| T2 | Physics vs blur, image fidelity | paired Wilcoxon | 23 | NCC 0.497 vs 0.516, p = 0.02 (blur better); SNR error p < 0.001 (blur closer) |
| T3 | Physics vs blur, downstream nWBV | unpaired Mann-Whitney | 75/arm | MAE 0.0146 vs 0.0148, p = 0.81 (n.s.) |
| T4 | Denoising vs DINO pretraining, real 64 mT LOOCV | paired Wilcoxon | 23 | MAE 0.0134 vs 0.0220, p = 0.070 (n.s.); bootstrap ΔMAE −0.0086 [−0.017, −0.001] |
| P1 | Inter-session ICC vs permutation null | permutation, 20,000 | 23 | ICC 0.6146, p = 0.0024 |
| P2 | Prediction vs true nWBV | permutation, 20,000 | 23 | r = −0.119, p = 0.590 (n.s.) |

**Required sentence under the table:**

> Two of these comparisons favour a baseline over the adopted configuration (T1, T2)
> and one of the two significant permutation results is a null against the model (P2);
> all are reported as measured. T4 favours the adopted configuration directionally but
> does not reach significance at this sample size.

**WHY:** R1.8 (the literal request), R2.1, R2.3. Source: `significance_tests/`,
`permutation_test/`, T4 re-verified for this plan.

### M22 | Tier 1 | §VII Conclusion (pp. 10–11) | MUST

**BEFORE (two sentences to replace):**

> "…MAE = 0.0134nWBV [0.010–0.017] — entirely below the 0.020 clinical reference
> threshold, but comparable to a constant-mean LOOCV baseline (MAE = 0.0126) given the
> narrow ds006557 nWBV range."

> "Physics-constrained simulation shows a small positive trend over Gaussian-blur
> degradation (Δr = +0.018, not statistically significant at n=75, bootstrap 95 % CI
> [−0.021, +0.057]), and SynthSeg+ pseudo-labels are a viable deployment substitute
> for FastSurfer ground truth (MAE penalty +0.005)."

**AFTER (respectively):**

> …MAE = 0.0134 nWBV [0.010–0.017] — entirely below the 0.020 clinical reference
> threshold, but not smaller than a constant-mean LOOCV baseline (leave-one-out
> MAE = 0.0128; global 0.0122) given the narrow ds006557 nWBV range. A permutation
> test establishes that the model is input-dependent rather than a constant predictor
> (p = 0.0024), while its reproducible signal does not yet track true nWBV
> (r = −0.119, p = 0.590).

> Physics-constrained simulation shows no statistically significant advantage over
> Gaussian-blur degradation on the downstream task (MAE 0.0146 vs 0.0148, unpaired
> Mann-Whitney p = 0.81) and is not superior on image fidelity; all four
> reviewer-named self-supervised objectives were evaluated, three outperform the
> denoising objective on high-field data, and the strongest (DINO) does not transfer
> its advantage to 64 mT hardware. SynthSeg+ pseudo-labels are a viable deployment
> substitute for FastSurfer ground truth (MAE penalty +0.005).

**WHY:** R1.1, I1, I2, R2.3, R3.2. The remaining conclusion sentences (bounded to
healthy adults; contribution statement) stand.

### M23 | Tier 2 | §III-A — parameter sensitivity | SHOULD

**Insert after Table I (tissue relaxation parameters):**

> A sensitivity sweep perturbing each simulation parameter by ±20 % (effective SNR, B0
> inhomogeneity amplitude, T1/T2 relaxation values) changes the downstream OASIS-1
> test MAE by at most |ΔMAE| = 0.0062, with the maximum arising from a 20 % reduction
> in relaxation values; SNR and B0 perturbations change MAE by 0.0005 or less
> (Fig. 2 in the revised numbering). The pipeline is therefore not finely tuned to a
> single parameter choice, and the one influential parameter group — tissue relaxation
> — is the one sourced from published low-field relaxometry [24] rather than fitted.

**WHY:** R1.10. Source: `simulation_sensitivity/results.json`.

### M24 | Tier 0 | Whole document — redundancy consolidation | MUST (run LAST)

Rule (from `WORD_AGENT_PROMPT_R1.11.md`, adopted): a quantitative result appears at
most three times — once with full statistics in the Results subsection where it is
derived, once in the Abstract, once in the Conclusion if headline. Introduction and
Discussion restatements become cross-references or digit-free interpretation. Target
≈ 9,800–10,000 words. Never cut a unique caveat: every CDR = 1.0 mention keeps
"simulated, n = 2, directional"; every ICC mention keeps its wide-CI caveat where
present; every MC Dropout mention keeps the not-for-clinical-use warning. **WHY:**
R1.11 — and the response letter's claim that consolidation happened must be true.

### M25 | Tier 2 | §III-B ViT3D Architecture — selection rationale | MUST

**Insert after the parameter-count sentence ("Total parameters: 4.23M…"):**

> Architecture selection rationale. ViT3D was selected a priori for end-to-end
> tractability, not accuracy: the complete Stage-3 protocol (23-fold cross-session
> LOOCV with per-fold adaptation, five-seed repetition, permutation testing, conformal
> calibration and failure analysis) runs in approximately 20 h on the study's CPU-only
> hardware for a 4.23 M-parameter model, against an estimated ~13 days per arm for a
> 62 M-parameter hierarchical transformer. Larger architectures are evaluated in the
> OASIS-1 context in Section V; they were not carried through Stage 3, and this
> evaluation-depth asymmetry is stated as a limitation.

**WHY:** R2.2 — the justification R2 asked for, framed as a constraint rather than a
virtue. The "~13 days" figure is an estimate and must keep "estimated".

### M26 | Tier 1 | §III-C Rationale for LayerNorm+head adaptation (p. 4) | MUST

**BEFORE (final sentence of the rationale paragraph):**

> "A formal empirical comparison of head-only, LayerNorm+head, and full fine-tune
> strategies is identified as future work; the present choice is motivated by the
> structure of the observed domain shift rather than an exhaustive search."

**AFTER:**

> A formal empirical comparison is reported in Section V (Adapter Strategy Ablation):
> head-only, LayerNorm+head, LoRA and full fine-tuning are statistically
> indistinguishable at n = 23, and LayerNorm+head is retained for parameter economy
> and training stability rather than demonstrated optimality.

**WHY:** R2.4 — the submitted sentence deferring the comparison to future work is
exactly what R2 quoted; it must be replaced by the completed result, and the preceding
mechanistic rationale sentences ("full fine-tuning would risk catastrophic
forgetting…") must be softened by appending: "these expectations were not borne out as
performance differences (Section V)."

### M27 | Tier 1 | Title, Abstract, Discussion, Conclusion — framing audit | MUST (run after all other edits)

The title is unchanged ("A Reproducible Feasibility Baseline for …" — already what
R3.2 prescribes; do not retitle). Audit all four locations after every insert lands:
zero occurrences of "clinical readiness", "deployable", "clinically deployable",
"ready for deployment" applied to this system; 47 ms always bounded as computational
feasibility on a standard GPU (subject to gate G2) with edge devices not benchmarked;
every newly inserted result carries its limitation in the same paragraph. **WHY:**
R3.2, R3.7.

### M28 | Tier 2 | §VI — scope subsection for Reviewer 4 | MUST

**Insert as a short subsection in the Discussion (may be merged with M18(b) as its
final paragraph if space demands):**

> Scope of comparative evaluation. This model outputs a single scalar (nWBV) per scan;
> it performs no image synthesis or super-resolution, so a qualitative image-to-image
> comparison against reconstruction or enhancement methods has no output image to
> compare. The image-level evaluation appropriate to this design is reported for the
> one image-producing component, the physics simulator: a qualitative comparison of
> real and simulated slices (Fig. 1) and a quantitative fidelity analysis against an
> established degradation baseline (Section III). The state-of-the-art comparison
> appropriate to the task is quantitative: SynthSeg+ as the segmentation-based
> accuracy ceiling (r = 0.918, MAE = 0.005), CNN3D, Swin-UNETR and UNETR, with
> significance testing (Section V). The study is likewise intentionally
> single-modality: the simulator, the domain-adaptation protocol and the ground-truth
> pipeline are specific to 64 mT brain MRI, and evaluation across other medical
> imaging modalities would constitute a different research question; it is identified
> as future work.

**WHY:** R4.4, R4.5, R4.6 — the scope argument must exist in the manuscript body
(`MANUSCRIPT_AUDIT.md` D9), not only in the letter.

---

## PART B — Corrections to revision-added (WIP) content (W-series)

These target text that exists only in the working document (not in the submitted PDF),
so no submitted-PDF BEFORE exists; the FIND strings are from the WIP/17-page export.

### W1 | Tier 0 | Remove the frozen-encoder transfer probe everywhere | MUST

**FIND (four known locations — Abstract; §V comparators; fig05 caption; Discussion):**
any sentence containing "0.0112" or "0.0117" or "frozen-encoder transfer probe" or
"frozen encoder + ridge probe", including the Abstract clause "…but a frozen-encoder
transfer probe shows that advantage does not carry to 64mT (ViT MAE = 0.0112 vs Swin
0.0117); ViT3D is retained as the system studied through the complete protocol."

**ACTION:** Delete the probe claim in all locations. Where the Abstract clause is
deleted, the DINO transfer sentence from M1 already carries the does-not-transfer
finding. In §V, if a transitional sentence is needed, use:

> Whether the high-field advantage of larger architectures transfers to 64 mT was not
> tested at matched footing and remains future work; the transfer question is
> addressed for pretraining objectives via the DINO experiment below.

fig05 panel B is replaced per `07_FIGURE_PLAN.md`.

**WHY:** The probe compared ViT3D's 256-dim encoder features against Swin-UNETR's
8-dim decoder output channels — a 32× representation handicap; documented as unsound
in `RESPONSE_LETTER_ALL_REVIEWERS.md` Blocker 1. No fair re-run exists. Publishing it
would hand Reviewer 2 a methodological kill in a one-shot resubmission.

### W2 | Tier 0 | SimMIM-only SSL paragraph → four-objective outcome | MUST

**FIND:** the WIP passage reporting SimMIM alone (in substance: "Replacing the Stage-1
denoising objective with SimMIM … MAE 0.0538, r = 0.907 … SimMIM is stronger"), and
the WIP Discussion sentence "SimMIM pretraining outperforms our denoising objective".

**ACTION:** Superseded by M13 — all four objectives, three of which beat denoising,
plus the DINO transfer. The WIP's "the operative contribution is the physics-simulation
recipe, not the specific reconstruction objective" sentence must also be deleted or
rewritten — with three of four objectives beating denoising *and* the recipe itself not
beating blur, the supportable statement is only:

> Reconstruction-based fine-tuning on physics-simulated inputs yields usable OASIS-1
> performance under several objectives; no claim of objective-level or recipe-level
> superiority is made.

### W3 | Tier 0 | Mechanical WIP cleanups (adopted from the audit pass) | MUST

Apply as specified in `WORD_AGENT_PROMPT_AUDIT_TEXT.md` (verified correct against the
audit): all 10 in-text "Table N" citations → Roman ("Table II" etc.); cite Figure 17
once in §V (pseudo-label ablation); unit spacing per the IEEE checklist (64 mT, 47 ms,
0.020 nWBV — title excepted; 3T/1.5T left as authorial choice); repair the page-number
footer; remove the Table III "Notcom" artifact (also covered by M8); Table IX stray
weight-decay row; reference [42] (van den Broek) placed in citation order and
un-highlighted once inserted (gate G3).

### W4 | Tier 0 | Latency double-reporting | MUST

**FIND:** WIP §III-B latency figures ("6.9 ms" / "2034 ms") coexisting with the 47 ms
claims (MANUSCRIPT_AUDIT D8).

**ACTION — REVISED 2026-08-23 (G2 failed; see I10/M29).** There is no longer a 47 ms
GPU figure to reconcile against: it is withdrawn as unmeasured. Every latency number
in the paper must now trace to `experiments/inference_latency/results.json` and carry
its hardware, thread count and repetition count. Delete any number that cannot.

Note the WIP's "6.9 ms" is consistent with the independent CPU re-measurement (~5 ms,
4 threads), which is further evidence that the 47 ms GPU figure was never a
measurement of this model. Whichever value the released benchmark yields is the one
that ships — do not average, round, or reuse a previous number.

### W5 | Tier 1 | WIP Abstract comparator clause consistency | MUST

After M1/W1/M13 land, re-read the Abstract once: it must contain exactly one
comparator sentence (the M1 version), no transfer-probe clause, no "SimMIM r = 0.907"
orphan, and the Swin/UNETR values as r = 0.964 / r = 0.923 (point estimates; not the
stale "MAE 0.019, r = 0.939" pre-retrain values, which must not reappear).

---

## Consistency with the integrity report

Every item in `08_FIGURE_DATA_INTEGRITY_REPORT.md` Part 4 is covered: r/p correction
(M5, M6 + Fig 15 caption), deletion of the well-calibrated-direction sentence (M5),
the Figure 20 decision (resolved — real OASIS MC Dropout run 19 Aug; M16 + fig plan),
full Figure 22 rebuild (gate G4), figure-source audit and the `np.random` strip
(07_FIGURE_PLAN §protections).

## Edit priority index

MUST: M1–M22, M24–M28, W1–W5 · SHOULD: M23 · OPTIONAL: none (anything not listed is
deliberately unchanged).

---

# PART D — Edits added 2026-08-23 (post-Branch-B, post-latency-audit)

These supersede parts of the M-series written on 2026-08-19. Apply them together
with the REVISED notes on M1, M19 and M20.

### M29 | Tier 0 | Abstract, §V-G, §VI-A, §VI-D, fig10 | MUST

**Integrity item I10.** The manuscript states inference latency as **"47 ms on a
standard GPU"** in four places (Abstract; §V-G; §VI-A Deployment; §VI-D). No
measurement file for this figure exists anywhere in the repository — it appears only
as a hardcoded literal in `generate_all_figures.py` (line 333), `generate_new_figures.py`,
`regen_updated_figures.py` and `fix_figures_r2.py`, and the same `0.047` constant is
assigned to two different model configurations in the fig10 latency array.

It also contradicts the repository's own record: `CLAUDE.md` states that training and
evaluation ran on **CPU** because "the development machine's GPU was CUDA-incompatible".
A GPU benchmark cannot have been produced on that hardware.

**BEFORE (four locations, same claim):**

> "...with 47 ms inference." (Abstract)
> "Inference completes in 47 ms on a standard GPU without FreeSurfer..." (§V-G / §VI-A)
> "...compute-light inference (47 ms on a standard GPU); edge-device latency is not yet
> measured..." (§VI-D)

**AFTER (fill `<T>`, `<IQR>`, `<N>`, `<CPU>` from `experiments/inference_latency/results.json`):**

> Abstract: "...with a measured inference time of `<T>` ms per volume on CPU."
>
> §V-G / §VI-A: "A single forward pass over one 64×64×64 volume completes in `<T>` ms
> (IQR `<IQR>`, `<N>` repetitions, `<CPU>`, `<threads>` threads), measured with the
> benchmark released as `experiments/inference_latency/`. No GPU is required, and no
> FreeSurfer or SynthSeg segmentation is invoked. The matched-protocol CNN3D
> regressor runs in 11.9 ms on the same machine, so the speed advantage reported here
> follows from avoiding segmentation rather than from the transformer architecture:
> the segmentation-based upper bound (SynthSeg+) requires on the order of 150 s per
> volume [cite], roughly four orders of magnitude more than either direct regressor.
> Edge-device latency on ULF scanner hardware has not been benchmarked."

**WHY:** I10; R3.7 (deployment claims bounded, which R3 asked for explicitly);
R4.6. **Do NOT substitute an estimate** — if the benchmark cannot be produced, delete
the latency claim entirely rather than soften it to "<50 ms", which would remain
unmeasured. CPU-only operation is the stronger point-of-care claim in any case:
bedside ULF scanners are unlikely to carry a GPU.

**Note — the corrected measurement changes the claim.** The fabricated array made
CNN3D look slow (95 s) and the ViT uniquely fast. Measured, CNN3D is 11.9 ms against
ViT3D's 4.53 ms — a 2.6x ratio that simply tracks the 1.94x parameter ratio. The
four-orders-of-magnitude gap is against **segmentation**, not against the CNN. Any
sentence implying the ViT architecture is what makes the method fast must be rewritten
to credit the segmentation-free formulation. This is the more accurate claim and it is
also the one consistent with the rest of the revision, which removes every
architecture-superiority claim (M3, M18, M27).

**Also:** remove "on a standard GPU" everywhere; rebuild fig10 with the measured
value; add `47 ms` and `standard GPU` to the zero-count grep gate in `08` Risk 5.

### M30 | Tier 2 | NEW §V-J "External validation" (after §V-I) | MUST

**BEFORE:** no such section exists.

**AFTER** (age paragraph resolved 2026-08-23 — ds006557 mean 45.1 yr [21–69, SD 14.9]
from `participants.tsv`; Zenodo mean 30 yr [19–65] from the data descriptor PDF,
cohort-level only. **Age explains 17–61 % of the offset, not all of it** — do not
write the simple biological explanation; the arithmetic is two numbers and a reviewer
will do it):

> **J. External validation on an independent 64 mT cohort**
>
> To test generalisation beyond ds006557 we evaluated the model on the second public
> paired 64 mT/3T cohort (van den Broek et al. [42]): 11 paired subjects, of which 10
> yielded complete 3T segmentation. Ground truth was derived with the same FastSurfer
> `--seg_only` pipeline and the same nWBV definition (BrainSeg/Mask) used for OASIS-1
> and ds006557, so the external labels are directly comparable to those used
> throughout this study. No subject from this cohort was seen in any training,
> pre-training or adaptation stage.
>
> The result does not support generalisation. The 64 mT-adapted model attains
> MAE = 0.0731 (Pearson r = 0.255, p = 0.477); the OASIS-1 model without 64 mT
> adaptation attains MAE = 0.0327 (r = 0.264, p = 0.461). Both are far above the
> cohort's own constant-mean baseline (MAE = 0.0089), and the adapted model is more
> than twice as far from the truth as the unadapted one — the cross-session adapter,
> which improves within-ds006557 performance, transfers negatively.
>
> The failure has a specific and measurable cause. The two cohorts' nWBV
> distributions do not overlap: ds006557 spans [0.752, 0.808] (mean 0.784) while the
> external cohort spans [0.836, 0.869] (mean 0.856). The adapted model's predictions
> occupy [0.7787, 0.7870] — a spread of 0.008 against a true spread of 0.033 — and
> centre on 0.7824, within 0.002 of the adaptation cohort's mean. Every prediction
> falls below every ground-truth value. The model therefore returns its training-set
> mean and does not extrapolate beyond the label range it was adapted on, which is
> the same range-compressed behaviour identified internally in Section V-E
> (r = −0.119, p = 0.590), now confirmed on data from an independent site and scanner.
> The external cohort is younger (mean 30 years, range 19–65; cohort-level
> demographics only, no per-subject ages are distributed) than the adaptation cohort
> (mean 45.1, range 21–69), and part of the nWBV offset is attributable to age.
> Applying the age–nWBV slope observed within ds006557 accounts for approximately
> 0.012 of the 0.071 difference; the steeper slope implied by the OASIS-1 age span
> accounts for approximately 0.043. A residual offset therefore remains that age alone
> does not explain, and which may reflect cohort or segmentation-pipeline differences
> between the two datasets; we cannot resolve it with the data available. This does not
> affect the finding: the adapted model's predictions vary by 0.008 against a true
> spread of 0.033 and centre on the adaptation cohort's mean, so it does not track
> anatomy in this cohort irrespective of the cause of the distribution shift.
>
> We report this as the primary external finding rather than a caveat. It bounds the
> claim of this paper precisely: the protocol and the adaptation procedure are
> reproducible and demonstrably input-dependent, but the resulting estimator is not
> transportable to cohorts whose nWBV distribution differs from the adaptation set,
> and must not be applied as a general-purpose nWBV predictor.

**WHY:** R1.3 (the concern is now answered with an experiment rather than conceded),
R3.5, R3.6, R1.2. Insert **fig24** here. Source:
`experiments/zenodo_external_validation/results.json`, commit e389ba6.

**Cross-effects:** M1 (abstract) gains one external-validation sentence; §VI-E
Limitations gains the transportability bound; the contribution list (M3) must not
claim generalisation.

### M31 | Tier 0 | Global | MUST

Subject-count discipline for the external cohort. The dataset contains **11 paired**
subjects (Zenodo record 10.5281/zenodo.15471394); **10** were analysable. Every
mention must distinguish the two — write "11 paired subjects, 10 with complete 3T
segmentation" on first use and "n = 10" for all results. `sub-0064` did not complete
segmentation.

**WHY:** the count appears inside a disclosed correction; a wrong number there costs
more than anywhere else in the paper.


### M5b | Tier 0 | §V-I, sentence immediately after M5 | MUST

Consequence of M5. The surviving sentence opens "Both observations", whose original
antecedents were (a) the 4.3 % coverage and (b) *the positive r value*. M5 deletes the
second. One word restores the reference.

**BEFORE:** > "Both observations are consistent with the known failure mode of dropout
based uncertainty under covariate shift [40]:"

**AFTER:** > These observations are consistent with the known failure mode of dropout
based uncertainty under covariate shift [40]:

Change nothing else in that sentence — the covariate-shift explanation that follows is
correct and stands.

**WHY:** I3 follow-on. Flagged by the Word pass on 2026-08-24.

### M32 | Tier 0 | Whole document — figure renumbering | MUST (Phase 8, after all text edits)

The submitted manuscript has **17** figures; the revision has **24**. Several edits
(M5, M6, M11, M16, and others) cite the **revised** numbering, so until this edit runs
the body and the captions disagree. That is expected and must not be "fixed" early by
renumbering piecemeal.

Run once, after every text edit, in this order:

1. Insert all new figures at their anchor points, using the placeholder convention.
2. Renumber captions 1–24 to match `07_FIGURE_PLAN.md` order.
3. Sweep every in-text callout ("Fig. N", "Figure N", "Figs. N and M") against the new
   numbering. Known remappings from the submitted set: **14 → 20**, **15 → 21**,
   **17 → 23**.
4. Verify: every figure is cited at least once in the body, and every callout resolves
   to an existing figure. Zero orphans in either direction.

**WHY:** R4.3 and basic correctness. A reviewer checking figure callouts is doing the
cheapest possible verification, and a mismatch there undermines the rest.

### M33 | Tier 1 | §V-E Cross-Session LOOCV Results (p. ~6, para 210) | MUST

**COVERAGE GAP — added 2026-08-24.** Found by the Word pass's forbidden-value audit,
not by the original plan. This is the third `0.0126` and the only one no edit reached;
M1 (abstract), M3 (contributions) and M22 (conclusion) cover the others. Without this
edit the wrong baseline ships in the Results section — the single place a reviewer
checking R1.1 would look first.

The passage also carries a claim that must be corrected on its own merits: it asserts
that inter-session ICC "cannot be produced by a constant baseline". That is the right
intuition but it was asserted, not demonstrated. We now demonstrate it with the
permutation test, and the passage should point there.

**BEFORE:**

> "constant-mean LOOCV baseline (predict leave-one out cohort mean for each heldout
> subject) yields MAE = 0.0126. The ViT3D LOOCV MAE (0.0134) is therefore not
> meaningfully smaller than the constant-mean MAE in absolute terms, which is an
> expected consequence of the 0.056 nWBV range: in such a regime, the MAE metric is
> dominated by the cohort spread rather than by the model. The result that
> distinguishes the model from a constant predictor is the inter-session ICC reported
> below, which requires inputconditional behaviour and cannot be produced by a
> constant baseline."

**AFTER:**

> constant-mean LOOCV baseline (predict the leave-one-out cohort mean for each
> held-out subject) yields MAE = 0.0128; predicting the global cohort mean yields
> 0.0122. The ViT3D LOOCV MAE of 0.0134 is therefore not smaller than either — it is
> numerically above both. This is an expected consequence of the 0.056 nWBV range: in
> such a regime MAE is dominated by the cohort spread rather than by the model, and we
> do not claim an accuracy advantage over a constant predictor. What distinguishes the
> model from a constant predictor is not accuracy but input-dependence, established by
> the permutation test in Section V-F: a constant predictor cannot produce agreement
> between two independent acquisitions of the same subject, and the observed
> inter-session ICC(3,1) = 0.615 exceeds its permutation null (mean ≈ 0) at
> p = 0.0024.

**WHY:** Integrity I1 (0.0126 → 0.0128, plus the global-mean figure), R1.1, R3.4. The
substantive change is that the previous text asserted a constant baseline "cannot"
produce the ICC; the revision demonstrates it instead of asserting it, and concedes
the accuracy comparison outright rather than describing it as "not meaningfully
smaller" — which read as a hedge and is what Reviewer 1 objected to.

**Depends on M10** (which creates the §V-F permutation subsection). Apply M33 after
M10 so the cross-reference resolves, or apply now and confirm the section letter when
M10 lands.

### M34 | Tier 0 | §VI — the conformal "future work" sentence | MUST

**COVERAGE GAP — added 2026-08-24**, found by the Word pass's audit of the phrase
"identified as future work". Same class as M33: an integrity-I9 item that no edit
reached. M20 clears the Future Works *list*; this is an inline sentence elsewhere.

The submitted text says split-conformal calibration "is the appropriate remedy and is
identified as future work". M11 now reports it as a completed result (91.3 % coverage
at nominal 90 %, 95.7 % at nominal 95 %). Left unedited, the paper defers as future
work something it reports two sections earlier — exactly the self-contradiction
Reviewer 1 would seize on.

**BEFORE:** > "Split conformal calibration on prospective acquisitions is the
appropriate remedy and is identified as future work."

**AFTER:** > Split conformal calibration is the appropriate remedy and is applied in
this revision (Section V, Uncertainty Quantification): it restores near-nominal
coverage on the present cohort, at interval widths that exceed the cohort's nWBV
range. Extension to prospective multi-site acquisitions remains future work.

**WHY:** Integrity I9, R1.7. Sources: `conformal_calibration/results.json`.

**Audit note — the other three "identified as future work" instances:**
- §III-C adapter comparison — already fixed by M26.
- Latency / edge devices — inside M29's scope; leave for M29.
- CNN3D adapter bias ("whether the same adapter would correct CNN3D's −0.076 bias")
  — **still true, leave it.** CNN3D was never carried through adaptation.

### M35 | Tier 0 | §III-A — Arabic section cross-reference | MUST

The document references "Section 6.5" in Arabic form where every other cross-reference
uses roman numerals. M11 removed the instance in the Uncertainty Quantification
subsection; one survives in Section III-A.

**BEFORE:** > "This SNR mismatch is the primary source of the preadaptation bias
(discussed in Section 6.5)."

**AFTER:** > This SNR mismatch is the primary source of the pre-adaptation bias
(discussed in Section VI-<X>).

Substitute `<X>` with the letter of the Section VI subsection that actually discusses
the pre-adaptation bias — confirm it in the document rather than assuming, since
Section VI's lettering was not audited during the Section V insertions. Also fixes
"preadaptation" → "pre-adaptation".

**WHY:** Consistency; a dangling Arabic reference in an otherwise roman-numbered paper
reads as an unfinished draft.
