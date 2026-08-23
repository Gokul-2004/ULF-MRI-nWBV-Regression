# 11 — Master prompt for the Claude extension in Word

**Paste the block below ONCE at the start of the Word session.** It sets the rules.
Then work phase by phase: paste one edit at a time from `05_MANUSCRIPT_CHANGES.md`,
following the order in `10_APPLY_ORDER_WORKSHEET.md`.

Do not paste all 36 edits at once — the extension will lose fidelity on the exact
quotes, and exactness is the whole point.

---

## THE MASTER PROMPT — paste this first

```
You are helping me revise a rejected IEEE Access manuscript for its ONE permitted
resubmission. Manuscript ID Access-2026-28453. If this resubmission fails, the paper
is dead. The same four reviewers will see it and will check the paper against a
point-by-point response letter.

The document currently open is the UNEDITED original submission. I will give you
edits one at a time, each with an ID (M1..M31, W1..W5), an exact BEFORE quote, and
the exact AFTER text.

ABSOLUTE RULES — these override any instinct to improve on what I give you:

1. Use my AFTER text VERBATIM. Do not rewrite it, do not "improve" the phrasing, do
   not adjust the tone, do not shorten it. It has been checked against source data
   and every number traces to a file. If you reword it you may break a number.

2. NEVER invent, infer, round, or "correct" a number. If an edit needs a value I
   have not supplied, stop and ask. Several numbers in this paper were previously
   fabricated and that is the single thing this revision exists to fix.

3. If the BEFORE quote does not match the document exactly, STOP and tell me what
   the document actually says. Do not apply the edit to approximately-matching text
   and do not guess which passage I meant.

4. HIGHLIGHT every change in yellow. A highlighted PDF is a required upload; an
   unhighlighted change is an invisible change.

5. Do not touch anything I have not asked you to change. No silent copy-editing, no
   reformatting, no reference renumbering unless I ask.

THESE VALUES ARE FORBIDDEN. They are wrong or fabricated. If you ever find yourself
writing one, or if you see one surviving in the document after we finish a section,
flag it to me immediately:

    0.0126        (wrong constant-mean baseline; correct value is 0.0128)
    47 ms / 47ms  (unmeasured GPU latency claim; withdrawn)
    standard GPU  (no GPU was ever used in this work)
    4.1M / 4.1 M  (wrong CNN3D parameter count; correct is 8.22M)
    p = 0.847 / p=0.847   (invalid paired Wilcoxon; the valid test gives p = 0.81)
       ^ ONLY as a p-value. Bare "0.847" also appears as the upper bound of a
         Fisher confidence interval, r = 0.722 [0.524-0.847]. That one is REAL
         DATA and must never be changed.
    +0.34         (FABRICATED from placeholder random data; correct is -0.164)
    1.368         (mislabelled effect size; correct pooled-SD Cohen's d is 1.463)
    only public   (false claim of sole dataset existence)
    0.0112 / 0.0117  (from a methodologically unsound experiment being removed)

THESE VALUES MUST APPEAR in the finished document. If any is missing at the end,
an edit was skipped:

    0.0128      constant-mean baseline (at least 4 places)
    -0.164      CI-width vs error correlation
    23.7        OASIS MC Dropout coverage percent
    8.22        CNN3D parameter count in millions
    4.53        measured ViT3D inference time in ms
    0.0731      external validation MAE
    van den Broek   the external cohort citation

FIGURES: when an edit calls for a figure, insert a placeholder line in this exact
form, on its own line, highlighted, and put the caption immediately beneath it:

    [[INSERT FIGURE <n>: <filename>]]

I will replace the placeholders with the actual images myself. Never generate,
sketch, or describe a figure — only the placeholder and the caption.

Confirm you understand these rules, then wait for my first edit.
```

---

## Figure placeholder map

Files live in `Review - 2/figures_to_insert/`. **Insert the `.svg` for Figures 2–24;
insert the `.png` for Figure 1 only** (it is MRI slice imagery — its SVG is just
wrapped bitmaps and bloats the document).

| # | Placeholder line to use |
|---|---|
| 1 | `[[INSERT FIGURE 1: fig01_axial_slices_3T_sim_real.png]]` ← PNG |
| 2 | `[[INSERT FIGURE 2: fig02_simulation_parameter_sensitivity.svg]]` |
| 3 | `[[INSERT FIGURE 3: fig03_three_stage_training_pipeline.svg]]` |
| 4 | `[[INSERT FIGURE 4: fig04_predicted_vs_true_oasis.svg]]` |
| 5 | `[[INSERT FIGURE 5: fig05_architecture_pretraining_comparators.svg]]` |
| 6 | `[[INSERT FIGURE 6: fig06_true_nwbv_by_cdr_stage.svg]]` |
| 7 | `[[INSERT FIGURE 7: fig07_physics_vs_gaussian_blur.svg]]` |
| 8 | `[[INSERT FIGURE 8: fig08_multiseed_loocv_stability.svg]]` |
| 9 | `[[INSERT FIGURE 9: fig09_method_comparison_real64mt.svg]]` |
| 10 | `[[INSERT FIGURE 10: fig10_fourway_comparison_with_latency.svg]]` |
| 11 | `[[INSERT FIGURE 11: fig11_age_vs_nwbv_truth_and_predictions.svg]]` |
| 12 | `[[INSERT FIGURE 12: fig12_cross_session_loocv_scatter.svg]]` |
| 13 | `[[INSERT FIGURE 13: fig13_bootstrap_mae_and_bland_altman.svg]]` |
| 14 | `[[INSERT FIGURE 14: fig14_permutation_tests_input_dependence.svg]]` |
| 15 | `[[INSERT FIGURE 15: fig15_intersession_reliability_icc.svg]]` |
| 16 | `[[INSERT FIGURE 16: fig16_cdr_stratified_mae_simulated.svg]]` |
| 17 | `[[INSERT FIGURE 17: fig17_pseudolabel_ablation.svg]]` |
| 18 | `[[INSERT FIGURE 18: fig18_adapter_strategy_ablation.svg]]` |
| 19 | `[[INSERT FIGURE 19: fig19_split_conformal_calibration.svg]]` |
| 20 | `[[INSERT FIGURE 20: fig20_mcdropout_intervals_oasis.svg]]` |
| 21 | `[[INSERT FIGURE 21: fig21_mcdropout_intervals_real64mt.svg]]` |
| 22 | `[[INSERT FIGURE 22: fig22_failure_analysis.svg]]` |
| 23 | `[[INSERT FIGURE 23: fig23_mcdropout_calibration_curve.svg]]` |
| 24 | `[[INSERT FIGURE 24: fig24_external_validation_zenodo.svg]]` |

**NEVER insert these two:**

- `fig22D_ci_width_vs_error_CORRECTED.*` — this is the source for fig22's panel D,
  already merged into fig22. It is not a standalone figure.
- `_superseded/fig20_mcdropout_intervals_oasis.*` — the synthetic version, quarantined.
  The fig20 in the main folder is the real-data rebuild.

## Captions that change (do not reuse the originals)

| Fig | Required caption content |
|---|---|
| 5 | Eight comparators, true parameter counts, adopted config ranks 7th of 8; panel B is denoising vs DINO under identical 64 mT LOOCV |
| 7 | Paired test invalid (separate splits); valid comparison is unpaired Mann-Whitney **p = 0.81** |
| 10 | Two learned models **measured** on CPU; SynthSeg+ is a **published** runtime, hatched. Speed advantage is from being segmentation-free, not from the transformer |
| 14 | BOTH halves: input-dependent **p = 0.0024**, AND predictions do not track true nWBV **r = -0.119, p = 0.590** |
| 18 | All four CIs overlap; no strategy significantly superior; LN+head retained for parameter economy |
| 20 | Coverage **23.7 %** (9/38), mean width 0.0284 — NOT 4.3 % |
| 21 | Label the model **unadapted**; r = **-0.164**, p = 0.454 |
| 22 | Keep the word **LOOCV** in panel C's description. Panel D: r = **-0.164**, p = 0.454 |
| 24 | n = 10 analysable of 11 paired; adapted 0.0731 vs unadapted 0.0327 vs cohort mean 0.0089; ranges do not overlap |

## Working method

1. Paste the master prompt. Wait for confirmation.
2. Work the phases in `10_APPLY_ORDER_WORKSHEET.md` **in order** — the ordering is
   load-bearing (W1/M31 are global and run first; M24/M27 run last or they undo
   content edits; new §V subsections must insert in sequence or the letters break).
3. For each edit, paste: the ID, the BEFORE quote, the AFTER text, verbatim from `05`.
4. After each phase, ask: *"List any forbidden values still present in the document."*
5. At the very end, run the full Phase 9 gate.
