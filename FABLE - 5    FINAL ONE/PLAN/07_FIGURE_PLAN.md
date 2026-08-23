# 07 — FIGURE PLAN

Baseline: the 23-figure set in `07_FINAL_FIGURES/` (600 dpi PNG + true-vector SVG per
figure), built against the revised manuscript's numbering. Decision: **keep all 23 in
the main text, in the existing order** — no reviewer objected to figure count (R1.11
concerns text repetition; R4.3 concerns sharpness), and renumbering 23 placed figures
in Word is the highest-risk mechanical operation in the whole resubmission. Two
figures are REBUILT before insertion (fig05, fig22); one caption number changes
(fig20).

## Format rule (per `MANIFEST.md`, adopted)

- **Figure 1: insert the PNG** (MRI slice imagery; its SVG is just wrapped bitmaps).
  Disable Word image compression (per-picture "Compress Pictures… → highest quality →
  all pictures" on Mac builds without the global option).
- **Figures 2–23: insert the SVG** (vector; immune to Word downsampling). After the
  first SVG insert (use fig02), check text rendering; if fonts substitute badly, fall
  back to the 600 dpi PNGs throughout — both satisfy R4.3.
- Verify by exporting to PDF and zooming to 400 % on Figure 1's slices before
  declaring R4.3 done. The response letter's claims must be true of the uploaded file.

## Figure-by-figure disposition

| # | File (basename) | Status / action | Caption action |
|---|---|---|---|
| 1 | `fig01_axial_slices_3T_sim_real` | Insert PNG. | Keep. Serves R4.4 (qualitative image comparison of the simulator). |
| 2 | `fig02_simulation_parameter_sensitivity` | Insert (was a `[FIGURE PLACEHOLDER` marker). | Caption states max \|ΔMAE\| = 0.0062 and the ±20 % design (M23). |
| 3 | `fig03_three_stage_training_pipeline` | Insert (★ regenerated — layout fixed). | Keep. |
| 4 | `fig04_predicted_vs_true_oasis` | Insert. | Keep (n = 38, ViT3D; atrophy failure mode note). |
| 5 | `fig05_architecture_pretraining_comparators` | **REBUILD REQUIRED before insertion** — see below. Placeholder in WIP. | New caption below. |
| 6 | `fig06_true_nwbv_by_cdr_stage` | Insert (★ regenerated — labels/axis fixed). | Keep. |
| 7 | `fig07_physics_vs_gaussian_blur` | Insert. | Caption must say "separate training conditions; descriptive r values — the valid comparison is the unpaired Mann-Whitney, p = 0.81 (Table VIII)" per M8. |
| 8 | `fig08_multiseed_loocv_stability` | Insert (placeholder in WIP). | Caption: five seeds; MAE 0.0130 ± 0.0004; ICC 0.644 ± 0.058; seed-7 ICC 0.563 visible (M12). |
| 9 | `fig09_method_comparison_real64mt` | Insert. | Keep (SynthSeg+ / CNN3D / ViT3D, no adaptation). If the R1.11 pass needs a cut, this is the one merge candidate (its content overlaps fig10) — OPTIONAL, only if word/page pressure demands, and only by merging into fig10, never by silent deletion. |
| 10 | `fig10_fourway_comparison_with_latency` | Insert. | Latency labels subject to gate G2 (47 ms provenance). |
| 11 | `fig11_age_vs_nwbv_truth_and_predictions` | Insert (⚠ 600 dpi rebuild used). | Caption cites ρ = −0.778 / ρ = +0.232 — frozen until gate G1 (age re-verification) clears. |
| 12 | `fig12_cross_session_loocv_scatter` | Insert. | Keep. |
| 13 | `fig13_bootstrap_mae_and_bland_altman` | Insert. | Keep (bias −0.0022). |
| 14 | `fig14_permutation_tests_input_dependence` | Insert (placeholder in WIP). | Caption must carry BOTH halves: "Observed ICC 0.6146 vs null ≈ 0, p = 0.0024 (input-dependent). Predictions do not track true nWBV (r = −0.119, p = 0.590)." (M10). |
| 15 | `fig15_intersession_reliability_icc` | Insert. | Keep (ICC 0.615 [0.236–0.866]). |
| 16 | `fig16_cdr_stratified_mae_simulated` | Insert. | Keep; must retain "simulated, no adaptation; CDR 1.0 n = 2 directional" (M17). |
| 17 | `fig17_pseudolabel_ablation` | Insert. | Keep — AND add the missing in-text citation in §V (pseudo-label ablation) (W3). |
| 18 | `fig18_adapter_strategy_ablation` | Insert (placeholder in WIP). | Caption: "All four 95 % CIs overlap; no strategy significantly superior. LN+head retained for parameter economy." (M15). |
| 19 | `fig19_split_conformal_calibration` | Insert (placeholder in WIP). | Caption: conformal 91.3 % @ 90 % (w 0.0571), 95.7 % @ 95 % (w 0.0664) vs MC Dropout 4.3 % (w 0.0286); group-level tool (M11). |
| 20 | `fig20_mcdropout_intervals_oasis` | Insert the **19 Aug rebuild** (real MC Dropout, N = 100). The synthetic original is quarantined in `_superseded/` — never insert it. | **Caption number changes: 4.3 % → 23.7 %** (9/38), mean width 0.0284 (M16). |
| 21 | `fig21_mcdropout_intervals_real64mt` | Insert (★ rebuilt from real per-subject intervals). | Caption: r = −0.164, p = 0.454 (M6); label the model **unadapted** (MC MAE 0.0403), not "LOOCV (adapted)". |
| 22 | `fig22_failure_analysis` | **REBUILD REQUIRED on the dataset machine** (gate G4): panel C needs `participants.tsv`; assemble with the corrected panel D from `fig22D_ci_width_vs_error_CORRECTED.*` (r = −0.164, p = 0.454). Until rebuilt, do not insert the old four-panel version — its panel D is synthetic. | Panel-D caption text per M6. |
| 23 | `fig23_mcdropout_calibration_curve` | Insert. | Keep (4.3 % at nominal 95 %, real 64 mT — this 4.3 % is correct and stays). |

## fig05 rebuild specification (blocking — current file is wrong on two counts)

The existing `fig05` (A) shows only CNN3D / ViT3D / ViT3D+SimMIM / Swin-UNETR / UNETR
and labels CNN3D "4.1M"; (B) shows the frozen-encoder ridge probe (ViT 0.0112 vs Swin
0.0117) annotated "capacity gives no benefit here" — the probe is methodologically
unsound (8-dim vs 256-dim features) and is removed from the paper (W1).

Rebuild from `scripts/generate_review2_figures_part2.py` (extended), reading only
`experiments/*/results.json`:

- **Panel A — OASIS-1 test MAE (n = 38) by model/objective, with parameter counts:**
  CNN3D **8.2M** 0.0242 (r 0.877) · ViT3D (ours, denoising) 4.2M 0.0584 (0.722) ·
  ViT3D+DINO 0.0270 (0.903) · ViT3D+MAE 0.0319 (0.756) · ViT3D+SimMIM 0.0538 (0.907)
  · ViT3D+contrastive 0.0589 (0.744) · Swin-UNETR 62.2M 0.0148 (0.964) · UNETR 92.7M
  0.0198 (0.923). Sources: `oasis_bootstrap`, `ssl_comparator_*`,
  `arch_comparator_*`. No editorialising annotations ("larger models win" may stay as
  a neutral descriptor; nothing about capacity/benefit claims).
- **Panel B — replace the probe with the sound transfer test:** denoising vs DINO
  under the identical cross-session 64 mT LOOCV — MAE 0.0134 vs 0.0220, ICC 0.615 vs
  0.523, subjects below 0.020: 19/23 vs 10/23. Source: `dino_headline_loocv`,
  `loocv_cross_session`. Annotation limited to: "high-field advantage does not
  transfer (paired Wilcoxon p = 0.070, n.s.)".
- **New caption:** "Architecture and pretraining comparators. (A) OASIS-1 test MAE
  (n = 38) under the identical protocol; several comparators outperform the adopted
  configuration, reported plainly. (B) Pretraining-objective transfer to real 64 mT:
  the strongest high-field objective (DINO) degrades under the identical cross-session
  LOOCV (MAE 0.0220 vs 0.0134; 10/23 vs 19/23 subjects below the 0.020 threshold)."
- Export 600 dpi PNG + SVG, same basename, replacing both files; verify no
  `np.random` anywhere in the generating code path.

## Protections (so synthetic data cannot reach the manuscript again)

1. Never regenerate any manuscript figure with `scripts/generate_all_figures.py`
   until its three `np.random` interval blocks (lines ~433, ~474, ~532) are stripped;
   prefer `generate_review2_figures.py` / `_part2.py` (audited clean: zero
   `np.random`, zero hardcoded results).
2. `_superseded/` stays quarantined; nothing in it is ever inserted.
3. After placement, verify: `FIGURE PLACEHOLDER` count = 0 (six markers existed:
   figs 2, 5, 8, 14, 18, 19); all 23 figures cited in text at least once (fig17 was
   the orphan); every caption statistic matches `02_EVIDENCE_INVENTORY.md`.

## Ordering and numbering

Keep the 1–23 order as in `MANIFEST.md` (it follows the revised manuscript's reading
order: simulation → pipeline → OASIS → comparators → CDR → ablations → real-64 mT →
LOOCV → reliability → dementia → pseudo-labels → adapter → uncertainty → failure).
No figure moves to supplementary: IEEE Access has no length surcharge, every figure
answers a named reviewer request, and the risk of a broken cross-reference outweighs
the benefit of a shorter figure list. The only sanctioned reduction is the optional
fig09→fig10 merge noted above, and only if the R1.11 pass demands it.


---

## Revision 2026-08-23

**fig05 — REBUILT and inserted.** Built by `scripts/generate_review3_figures.py`
(600 dpi PNG + true-vector SVG, 0 embedded rasters, no `np.random`; every value read
from committed JSON and printed with its source). Panel A now carries all eight
comparators with true parameter counts — CNN3D at **8.22M**, not the false 4.1M — and
the adopted configuration ranks 7th of 8, shown plainly. Panel B replaces the removed
frozen-encoder probe (W1) with denoising vs DINO under the identical cross-session
64 mT LOOCV; the lines cross, which is the transfer argument in one panel.

**fig24 — NEW, inserted in the new §V-J.** External validation (R1.3). Panel A:
predicted vs FastSurfer nWBV for the adapted and unadapted models, with the
adaptation-cohort range shaded — predictions sit flat and entirely below identity.
Panel B: the four label/prediction ranges as bars, showing the adaptation range
[0.7523, 0.8084] and the external range [0.8360, 0.8694] do not overlap. Caption must
carry: n = 10 analysable of 11 paired; adapted MAE 0.0731 vs unadapted 0.0327 vs
cohort constant-mean 0.0089; predictions centre on 0.7824 against an adaptation mean
of 0.7841.

**fig10 — NOW REQUIRES REBUILD (was: insert as-is).** Its latency labels carry the
withdrawn "47 ms" claim. Gate G2 has failed — no measurement exists. Rebuild once
`experiments/inference_latency/results.json` is produced (M29). Until then fig10 must
not be inserted.

**fig11 — UNBLOCKED.** Gate G1 cleared: ρ = −0.7777 / +0.2318 re-derived on the
dataset machine and both match the manuscript. Caption may be finalised.

**fig22 — still blocked (G4)**, but the blocker is now only scheduling: the dataset
machine has `participants.tsv`, so panel C can be built. Assemble with the corrected
panel D from `fig22D_ci_width_vs_error_CORRECTED.*`.

**Count:** 24 figures. If fig24 should sit beside the LOOCV results rather than last,
renumber before Word placement, not after.

**fig22 — REBUILT 2026-08-23** (`scripts/rebuild_fig22_failure_analysis.py`, 600 dpi +
SVG, no `np.random`). Panel D now carries the real value: **r = −0.164, p = 0.454**,
trend line sloping down, labelled n.s. Panel C built from real ages in
`participants.tsv`. Gate G4 CLEARED.

⚠ **Caption trap — three different "age" statistics exist. Do not interchange them:**

| Quantity | Model | Value | Where |
|---|---|---|---|
| predictions vs age | LOOCV-adapted | ρ = +0.232, p = 0.287 | abstract, §V-E, discussion, fig11 |
| **error** vs age | LOOCV-**adapted** | ρ = −0.088, p = 0.690 (n.s.) | **fig22 panel C only** |
| **error** vs age | **unadapted** | r = +0.678, p = 0.0004 (sig.) | `paper_statistics/failure_analysis.json` — **not cited in the manuscript, and must not be added without the "unadapted" label** |

The second and third point in opposite directions and are both correct; they describe
different models. The fig22 caption must keep the word "LOOCV". Verified 2026-08-23:
`0.6776` appears nowhere in the manuscript, so there is currently no conflict.
