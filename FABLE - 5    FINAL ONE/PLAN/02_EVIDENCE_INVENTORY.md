# 02 — EVIDENCE INVENTORY: what this repository actually contains

Every number below was read directly from the named JSON for this plan (2026-08-19), not
from the manuscript or from prior draft letters. Where a value was additionally
**re-derived from per-subject data** for this plan, that is marked ✔R. Figure numbers
refer to the resubmission figure set in `07_FINAL_FIGURES/` (fig01–fig23).

Legend for "Status": **USED-v1** = in the submitted manuscript · **NEW** = run after the
19-Jul rejection · **UNUSED** = exists but not in any manuscript version ·
**EXCLUDED** = must not be cited (reason given).

---

## A. Headline result and its interrogation (real 64 mT, ds006557, n = 23)

| Experiment | What it tested | Real numbers (from file) | Figure | Serves | Status |
|---|---|---|---|---|---|
| `loocv_cross_session/results.json` | Cross-session LOOCV, LN+head adapter (769 p), HFC→HFE | **MAE 0.0134** [bootstrap 95 % 0.010–0.017], RMSE 0.016, bias −0.0022, 19/23 < 0.020; Pearson r (pred vs truth) **−0.1191** (p 0.5882), Spearman −0.2461 (p 0.2576); **ICC(3,1) 0.6146** [0.2364–0.8658]. Per-subject worst: HYPE21 0.0334, HYPE01 0.0288, HYPE16 0.0257, HYPE19 0.0207 ✔R | fig12, fig13, fig15 | headline; R1.1, R3.4 | USED-v1 |
| — derived from the same per-subject data ✔R | Constant-predictor baselines | **LOO cohort-mean MAE 0.01281**; global-mean 0.01225; LOO-median 0.01227; cohort true range 0.7523–0.8084 (width **0.0561**), true SD 0.0152 vs prediction SD **0.0041** (≈4× range compression) | — | R1.1, R3.4 | corrects v1's "0.0126" |
| `permutation_test/results.json` | Is the model a constant predictor? (20,000 perms, seed 42) | Test 1 inter-session ICC: observed 0.6146, null mean ≈ 0, **p = 0.0024** (input-dependent). Test 2 anatomical tracking: r = −0.1191, **p = 0.590** (does NOT track true nWBV). File also stores global_mean_mae 0.0122, loo_mean_mae 0.0128 | fig14 | **R1.1, R3.4, R3.5** (file's own `addresses` field) | NEW |
| `multiseed_loocv/results.json` | Seed robustness (42, 1, 7, 123, 2024) | **MAE 0.0130 ± 0.0004** (range 0.0123–0.0134); **ICC 0.6437 ± 0.0583** (min seed 7: 0.5629, max seed 123: 0.7142); below-threshold counts 21/18/19/19/20 | fig08 | R1.9, R1.2 | NEW (pre-dates rejection per CLAUDE.md; folded in revision) |
| `ablation_adapter/results.json` + `ablation_lora/results.json` | 4-way adapter ablation, identical LOOCV protocol | head-only 257 p **0.0133** [0.0099–0.0170]; LN+head 769 p **0.0137** [0.0104–0.0172]; LoRA r=4 41,217 p **0.0128** [0.0099–0.0160]; full-FT 4,225,537 p **0.0123** [0.0091–0.0158]. All CIs overlap. `ablation_lora` self-labels `addresses: R2.4` | fig18 | R2.4, R1.5, R3.5 | NEW (LoRA arm) |
| `conformal_calibration/results.json` | Split-conformal (leave-one-out) on LOOCV residuals | 90 %: **91.3 % coverage**, width 0.0571; 95 %: **95.65 % coverage**, width **0.0664** (vs cohort range 0.056). MC-Dropout baseline in same file: 4.3 %, width 0.0286. `addresses: R1.7` | fig19 | R1.7, R3.7 | NEW |
| `real64mt_eval/mc_dropout_ci.json` | MC Dropout on real 64 mT (N = 100 passes), **unadapted** model (MAE 0.0403 = the no-adapt figure) | width **0.0286**, coverage **4.3 %**, MAE 0.0403 | fig21, fig23 | R1.7 | USED-v1 (but v1 figures drew CIs from `np.random`; rebuilt) |
| `paper_statistics/failure_analysis.json` | Failure characterisation | OASIS tertiles: low-nWBV MAE **0.1254** vs high 0.009 (13/12/13 split); worst case true 0.645 → pred 0.8407. Real 64 mT: age-vs-error r 0.6776 (p 0.0004, unadapted); **CI-width-vs-error r = −0.1644, p = 0.4535** (the correct values for v1's "+0.34/0.108") | fig22 (+fig22D corrected) | R1.7 integrity fix; failure analysis | USED-v1 (stats corrected) |
| `dino_headline_loocv/results.json` | Best OASIS SSL objective (DINO) carried through the full cross-session LOOCV | DINO: **MAE 0.0220**, bias +0.0085, **ICC 0.5231**, **10/23** below threshold (vs denoising 0.0134 / 0.615 / 19/23). Per-subject data present. Paired vs denoising ✔R: denoising lower in 14/23; Wilcoxon W⁺ 78 / W⁻ 198, z ≈ −1.83, **p ≈ 0.068–0.070 (n.s.)**; paired bootstrap ΔMAE **−0.0086**, 95 % CI ≈ **[−0.0166, −0.0013]** (RNG-dependent in the last digit — report rounded [−0.017, −0.001] or state seed) | fig05 panel B (to be rebuilt — see 07) | **R2.3 headline decision** (file's own label), R2.2 transfer narrative | NEW (18 Aug) |

## B. OASIS-1 / high-field context (n = 38 test)

| Experiment | What it tested | Real numbers | Figure | Serves | Status |
|---|---|---|---|---|---|
| `oasis_bootstrap/results.json`, `oasis_cnn_comparison/results.json` | ViT3D vs CNN3D on OASIS-1 | ViT3D **MAE 0.0584** [0.0408–0.0761], r **0.7212** [0.523–0.846], bias +0.0533, RMSE 0.0805; CNN3D **MAE 0.0243** [0.0176–0.0314], r **0.8773** [0.775–0.935], bias −0.0069 | fig04 | R2.1 | USED-v1 |
| `significance_tests/results.json` | Formal tests (`addresses: R1.8`) | **T1** ViT vs CNN OASIS, paired Wilcoxon n = 38: ΔMAE 0.0341 [0.016–0.0531], **p = 0.0037** (CNN better). **T2** physics-vs-blur *fidelity*, paired n = 23: NCC 0.4967 vs 0.5156, **p = 0.02 (blur better)** — note the file's own `note` field wrongly says "non-significant"; the stored W⁺ 61 / W⁻ 215, z = −2.327, p = 0.02 is the authority — SNR-error 0.8942 vs 0.1267, **p < 0.001** (blur closer). **T3** physics-vs-blur *downstream*, **unpaired** (different test splits) Mann-Whitney n = 75/arm: MAE 0.0146 vs 0.0148, **p = 0.8099**; file states v1's "Wilcoxon p = 0.847" is not reproducible | — (Table VIII) | R1.8, R2.1, R1.4 | NEW |
| `ablation_gaussblur/results.json` | Physics-sim vs Gaussian-blur pre-training (separate 75-sample test draws) | physics r 0.9493 [0.9207–0.9678], MAE 0.0146; blur r 0.9314 [0.8932–0.9562], MAE 0.0148; Δr +0.0179, ΔMAE 0.0002 — **descriptive only; arms not paired** | fig07 | R1.5 | USED-v1 (stats corrected via T3) |
| `simulation_sensitivity/results.json` | ±20 % on SNR / B0 / relaxation (OASIS test, n = 38, baseline MAE 0.0583) | **max \|ΔMAE\| = 0.0062** (relaxation −20 %); SNR ±20 %: ∓0.0005; B0 ±20 %: ≤ 0.0002 | fig02 | R1.10 | NEW-ish (pre-rejection run, unused in v1) |
| `sim_validation/results.json` | Simulation fidelity vs real 64 mT, physics vs Arnold-2021 baseline (Gaussian blur + histogram matching), n = 23 paired | physics NCC **0.4967 ± 0.0463**, SSIM ≈ 0.07–0.09, SNR ≈ 33 (snr_err 0.894); blur NCC **0.5156 ± 0.0558**, SSIM ≈ 0.14–0.22, snr_err 0.127. Real Hyperfine SNR ≈ 320–350 ✔R | — (candidate new table) | **R1.4** | UNUSED in v1 — promote |
| `arch_comparator_swin/results.json` | Swin-UNETR (62,187,011 p) on OASIS-1, same seed-42 protocol | **MAE 0.0148**, r **0.9636** ✔R (bootstrap ✔R: MAE [0.0113–0.0184], r [0.9407–0.9826]) | fig05 panel A | **R2.2** | NEW (3 Aug) |
| `arch_comparator_unetr/results.json` | UNETR (92,685,649 p), same protocol | **MAE 0.0198**, r **0.9226** ✔R [0.8827–0.9566]; **inference 267 ms CPU** | fig05 panel A | **R2.2** | NEW |
| `ssl_comparator_dino/results.json` | DINO pretraining (25 ep) → OASIS finetune | **MAE 0.0270**, r 0.9032 | fig05 panel A (rebuild) | **R2.3** | NEW (18 Aug) |
| `ssl_comparator_mae/results.json` | Masked Autoencoder (mask 0.75) | **MAE 0.0319**, r 0.7561 | fig05 panel A (rebuild) | **R2.3** | NEW (18 Aug) |
| `ssl_comparator_simmim/results.json` | SimMIM (mask 0.5) | **MAE 0.0538**, r 0.9069 | fig05 panel A | **R2.3** | NEW (3 Aug) |
| `ssl_comparator_contrastive/results.json` | SimCLR InfoNCE | **MAE 0.0589**, r 0.7442 | fig05 panel A (rebuild) | **R2.3** | NEW (18 Aug) |
| `oasis_mc_dropout/results.json` | **Real** MC Dropout on OASIS-1 test (N = 100) | coverage **23.68 %** (9/38), mean width **0.0284** | fig20 (rebuilt 19 Aug) | R1.7 + integrity I4/I5 | NEW (19 Aug) |
| `simulated_dementia/results.json` | CDR-stratified stress test (physics-sim 64 mT, no adaptation) | overall 0.0554 (r 0.2346, p 0.156); no-CDR (n = 20) **0.0156 ± 0.0100**; CDR 0.0 (8) **0.0865 ± 0.0512**; CDR 0.5 (8) **0.0987 ± 0.0314**; CDR 1.0 (2) **0.1567 ± 0.0117** | fig16 | R1.6, R3.6 | USED-v1 |
| `pseudolabel_ablation/results.json` | SynthSeg+ pseudo-labels vs FastSurfer GT (15 train / 8 test) | label agreement r **0.9183**, MAE 0.0051, bias −0.0022; adapter: GT **0.0121** [0.0048–0.0212] bias +0.006 vs pseudo **0.0171** [0.0116–0.0227] bias −0.0105; penalty **+0.005** | fig17 | deployment feasibility | USED-v1 |
| `synthseg_output/synthseg_comparison.json` | SynthSeg+ on the axial 64 mT scans (n = 23) | r **0.9183**, Spearman 0.9209, MAE **0.0051** — the "SynthSeg+ upper bound r 0.918 / MAE 0.005" | fig09/fig10 | accuracy ceiling; R4.4 | USED-v1 |
| `ablation_vit_vs_cnn/results.json` | ViT vs CNN on real 64 mT, no adaptation | ViT r 0.2907, **MAE 0.0403**, bias +0.0403; CNN r 0.5135, **MAE 0.0763**, bias −0.0763; age-error correlations −0.49/−0.58 (unadapted — **do not surface**, see R1 Exposure 2 history) | fig09, fig10 | R2.1 regime argument | USED-v1 |
| `paper_statistics/*.json` | Aggregates | Stage-1 ViT denoising r 0.888; OASIS CDR-group *predicted* means all ≈ 0.844 (regression to mean); age-stratified unadapted MAE <35: 0.0317 / 35–55: 0.0357 / >55: 0.0529 | fig22 panel C | failure analysis | USED-v1 |

**CDR statistics caution** (from `Review - 2/MANUSCRIPT_AUDIT.md`, confirmed logic): the
manuscript's Kruskal–Wallis H = 4.930, p = 0.085 is computed on **true** nWBV of the 18
CDR-labelled test subjects and reproduces; `paper_statistics/cdr_statistics.json` stores
H = 2.883 computed on **predictions** — a different question. Do not "fix" one with the
other. Effect size: pooled-SD Cohen's d = **1.4625** (manuscript's 1.368 ≈ Glass's delta,
mislabelled).

## C. Parameter counts (hand-verified against `models/baselines.py` for this plan) ✔R

| Model | Exact count | Notes |
|---|---|---|
| ViT3D | **4,225,537** (4.23 M) | matches manuscript and `ablation_adapter.full_ft.trainable_params` |
| CNN3D | **8,222,337** (8.22 M) | manuscript's "≈ 4.1 M" and the `cnn3d_params: 4100000` fields in two JSONs are **wrong**; correct everywhere, incl. fig05 label |
| Swin-UNETR | 62,187,011 | from `arch_comparator_swin` |
| UNETR | 92,685,649 | from `arch_comparator_unetr` |
| Adapter | LN 512 + head 257 = **769** | consistent |

## D. Excluded / do-not-cite material (each with the reason)

| Item | Why excluded |
|---|---|
| `transfer_probe_64mt/results.json` (ViT 0.0112 vs Swin 0.0117 frozen-feature ridge probe) | **Methodologically unsound as run**: Swin was probed on `feat_dim: 8` (decoder segmentation channels) vs ViT's 256-dim encoder features — a 32× representation handicap, documented as Blocker 1 in `RESPONSE_LETTER_ALL_REVIEWERS.md`. No fair re-run exists. **Remove from manuscript and figures** (it is present in the WIP abstract, §V, fig05 panel B, Discussion). Its narrative role is carried by the DINO LOOCV transfer (row A, sound). |
| `loocv_cross_session_v2/results.json` (TTA + top-3 ensemble + bias correction: MAE 0.0129, ICC 0.330) | Different protocol; its ICC (0.330 [−0.043, 0.673]) would silently contradict the reported 0.615. July plan flag F4 — concur. Never cite. |
| `real64mt_loocv/loocv_results.json` (MAE 0.0124, r 0.170) | Earlier LOOCV variant, superseded by `loocv_cross_session/`. Citing both invites an inconsistency hunt. |
| `zenodo_validation/zenodo_validation_results.json` (van den Broek Zenodo cohort, 10 subjects) | Evaluates **different biomarkers** (BTF/VBR/TCR/MCI, intensity-threshold proxies; GT btf ≈ 0.30–0.42, incommensurate with nWBV ≈ 0.78) from an earlier project lineage. **Cannot honestly serve as external validation of the nWBV pipeline.** Its only legitimate uses: (a) proof the Zenodo dataset exists (R1.3 correction), (b) the honest statement that prior exploratory work on that cohort used non-comparable proxies. |
| `ablation_arnold/results.json`, `oasis_validation/*.json` | Same earlier multi-biomarker lineage (BTF/TCR/VBR/MCI); not comparable to this paper's nWBV task. Do not cite numbers from these. |
| `real64mt_finetune/`, `synthseg_finetune/` (fixed 15/8 splits, MAE 0.0096–0.0098) | Superseded by the LOOCV protocol; the fixed-split MAEs on n = 8 would read as cherry-picks next to 0.0134. The **pseudo-label** portion survives via `pseudolabel_ablation/`. |
| `stage1*/`, `stage2/` folders | Training artifacts/logs; only the Stage-1 denoising r 0.888 (in `paper_statistics`) is citable. |

## E. Claims in circulation that have **no measurement behind them** (verify or soften)

| Claim | Where it appears | Status |
|---|---|---|
| 47 ms inference on "a standard GPU" | abstract, contributions, discussion, conclusion; hardcoded in `generate_all_figures.py` etc. | No measurement file in repo. No reviewer challenged it (R3 endorsed it). **Keep only if an author can vouch for the original measurement**; otherwise restate as parameter-count/compute-class feasibility. Decision gate before upload. |
| Age correlations ρ = −0.778 (GT) / +0.232 (adapted preds) | abstract, §V, discussion, fig11 | `participants.tsv` absent on this machine; not re-derivable here (only group means survive in `age_stratified_error.json`). **Verify on the dataset machine** (MANUSCRIPT_AUDIT B2). |
| "~13 days per arm (Swin) vs ~20 h (ViT3D)" full-protocol compute estimate | Aug-10 letter, WIP discussion | An estimate, not a measurement. Present as an estimate ("approximately", "on our CPU-only hardware") — never as a benchmark. |
| Swin/UNETR/SSL comparator CIs quoted in the Aug-10 letter (e.g. Swin r CI [0.9340, 0.9828]) | letter draft | Bootstrap RNG-dependent; my re-derivation gives [0.9407, 0.9826]. **Cite point estimates only** in the letter/manuscript, or recompute-and-freeze one CI set with a stated seed. |

## F. Comparability caveats to state wherever the comparators appear

1. SSL comparators used a **25-epoch pretraining budget, single seed, untuned per
   objective**; whether the paper's own Stage-1 budget matched is unverified — never
   claim matched budgets (per `WORD_AGENT_PROMPT_R2.3_SSL.md`, confirmed).
2. Only DINO (the strongest OASIS comparator) was carried to 64 mT LOOCV; MAE/SimMIM/
   contrastive were not — never imply a 64 mT comparison that doesn't exist.
3. Swin-UNETR/UNETR were evaluated on OASIS-1 only, not through Stage 3 — state the
   evaluation-depth asymmetry as a limitation.
4. `ablation_gaussblur` arms trained/evaluated on different test draws — descriptive r
   values only; the valid test is T3 (unpaired, p = 0.81).
5. The headline LN+head run (0.0134) and the ablation LN+head arm (0.0137) are
   independent runs of the same nominal config; report both with the run-provenance
   footnote (July plan F2 — concur).

---

## Added 2026-08-23

| Experiment | What it tested | Numbers (from the raw JSON) | Figure | Serves |
|---|---|---|---|---|
| `zenodo_external_validation/` (commit e389ba6) | External validation on van den Broek et al. Zenodo cohort. FastSurfer nWBV ground truth via the same `--seg_only` pipeline and BrainSeg/Mask definition as OASIS-1/ds006557 — labels directly comparable, not proxies. 11 paired subjects, **10 analysable** (`sub-0064` incomplete). No subject in any training/pretraining/adaptation stage. | **64 mT-adapted:** MAE 0.0731, bias −0.0731, r 0.255 (p 0.477), preds [0.7787, 0.7870], mean 0.7824. **OASIS unadapted:** MAE 0.0327, bias −0.0327, r 0.264 (p 0.461), preds [0.8186, 0.8282]. **GT:** mean 0.8555, SD 0.0109, range [0.8360, 0.8694]. **Cohort constant-mean baseline: 0.0089.** Adaptation-cohort GT: mean 0.7841, range [0.7523, 0.8084]. Ranges do not overlap; every prediction below every GT; pred SD 3.8× narrower than truth SD. | fig24 | **R1.3**, R3.5, R1.2 |
| `model_param_counts/` | Authoritative parameter counts, analytic from `models/baselines.py`. | ViT3D **4,225,537** (manuscript 4.23M ✔). LN+head adapter **769** (✔). CNN3D **8,222,337** — manuscript says 4.1M ✘, actual is ~1.95× the ViT, so "matched-parameter comparator" was never true. Swin-UNETR 62,187,011; UNETR 92,685,649. | fig05 | R2.1, R2.2 |

### Do-not-cite additions

- **`zenodo_validation/`** (the earlier run) — **superseded, do not cite.** It compared
  intensity-threshold proxies (BTF = brain voxels / *total image voxels*, from
  `skull_strip_simple()`) against the model's FreeSurfer-defined nWBV. Different
  denominator (image volume, not intracranial volume), different derivation (threshold,
  not segmentation), and sensitive to field of view and cropping. Its weak correlations
  (BTF r 0.37, p 0.29) reflect that mismatch, not model generalisation. Superseded by
  `zenodo_external_validation/`, which uses comparable ground truth.
- **`arch_comparator_swin/results.json` → `comparison.cnn3d_params = 4100000`** — wrong;
  the source of the manuscript's false "4.1M" claim. Also `vit3d_params = 4230000` is
  rounded. Use `model_param_counts/` instead, and correct this file.

### Unverified — needs the dataset machine

- **Inference latency.** No measurement file exists; "47 ms on a standard GPU" is a
  hardcoded literal (see I10). Requires `experiments/inference_latency/results.json`
  before M29 and fig10 can be completed.
- **External cohort demographics.** Not distributed on the Zenodo landing page; check
  the data descriptor inside the archive. Determines whether the 0.07 nWBV range gap is
  biological (age) or methodological.
