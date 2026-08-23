# 01 — DIAGNOSIS: Why Access-2026-28453 was rejected

**Scope of this document.** Blunt cause-of-rejection analysis, concern triage, and the
strategic consequence. Every claim here is grounded in the verbatim reviews
(`02_Reviewer_Comments_VERBATIM_Review1.md`), the submitted PDF
(`01_Manuscript_AS_SUBMITTED_Review1.pdf`), and the raw `experiments/*/results.json`
(re-verified independently for this plan — see `02_EVIDENCE_INVENTORY.md`).

---

## 1. The one-sentence diagnosis

The paper was rejected because it asked reviewers to accept a system whose **every
component choice was unjustified by evidence** (architecture, pretraining objective,
adapter) and whose **headline accuracy does not beat a trivial baseline** — while the
honest feasibility framing that could have absorbed those facts was undercut by residual
overclaims, a non-reproducible statistic, and figure-quality problems.

IEEE Access is binary: with Reviewer 2 answering **No / No / No**, rejection was
automatic regardless of Reviewers 3 and 4 answering mostly Yes.

## 2. Who killed it, and why

| Reviewer | Verdict | What they actually objected to |
|---|---|---|
| **R2** | No / No / No — **the kill vector** | Every design choice presented without comparative evidence: ViT loses to CNN on OASIS yet is the headline architecture (R2.1); no justification vs Swin-UNETR/UNETR (R2.2); denoising pretraining never compared to MAE/DINO/SimMIM/contrastive (R2.3); LN+head adapter chosen "by intuition", with the paper itself deferring the comparison to future work (R2.4). |
| **R1** | Partially / Partially / Yes | The substance: model ≤ constant-mean baseline (R1.1); missing external validation (R1.3); missing ablations, significance tests, seeds, sensitivity (R1.5, R1.8–R1.10); miscalibrated uncertainty (R1.7); redundant text (R1.11). |
| **R3** | Yes / Partially / Yes — **the ally** | Endorses the feasibility framing and the LOOCV protocol explicitly; asks only that claims be bounded and that input-dependence be evidenced (R3.4, R3.5). R3's review is effectively a road map for acceptance. |
| **R4** | Yes / Yes / No | Blurred figures (R4.3) plus a generic demand for qualitative SOTA comparison across "broader medical image modalities" (R4.4–R4.6) that misreads the task (scalar regression, not image synthesis). |

## 3. Which criticisms are simply correct

Do not defend any of these. Concede them in the letter — they are true, and several are
now *worse* than the reviewers knew:

1. **R1.1 / R3.4 — the model does not beat the mean.** Correct, and the submitted
   baseline number was itself wrong: recomputed from the committed per-subject data, the
   leave-one-out constant-mean MAE is **0.0128** (global mean 0.0122), not 0.0126. The
   model's 0.0134 is **numerically above both**. Additionally, the adapted model's
   predictions do not correlate with true nWBV (Pearson r = −0.119, p = 0.590) and its
   prediction SD (0.0041) is ~4× narrower than the true SD (0.0152): this is
   range-compressed output, exactly as R3.4 suspects. The *only* rescue the data
   supports is input-dependence (permutation test on inter-session ICC, p = 0.0024) —
   which is real, but is *reproducibility without biological accuracy* and must be sold
   as exactly that.
2. **R2.1 — CNN3D beats ViT3D on OASIS.** Correct, and now formally significant (paired
   Wilcoxon p = 0.0037). Also newly discovered: the manuscript's "≈ 4.1 M parameters"
   for CNN3D is wrong — the committed model definition counts **8,222,337 (8.22 M)**,
   ~2× the ViT's 4,225,537. The "matched-parameter comparator" framing was never true.
3. **R2.2 / R2.3 — component choices unjustified.** Correct at submission time. Since
   run: Swin-UNETR (r = 0.964) and UNETR (r = 0.923) *both crush* ViT3D (r = 0.722) on
   OASIS-1; three of the four named SSL objectives (DINO 0.0270, MAE 0.0319, SimMIM
   0.0538) beat denoising (0.0584) on OASIS-1 MAE. The reviewers' suspicion was right.
4. **R2.4 — LN+head chosen by intuition.** Correct; the submitted paper literally says
   the comparison "is identified as future work". The 4-way ablation has since been run
   and shows LN+head is *not even the best arm* (0.0137 vs LoRA 0.0128, full-FT 0.0123,
   head-only 0.0133; all CIs overlap).
5. **R1.7 — severe miscalibration.** Correct (4.3 % coverage at nominal 95 %). Worse:
   the submitted §V-I's supporting statistic (r = +0.34, p = 0.108) was computed from
   `np.random.normal` placeholder intervals, not model output; the real value is
   **r = −0.164, p = 0.454**, and the sentence arguing from the positive sign is false.
   Three submitted uncertainty figures were built from those placeholders.
6. **R4.3 — blurred figures.** Correct (Word 220-ppi downsampling of 200–300 dpi
   sources).
7. **R1.11 — redundancy.** Correct; headline claims are restated up to 22 times each.
8. **R1.3's premise is stronger than the paper admits.** The submitted Limitations claim
   that ds006557 is "the only public paired 64 mT/3T brain MRI dataset" is **factually
   false** — the van den Broek et al. 2025 Zenodo cohort (11 paired subjects) exists, and
   this repository contains an exploratory analysis of it. A reviewer who knows that
   dataset would catch the paper in a falsifiable overclaim.

## 4. Triage of all 28 concerns

Full mapping in `03_COVERAGE_MATRIX.md`. Summary:

**Genuinely fatal if unaddressed — none are, given the post-rejection experiment record:**

- Nothing is unanswerable. The evidence base now covers every technical demand except
  two: external validation (R1.3) and broader modalities (R4.5). Both are handled by an
  honest limitation / scope argument, which the editor's letter explicitly invites.

**Closable with existing evidence (experiments already run, results committed):**

- R1.1, R1.2, R1.4, R1.5, R1.7, R1.8, R1.9, R1.10 (R1)
- R2.1, R2.2, R2.3, R2.4 (all of Reviewer 2 — this is the decisive change since
  submission; the July plan treated R2.2/R2.3 as rebuttal-only, but the August runs
  turned them into evidence-backed answers)
- R3.4, R3.5, R3.7
- R4.3 (figure set rebuilt at 600 dpi + vector SVG), the quantitative half of R4.6

**Framing-and-presentation only:**

- R1.6, R1.11, R3.2, R3.6, R4.1, R4.2 (positives/wording), R3.1, R3.3 (positives)

**Honest pushback / limitation (no evidence exists, none can be manufactured):**

- R1.3 (external validation — corrected claim + precise constraint + future work)
- R4.4 (qualitative image comparison — task outputs a scalar; argument must appear in
  the manuscript, not only the letter)
- R4.5 (broader modalities — out of scope for a single-modality feasibility baseline)

## 5. The integrity debt this revision must pay (self-inflicted, non-optional)

These are defects a re-checking reviewer *or editor* could catch, independent of any
concern number. Every one is fixed by existing evidence; all are Tier 0:

| # | Defect in the submitted manuscript | Correct value / action | Source |
|---|---|---|---|
| I1 | Constant-mean baseline "0.0126" (4 places) | **0.0128** (leave-one-out) / 0.0122 (global) | `permutation_test/results.json`; re-derived from `loocv_cross_session/results.json` per-subject data for this plan |
| I2 | "Wilcoxon p = 0.847" + paired Δr CI for physics-vs-blur | Invalid (arms used different test splits). Unpaired Mann-Whitney **p = 0.81** | `significance_tests/results.json` T3 |
| I3 | "r = +0.34, p = 0.108" (3 places) + the "consistent with well-calibrated direction" sentence | **r = −0.164, p = 0.454**; delete the direction sentence | `real64mt_eval/mc_dropout_ci.json`; `paper_statistics/failure_analysis.json` |
| I4 | Width 0.029 "on both the OASIS-1 test set and the real 64 mT cohort" — OASIS half had no measurement | Real OASIS MC Dropout now run: width **0.0284**, coverage **23.7 %** (9/38) | `oasis_mc_dropout/results.json` |
| I5 | Three uncertainty figures built from `np.random.normal` | Rebuilt from real data (fig20/21/22D) | `08_FIGURE_DATA_INTEGRITY_REPORT.md`; `oasis_mc_dropout/results.json` |
| I6 | "Cohen's d … = 1.368" | Pooled-SD Cohen's d = **1.463** (1.368 ≈ Glass's delta, mislabelled) | `Review - 2/MANUSCRIPT_AUDIT.md` B1 (recomputed from `oasis_finetune/finetune_results.json`) |
| I7 | CNN3D "≈ 4.1 M parameters" | **≈ 8.22 M** (8,222,337, hand-verified against `models/baselines.py` for this plan) | static count, this plan |
| I8 | "the only public paired 64 mT/3T brain MRI dataset" | False — van den Broek et al. (Zenodo, 11 paired subjects, 10 analysable) exists; external validation on it has since been RUN (commit e389ba6), so this claim is doubly wrong | `zenodo_validation/`; `WORD_AGENT_PROMPT_R1.3_FIX.md` |
| I10 | "47 ms on a standard GPU" (4 places) — no measurement file exists anywhere in the repo; the value is a hardcoded literal in four figure scripts, and `CLAUDE.md` records the development GPU as CUDA-incompatible, so a GPU benchmark could not have been produced | Withdraw the GPU claim; re-measure on CPU with a stated protocol; release `experiments/inference_latency/` (M29) | this plan, 2026-08-23 |
| I9 | Future-work items (iv) adapter ablation and (v) conformal prediction | Both are now done — must move from Future Work into Results or the paper contradicts itself | `ablation_lora/`, `conformal_calibration/` |

Also flagged (verify, don't assert): the age correlations ρ = −0.778 / +0.232 could not
be re-derived on this machine (`participants.tsv` absent) — verify on the dataset
machine before resubmission. The 47 ms GPU latency figure has no measurement file in the
repo (it is hardcoded in figure scripts); confirm its provenance with the authors — no
reviewer challenged it (R3 cited it approvingly), so it is retained, but the authors
must be able to stand behind it.

## 6. Assessment of the prior strategy (`Review - 2/`) — where I agree and where I depart

The July `RESUBMISSION_PLAN.md` and August response-letter draft are largely sound: the
exposure fixes (0.0128, p = 0.847, fidelity claim), the concede-first letter voice, and
the "protocol is the contribution" reframe are all correct and are retained. Four
departures, each argued in `04_THE_FOUR_HARD_ONES.md`:

1. **R2.2/R2.3 are no longer rebuttals.** The July plan rebutted them ("not run, future
   work"). The August experiments answered them. The revision must present the full
   four-objective SSL comparison + the DINO 64 mT transfer failure, and the
   Swin/UNETR comparison + compute rationale. A rebuttal where evidence now exists would
   look evasive *and* waste the strongest new material.
2. **The frozen-encoder transfer probe (ViT 0.0112 vs Swin 0.0117) must be removed, not
   kept "pending review".** As run, it pooled Swin-UNETR's 8 decoder output channels
   against ViT3D's 256-dim encoder features — a 32× representation handicap. The Aug 10
   letter itself flags it as methodologically unsound (Blocker 1); no fair re-run
   exists, and none may be run. Its narrative role ("high-field advantage does not
   transfer") is now carried legitimately by the DINO full-LOOCV transfer test
   (MAE 0.0220 vs 0.0134; 10/23 vs 19/23 below threshold). Every occurrence (abstract,
   results, fig05 panel B, discussion) goes.
3. **The revision must disclose its own corrections.** The Aug 10 letter predates the
   discovery of I3–I5. The new letter must own the synthetic-interval figure defect and
   the corrected statistics openly (under R1.7/R1.8): reviewers will see fig20/21/22
   change wholesale, and a disclosed correction reads as diligence where a silent one
   reads as concealment.
4. **No retitle, no headline-model change.** The title already says "Feasibility
   Baseline" (R3.2 endorses it); the ViT3D LOOCV stays the headline because it is the
   only system carried through Stage 3 — promoting CNN3D would require experiments that
   do not exist. R2.1 is answered by demotion of the *claim* (Tier 2), not replacement
   of the *result* (Tier 3).

## 7. What acceptance looks like

The resubmission wins if all four reviewers can each verify, inside the manuscript, that
their specific concern produced a visible change:

- **R1** sees: corrected baseline arithmetic, permutation test, conformal calibration,
  significance-test table, multi-seed, sensitivity sweep, tightened dementia language,
  a real consolidation pass, and an honest external-validation limitation that corrects
  a factual error in the prior version.
- **R2** sees: every one of his four concerns answered with a *run experiment*, three of
  which came out against the paper's own choices and are reported anyway, plus the
  removal of every architecture-superiority claim.
- **R3** sees: the feasibility framing preserved and extended exactly as prescribed, and
  the anatomy-vs-mean question answered with the permutation pair (input-dependent: yes,
  p = 0.0024; tracks anatomy: no, r = −0.119).
- **R4** sees: a visibly sharper figure set (600 dpi/vector), an expanded quantitative
  SOTA comparison, and — in the manuscript body, not just the letter — a stated scope
  argument for why qualitative image comparison and multi-modality evaluation do not
  apply to a scalar-output, single-modality feasibility study.

The residual gamble (R2's disposition toward "we kept the losing component, honestly";
R4's modality demand) is quantified in `08_RISK_ASSESSMENT.md`.
