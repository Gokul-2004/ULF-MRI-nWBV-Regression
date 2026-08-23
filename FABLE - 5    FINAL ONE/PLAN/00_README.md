# PLAN — Review-2 resubmission audit (Fable 5, 2026-08-19)

Analyst output only: nothing outside this folder was modified. Read in order.

| File | What it is |
|---|---|
| `01_DIAGNOSIS.md` | Why the paper was rejected; triage of all 28 concerns; the integrity debt (I1–I9); where this plan departs from the prior `Review - 2/` strategy. |
| `02_EVIDENCE_INVENTORY.md` | Every experiment in `experiments/`, with numbers read from the raw JSONs (✔R = re-derived from per-subject data for this plan); the excluded/do-not-cite list; unverifiable claims needing author sign-off. |
| `03_COVERAGE_MATRIX.md` | All 28 concerns × evidence × closable (YES 20 / PARTIAL 2 / NO 2) × closing edit IDs. |
| `04_THE_FOUR_HARD_ONES.md` | Positions on R1.1/R3.4 (mean-baseline), R2.1 (ViT vs CNN), R1.3 (external validation), R4.4/R4.5 (scope) — each with tier labels. |
| `05_MANUSCRIPT_CHANGES.md` | The edit spec: M1–M31 (M29–M31 added 2026-08-23) against the submitted PDF (exact BEFORE/AFTER), W1–W5 corrections to revision-added content, verification gates G1–G4. |
| `06_RESPONSE_TO_REVIEWERS.md` | Full 28-block response letter in IEEE template format, incl. the author-initiated-corrections preamble; ends with the concern → edit → evidence cross-check table. |
| `07_FIGURE_PLAN.md` | All 23 figures: format, order, caption changes, the fig05 rebuild spec (blocking), fig22 rebuild gate, anti-synthetic-data protections. |
| `08_RISK_ASSESSMENT.md` | Ranked residual risks, the pre-upload grep gate, and the named gamble. |

**Blocking before upload:** gates G1–G4 in 05; the fig05 rebuild and fig22 rebuild in
07; the zero/non-zero grep gate in 08 Risk 5.

**Headline judgment calls** (argued in 01/04): no retitle; ViT3D stays the headline
*system* with all superiority claims removed; R2.2/R2.3 answered with the August
experiment runs, not rebuttals; the frozen-encoder transfer probe is removed as
unsound (W1); the synthetic-interval figure defect and all five statistical
corrections are disclosed to the editor, not slipped in.
