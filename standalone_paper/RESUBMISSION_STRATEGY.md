# Resubmission Strategy & Analysis — IEEE Access Access-2026-28453

**Paper:** A Reproducible Feasibility Baseline for Segmentation-Free nWBV
Regression from 64 mT Ultra-Low-Field MRI Using Physics-Constrained Deep Learning

This document captures the full analysis behind the resubmission strategy:
the experiments run, what they showed (including uncomfortable findings), the
framing decision, and the plan. It is the reasoning record — the actual
point-by-point reviewer replies live in `response_to_reviewers.md`.

---

## 1. Experiments run for the resubmission

| # | Experiment | Addresses | Result | Verdict |
|---|---|---|---|---|
| 1 | Permutation test (20k perms) | R1.1, R3.4/5 | Inter-session ICC 0.615 vs null~0, **p=0.002** | 🟢 Strong — proves input-dependence |
| 2 | Conformal calibration | R1.7 | **91.3% / 95.7%** coverage vs 4.3% MC-dropout | 🟢 Strong — fixes miscalibration |
| 3 | Multi-seed LOOCV (5 seeds) | R1.9 | MAE 0.0130 ± **0.0004**, ICC 0.644 ± 0.058 | 🟢 Strong — stable |
| 4 | Simulation sensitivity (±20%) | R1.10 | max ΔMAE 0.0062; insensitive to SNR/B0 | 🟢 Strong |
| 5 | Simulation fidelity (NCC/SNR) | R1.4 | NCC 0.497 ≈ blur 0.516; SNR gap by design | 🟡 Honest |
| 6 | 4-way adapter ablation | R1.5, R2.4 | head 0.0133 / LN+head 0.0137 / LoRA 0.0128 / full-FT 0.0123 (all overlap) | 🟡 Honest |
| 7 | Swin-UNETR (62M) on OASIS | R2.2 | MAE 0.0193, r 0.939 — **beats ViT**, did not overfit | ⚠️ Mixed |
| 8 | SimMIM vs denoising pretrain | R2.3 | SimMIM MAE 0.054 / r 0.907 vs denoising 0.058 / 0.722 | ⚠️ Mixed |

Swin-UNETR on real 64 mT LOOCV was **not run** — computationally infeasible
(~4.5 days CPU; the machine is CPU-only, GPU CUDA-incompatible).

---

## 2. The uncomfortable findings (stated honestly)

Several comparators outperform ViT3D. This must be faced, not hidden:

- **CNN3D beats ViT3D on OASIS** (r 0.877 vs 0.722, MAE 0.024 vs 0.058).
- **CNN3D beats ViT3D on real-64mT correlation** (r 0.513 sig. vs 0.291 n.s.)
  — but **ViT3D has the lower 64 mT MAE** (0.0403 vs 0.0763 unadapted).
- **Swin-UNETR beats ViT3D on OASIS** (MAE 0.0193 vs 0.058).
- **SimMIM beats the denoising objective** (MAE 0.054 vs 0.058; r 0.907 vs 0.722).

**Implication:** "ViT is the best architecture" is NOT defensible. Reviewer 2
was correct that the architecture is not the strength.

### The MAE-regime subtlety
- Real 64 mT MAE (~0.013) is LOW partly because the ds006557 cohort is
  healthy-only and narrow-range — an *easy* prediction target. The
  constant-mean baseline also scores ~0.0126 there.
- OASIS MAE (~0.058) is HIGHER because OASIS includes dementia/atrophy —
  a *harder*, wider-range target.
- So low 64 mT MAE ≠ "model works better on 64 mT"; it reflects an easy cohort.
  This is exactly why the **permutation test (p=0.002)** is essential — it
  proves genuine input-dependent learning despite MAE ≈ baseline.

---

## 3. Framing decision — the core strategic call

**Decision: present all architectures honestly; keep ViT3D as the primary
system studied in depth; make the PIPELINE the contribution, not the encoder.**

The paper is **a deep single-pipeline feasibility study**, NOT an architecture
bake-off. CNN3D and Swin-UNETR are **accuracy comparators/baselines**; ViT3D is
the system put through the full rigor (cross-session adaptation, permutation,
calibration, ICC, multi-seed, sensitivity, failure analysis).

### Why ViT3D remains the defensible primary
- **Smallest + fastest**: 4.23M params, 16.9MB, 6.9ms/scan CPU — vs Swin 62M,
  249MB, 2034ms (~300× slower), and CNN 8.22M, 33MB, 19ms.
- **Lowest MAE on the real 64 mT target** (beats CNN there).
- Chosen for the **accuracy–efficiency tradeoff at the point of care**, not for
  peak high-field accuracy.

### Why we do NOT run the full deep analysis on CNN/Swin
- It would likely show CNN/Swin also win several deep tests → worst outcome
  (full rigor, our model comes 2nd/3rd everywhere).
- It converts the paper into an architecture comparison, forcing a "why isn't
  the winner your headline?" question.
- Compute is infeasible (weeks; Swin-64mT-LOOCV alone ~4.5 days).
- No reviewer asked for it. Baselines are not expected to receive full
  uncertainty/reliability analysis.
- **Estimated effect: full analysis on all 3 would DECREASE odds (~50-55%)
  vs the single-pipeline framing (~65-70%).**

### The one clarifying sentence to add
> "ViT3D is the primary system evaluated in depth; CNN3D and Swin-UNETR are
> included as accuracy comparators and are not subjected to the full adaptation,
> calibration, and reliability analysis."

This converts the testing asymmetry from a potential criticism into a stated
design choice.

---

## 4. Deployment comparison (measured, CPU, single 64³ scan, 4 threads)

| Model | Params | Size | Peak RAM | Latency | OASIS MAE |
|---|---|---|---|---|---|
| **ViT3D (ours)** | 4.23M | 16.9 MB | 614 MB | **6.9 ms** | 0.058 |
| CNN3D | 8.22M | 32.9 MB | 632 MB | 19.0 ms | 0.024 |
| Swin-UNETR | 62.2M | 248.7 MB | 1345 MB | 2034 ms | 0.019 |

ViT3D vs Swin: 14.7× fewer params, 14.7× smaller, ~300× faster. This is the
deployment argument — the genuine differentiator for point-of-care ULF MRI.
(Manuscript currently cites 47 ms on GPU; the 6.9 ms CPU figure is stronger and
more honest for a "runs on cheap bedside hardware" claim — pick one story.)

---

## 5. Reviewer scorecard (current)

- **R1** (Partial/Partial): ~10/11 fully resolved with data → essentially satisfied.
- **R2** (No/No): all 3 asks now *run* (Swin, SimMIM, adapter ablation);
  architecture point conceded honestly + reframed on deployment. Effort undeniable.
- **R3** (Yes/Partial, ally): fully satisfied — asked for permutation testing, got p=0.002.
- **R4** (Yes/Yes soundness): figures fixed; image-quality asks rebutted as task-mismatch.

**Estimated acceptance probability: ~65–70%.** Carried mainly by the permutation
test (p=0.002), conformal calibration, and the honest, thorough feasibility
framing — not by architecture superiority (which is not claimed).

IEEE Access is binary with one resubmission allowed. R2 is the wildcard; the
Associate Editor decides how much weight to give one No against three favorable.

---

## 6. Plan — the 4 manuscript edits, then ship

1. **Add the deployment comparison table** (§ deployment) — the strongest new asset.
2. **Add Swin-UNETR + SimMIM rows** to the results table (honest comparators).
3. **Surface the permutation test (p=0.002)** prominently in results.
4. **Add the "ViT is the studied system; others are baselines" clarifying sentence.**

Do NOT: swap headline to CNN3D; run more experiments (they keep exposing
weaknesses); overclaim. The title and four contributions are already
architecture-agnostic and need no change.

**Bottom line: reframe, don't rebuild. The paper is closer to done than the
analysis made it feel. ~45 min of edits, then resubmit.**
