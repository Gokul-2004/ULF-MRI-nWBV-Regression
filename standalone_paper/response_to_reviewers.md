# Response to Reviewers — IEEE Access Manuscript ID Access-2026-28453

**Title:** A Reproducible Feasibility Baseline for Segmentation-Free nWBV
Regression from 64 mT Ultra-Low-Field MRI Using Physics-Constrained Deep Learning

We thank the Associate Editor and all four reviewers for their careful and
constructive assessments. We are grateful that Reviewer 3 recognised the
leakage-free cross-session LOOCV protocol as "one of the strongest aspects of
the manuscript" and that the feasibility framing was judged appropriate. We
have addressed every concern below. New experiments have been added
(adapter-strategy ablation, multi-seed robustness, simulation-parameter
sensitivity, and simulation-fidelity validation), the ViT-vs-CNN result has
been reframed honestly, redundancy has been reduced, and all figures have been
regenerated at higher resolution.

For each comment we give: **(a)** the reviewer's concern, **(b)** our response,
**(c)** the action taken in the manuscript.

---

## Reviewer 1

### R1.1 — Model does not beat the constant-mean baseline; avoid overstating

**(a)** The proposed model does not outperform the constant-mean baseline
(MAE 0.0134 vs. 0.0126); please avoid overstating effectiveness.

**(b)** We agree and had already disclosed this comparison. We have never
claimed superiority over the constant predictor; the paper is explicitly framed
as a *feasibility baseline*. The evidence for input-dependent (non-trivial)
behaviour is the inter-session ICC(3,1) and the new adapter-strategy ablation
(R1.5 / R2.3), not a headline MAE win. We have made this bounding even more
explicit.

**(c)** Reinforced feasibility framing in the abstract, §I, and §VI; the
constant-mean comparison is stated plainly wherever the primary MAE appears,
and the claim of anatomy-dependent learning now rests on ICC + adapter ablation.

### R1.2 — Only 23 real subjects; discuss small-sample impact

**(b)** ds006557 (n = 23) is, to our knowledge, the only public paired
64 mT/3 T brain-MRI dataset at the time of writing; the sample size reflects
data availability, not study design. Bootstrap CIs are reported throughout.

**(c)** Small-sample impact is discussed in §VI (Limitations); we now also
report **multi-seed robustness** (5 seeds) so that the reported MAE/ICC are
shown to be stable rather than a single-seed artefact (see R1.9).

### R1.3 — External validation on an independent dataset

**(b)** No second public paired 64 mT dataset exists for external nWBV
validation. Our cross-session LOOCV (train on HFC sessions, test on the
held-out subject's HFE session) is itself an internal generalisation test
across independent acquisitions. We state the absence of an external cohort as
a limitation and identify prospective multi-site acquisition as the required
next step.

**(c)** Clarified in §VI; prospective external validation named as the primary
future-work item.

### R1.4 — Quantitatively validate the physics simulation against real 64 mT data

**(b)** Done. Because simulated and real volumes are not co-registered and have
different fields of view, voxelwise SSIM/PSNR are not physically meaningful
across field strengths; we instead report normalised cross-correlation (NCC)
of tissue structure and SNR/CNR statistics on the n = 23 paired subjects, and
compare against the Gaussian-blur baseline. NCC is statistically
indistinguishable between the two (0.497 vs. 0.516), and we now state
transparently that the simulator's SNR is deliberately far below the real
scanner's post-averaging SNR. We emphasise that the operative validation of the
simulator is the downstream physics-vs-blur task ablation.

**(c)** New "Simulation fidelity" subsection added to §III-A with the NCC and
SNR/CNR comparison and honest interpretation.

### R1.5 — More ablations (physics sim, SSL pretraining, ViT arch, LN+head)

**(b)** We have added an **adapter-strategy ablation** comparing head-only
(257 params), LayerNorm+head (769 params, ours), and full fine-tuning
(4.23 M params) under the identical cross-session LOOCV protocol. The
physics-vs-blur pre-training ablation was already present (Table III). We
address the architecture and self-supervised-objective questions in response to
Reviewer 2 (R2.2, R2.3).

**(c)** New adapter-strategy ablation table added to §V. Under the identical
cross-session LOOCV protocol: head-only (257 params) MAE $= 0.0133$
[0.0099, 0.0170]; LayerNorm+head (769 params) MAE $= 0.0137$ [0.0104, 0.0172];
full fine-tuning (4.23 M params) MAE $= 0.0123$ [0.0091, 0.0158]. The three
95\% confidence intervals overlap substantially, so the strategies are
statistically indistinguishable at $n = 23$; LayerNorm+head is retained for
its parameter efficiency in the compute-constrained deployment setting rather
than as the single lowest-MAE option, which we now state explicitly.

### R1.6 — Dementia evaluation relies on simulation and few pathological subjects

**(b)** Agreed. The CDR-stratified analysis uses *physics-simulated* atrophy on
OASIS-1 and n = 2 real CDR = 1.0 subjects; it is explicitly labelled
directional, and we draw no clinical conclusion from it.

**(c)** All CDR = 1.0 results carry the "n = 2, directional" caveat; §VI states
the model is not suitable for dementia screening or pathological morphometry.

### R1.7 — Uncertainty severely miscalibrated (4.3% coverage)

**(b)** Agreed; this is reported as a negative result. The MC Dropout intervals
are severely overconfident under domain shift, are reported for transparency
only, and are explicitly excluded from any accuracy or clinical claim.

**(c)** §V uncertainty subsection and Fig. (calibration) now state
"overconfident — intervals too narrow"; conformal calibration on prospective
data is identified as the remedy and future work.

### R1.8 — Strengthen statistical comparisons with significance tests

**(b)** The manuscript already reports Wilcoxon signed-rank (physics vs. blur),
paired bootstrap (ViT vs. CNN), Kruskal-Wallis and Mann-Whitney U (CDR
stratification), and bootstrap CIs on all MAE/ICC values. The new adapter
ablation and multi-seed analysis add further quantitative comparison.

**(c)** Significance tests retained and extended; adapter-strategy differences
reported with bootstrap CIs.

### R1.9 — Robustness across multiple random seeds / repeated experiments

**(b)** Done. We re-ran the full 23-fold LN+head LOOCV across five seeds
(42, 1, 7, 123, 2024) using the identical published protocol.

**(c)** New multi-seed robustness result added to §V. Across the five seeds the
LN+head LOOCV MAE is $0.0130 \pm 0.0004$ (range 0.0123--0.0134) and
ICC(3,1) is $0.644 \pm 0.058$ (range 0.563--0.714). The very small MAE spread
demonstrates that the reported result is stable and not a single-seed artefact;
seed 42 reproduces the manuscript value (0.0133).

### R1.10 — Justify simulation parameters; sensitivity analysis

**(b)** Done. We added a sensitivity analysis perturbing the three principal
simulation parameters (effective SNR, B0 inhomogeneity amplitude, and T1/T2
relaxation values) by ±20% about their published defaults, holding the trained
model fixed and re-evaluating OASIS-1 test MAE.

**(c)** New simulation-sensitivity table added to §III/§V. Holding the trained
model fixed and perturbing each parameter ±20\% about its published default, the
OASIS-1 test MAE (baseline 0.0583) changes by: SNR $\pm0.0005$; B0
$\pm0.0002$; and tissue relaxation $-0.0025$ / $+0.0062$. The result is
essentially insensitive to the noise (SNR) and B0 constants and only mildly
sensitive to the relaxation values that govern tissue contrast (worst-case
$|\Delta\mathrm{MAE}| = 0.0062$, $\approx 10\%$). This indicates the pipeline
does not depend on precise tuning of the simulation constants; relaxation
provenance is cited (Table II, low-field relaxometry).

### R1.11 — Reduce redundancy (MAE, ICC, limitations)

**(b)** Agreed. The ICC discussion was duplicated between §VI-D and §VI-E; we
consolidated it into a single passage with a cross-reference.

**(c)** Redundant ICC paragraph in §VI-E replaced by a cross-reference to
§VI-D; limitations tightened.

---

## Reviewer 2

### R2.1 — ViT underperforms CNN on OASIS; gains come from adaptation, not architecture

**(a)** ViT3D (r = 0.722, MAE = 0.058) is worse than CNN3D (r = 0.877,
MAE = 0.024) on OASIS; the reported gains appear to come from the adaptation
procedure rather than the transformer.

**(b)** This is a fair reading and we now state it explicitly rather than
implicitly. On the *high-field OASIS distribution* the CNN is indeed the
stronger regressor; we do not claim architectural superiority there. The ViT's
relevance is specific to the *adapted low-field* setting, and even there we are
careful to attribute the low LOOCV MAE partly to the narrow nWBV range rather
than to the architecture alone. We have removed any wording that could be read
as a general claim of ViT superiority.

**(c)** §V ViT-vs-CNN paragraph rewritten to state plainly that CNN3D
outperforms ViT3D on OASIS and that the architecture is not claimed to be
generally superior; the contribution is the pipeline (physics simulation +
leakage-free adaptation + failure characterisation), not the encoder choice.

### R2.2 — No justification for a basic 4-layer ViT over Swin-UNETR/UNETR/etc.

**(b)** The compact 4-layer ViT (4.23 M params) was chosen deliberately for the
compute-light, point-of-care setting that motivates the paper, and because the
target is a single global scalar (nWBV) rather than a dense segmentation, for
which hierarchical encoder–decoder transformers (Swin-UNETR, UNETR) are
designed. With only 64 patch tokens at 64³ input, a deep hierarchical model is
not warranted and risks overfitting on n ≤ 375 labelled volumes.

**(c)** §III-B now states this design rationale explicitly (task is scalar
regression, deployment is compute-constrained, small labelled set); a
systematic architecture comparison is named as future work. [If a Swin-UNETR
comparison is run, its result will be added here.]

### R2.3 — Denoising-AE pretraining not shown better than MAE/DINO/SimMIM

**(b)** Our Stage-1 objective is not a masked-patch reconstruction but a
*physics-grounded* one: reconstruct the real high-field volume from its
simulated low-field counterpart. The learning signal is the field-strength
degradation itself, which is precisely the domain gap the downstream task must
bridge — a signal that generic masked/contrastive SSL does not provide. We have
clarified this distinction and softened any claim that the denoising objective
is universally superior; a head-to-head comparison against MAE/SimMIM is named
as future work.

**(c)** §III rationale for the denoising objective clarified; over-claim
removed; SSL comparison added to future work.

### R2 (adapter design based on intuition)

**(b)** The LayerNorm+head choice is now supported by the new adapter-strategy
ablation (R1.5), which compares it directly against head-only (257 params) and
full fine-tuning (4.23 M params) under the same LOOCV protocol. We report the
result transparently: all three strategies fall within overlapping 95\%
confidence intervals (head-only 0.0133, LN+head 0.0137, full-FT 0.0123), so
none is demonstrably superior at $n = 23$. We therefore no longer present
LN+head as the empirically optimal adapter; it is selected for parameter
efficiency, and the ablation itself is the evidence the reviewer requested.

**(c)** Adapter-strategy ablation added; the LN+head justification is now
evidence-based and stated as a parameter-efficiency choice, not an MAE-optimality
claim.

---

## Reviewer 3

We thank Reviewer 3 for the detailed and supportive assessment.

### R3.2 — Maintain feasibility framing consistently

**(c)** Title, abstract, discussion, and conclusion checked for consistency;
all avoid implying clinical readiness.

### R3.4 / R3.5 — Better evidence of anatomy-dependent (not mean-regressed) learning

**(b)** Addressed by the new adapter-strategy ablation and multi-seed analysis,
which show the adapted model's behaviour depends on the trained adapter rather
than reducing to a constant, and by the retained ICC(3,1) inter-session result.

**(c)** New ablations added; ICC interpretation retained with honest CI bounds.

### R3.6 — State clinical generalisability more conservatively

**(c)** §VI states the healthy-only cohort limitation and that the model must
not be used for dementia screening, pathological morphometry, or longitudinal
monitoring in its current form.

### R3.7 — Bound physics-sim, uncertainty, and deployment claims

**(c)** Physics-vs-blur advantage stated as small and non-significant
(Wilcoxon p = 0.847); MC Dropout stated as severely miscalibrated; 47 ms is
described as computational feasibility only, with edge deployment as future work.

---

## Reviewer 4

### R4.3 — Blurred figures

**(c)** All figures regenerated at 300 dpi; embedded-text and axis labels
sharpened.

### R4.4–R4.6 — Compare with SOTA via qualitative image analysis and across
broader image modalities / image-quality comparisons

**(b)** We respectfully note a task mismatch. This work performs *scalar
biomarker regression* (a single nWBV value per scan), not image synthesis or
super-resolution, so image-quality metrics (SSIM/PSNR image comparison) and
qualitative image-based SOTA comparison are not applicable to the model output.
The relevant SOTA comparators for our task are morphometry pipelines
(FastSurfer, SynthSeg+), which are included as accuracy references, and the
CNN3D and Gaussian-blur baselines, which are compared quantitatively.
Broadening to additional imaging modalities is outside the scope of a 64 mT
nWBV feasibility study but is noted as a future direction.

**(c)** §III clarifies the output is a scalar and that image-quality comparison
is not applicable; SOTA morphometry and learning baselines retained; scope
statement added.

---

## Summary of new material added

1. Adapter-strategy ablation (head-only vs. LN+head vs. full fine-tune), same LOOCV protocol.
2. Multi-seed robustness (5 seeds) for MAE and ICC.
3. Simulation-parameter sensitivity (±20% SNR, B0, relaxation).
4. Simulation-fidelity validation (NCC + SNR/CNR vs. real 64 mT).
5. Honest reframing of the ViT-vs-CNN result.
6. Consolidated ICC discussion; reduced redundancy.
7. All figures regenerated at 300 dpi; calibration relabelled "overconfident".
