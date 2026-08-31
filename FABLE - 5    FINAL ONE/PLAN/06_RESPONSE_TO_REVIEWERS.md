# 06 — RESPONSE TO REVIEWERS (ready to paste into the IEEE template)

Manuscript Access-2026-28453 — *A Reproducible Feasibility Baseline for
Segmentation-Free nWBV Regression from 64mT Ultra-Low-Field MRI Using
Physics-Constrained Deep Learning*

Format follows `05_TEMPLATE_Response_to_Reviewers_IEEE.docx`: Reviewer #N, Concern #M /
Author response / Author action. All 28 concerns are covered. Edit IDs (M1…M28, W1…W5)
refer to the revision's change list; each maps to a highlighted location in the
resubmitted PDF (fill in final page numbers at assembly). Every number traces to
`PLAN/02_EVIDENCE_INVENTORY.md`. Bracketed ⚠ notes are assembly instructions — delete
them before upload.

---

## Preamble — author-initiated corrections (place before Reviewer 1)

In preparing this revision we audited every quantitative claim in the manuscript
against its source data. Beyond the reviewers' concerns, this audit found five errors
in the previous version, which we correct and disclose here rather than silently:
(1) the constant-mean baseline was misreported as 0.0126; the reproducible values are
0.0128 (leave-one-out) and 0.0122 (global mean), which makes the comparison *less*
favourable to our model, and we state it accordingly. (2) The "Wilcoxon p = 0.847"
for the physics-versus-blur pre-training comparison was computed as a paired test
across arms that used different test splits; it is withdrawn and replaced with the
valid unpaired test (p = 0.81). (3) Three uncertainty figures had been produced by a
plotting script using placeholder simulated intervals rather than the recorded MC
Dropout outputs, and one statistic derived from them was wrong: the width–error
correlation is r = −0.164 (p = 0.454), not r = +0.34 (p = 0.108). All three figures
are rebuilt from the recorded per-subject outputs; the miscalibration conclusion is
unchanged and in fact strengthened. (4) The CNN3D baseline's parameter count was
misstated as ≈4.1 M; the committed model definition counts 8.22 M, and the comparison
is no longer described as parameter-matched. (5) The claim that ds006557 is "the only"
public paired 64 mT/3T dataset was inaccurate and is corrected (see Reviewer 1,
Concern #3). None of these corrections alters a headline conclusion; each makes the
paper more conservative or more accurate, and all corrected passages are highlighted.

A sixth correction was found after that list was drafted. The previous version
reported inference latency as "47 ms on a standard GPU". We can produce no
measurement for that figure: it exists in our codebase only as a hardcoded constant,
and the development machine used throughout this work had no CUDA-capable GPU, so a
GPU benchmark could not have been produced on it. We have withdrawn the claim,
re-measured latency on CPU under a stated protocol (4.53 ms cold median, IQR 1.37, 100 repetitions, Intel Core
i7-8750H, 6 threads), and released the benchmark with the code (Concern #7 of Reviewer 3; M29). We report this in the same spirit as the
others: it was our error, it was not challenged by any reviewer, and correcting it
makes the deployment claim both smaller and verifiable.

A seventh change is an addition rather than a correction: external validation on a
second public cohort, which we had previously declined to perform, has now been run
and is reported in full (Reviewer 1, Concern #3). Its outcome is unfavourable to our
model and is presented as such.

Finally, in preparing this revision we audited the manuscript's internal
cross-referencing and found four systematic defects that we had not previously
noticed and that no reviewer raised explicitly, though we believe they contributed
to Reviewer 4's assessment that the material was not presented comprehensively.
Eleven section cross-references were written in Arabic form ("Section 5.9",
"Section 6.4") in a paper whose sections are numbered in Roman; two figure captions
had been transposed across a subsection boundary, so the failure-analysis figure
was captioned inside the uncertainty-quantification subsection and vice versa; one
table number was duplicated and another out of sequence; and one figure was never
cited in the text. All are corrected, with every cross-reference re-verified against
the content it points to rather than renumbered arithmetically.

---

# REVIEWER 1

## Reviewer #1, Concern #1

> *The main contribution should be reconsidered because the proposed model does not
> outperform the constant-mean baseline (MAE: 0.0134 vs. 0.0126). Please avoid
> overstating the model's effectiveness.*

**Author response:** We agree, and we go further than the reviewer asks. First a
correction: the baseline figure of 0.0126 was not reproducible from our data.
Recomputed from the committed per-subject values, the leave-one-out constant-mean
baseline is MAE = 0.0128 and the global mean 0.0122. Our model's 0.0134 is therefore
not merely comparable to a constant predictor — it is numerically above both, and the
revised manuscript says so wherever the comparison appears.

We accept the premise that MAE cannot carry the paper's claim in a cohort whose nWBV
range is 0.056 wide. The revision separates the two questions the previous version
conflated: *is the model more accurate than the cohort mean?* (no), and *is it
responding to its input at all, or is it a disguised constant predictor?* Permuting
subject labels 20,000 times and recomputing the inter-session ICC gives an observed
ICC = 0.6146 against a null mean of approximately zero, p = 0.0024: per-subject
predictions agree across two independent acquisitions far beyond chance, which a
constant predictor cannot produce. We report the unfavourable half in the same
paragraph: the reproducible signal does not track true nWBV (r = −0.119, p = 0.590),
and the model's prediction spread is roughly a quarter of the true between-subject
spread. The model is reproducible without being accurate in the biological direction.
The contribution is stated as exactly that — a feasibility baseline with its
discriminative evidence and its limits in the same breath.

**Author action:** Baseline corrected to 0.0128/0.0122 and reframed as "does not
outperform" in the Abstract (M1), Contributions (M3), Results (M10 environs), and
Conclusion (M22). New Results subsection *Input-Dependence (Permutation Test)* (M10)
with supporting figure (Fig. 12). A "What This Study Does Not Claim" subsection added
to the Discussion (M18).

## Reviewer #1, Concern #2

> *The study uses only 23 real 64mT MRI subjects, which is insufficient for strong
> conclusions. Please clearly discuss the impact of the small sample size.*

**Author response:** We agree, and the revision makes no strong conclusion. n = 23 is
the largest public paired 64 mT/3T cohort available (see Concern #3), so the sample
size reflects the state of public ultra-low-field data. We address the impact three
ways: bootstrap intervals on every estimate (the ICC's interval [0.236–0.866] is
stated as precluding a definitive reliability classification); a five-seed repetition
of the full protocol (MAE 0.0130 ± 0.0004, ICC 0.644 ± 0.058), showing the headline
values are not seed artifacts; and split-conformal calibration, which quantifies the
sample-size limit rather than asserting it — the calibrated 95 % interval width
(0.066 nWBV) exceeds the cohort's entire nWBV range (0.056), demonstrating directly
that individual-level nWBV is not resolvable here.

**Author action:** Expanded sample-size limitation (M19); multi-seed subsection and
figure (M12, Fig. 13); conformal subsection including the width-exceeds-range
statement (M11, Fig. 21).

## Reviewer #1, Concern #3

> *External validation on an independent dataset is required to demonstrate model
> generalization.*

**Author response:** We accept this requirement, and in this revision we satisfy it.

In addressing the concern we first found and corrected an inaccuracy in the previous
version, which stated that ds006557 is the only public paired 64 mT/3T dataset. That
was wrong: a second public paired cohort exists (van den Broek et al. [42], 11 paired
subjects), and the manuscript now cites it.

We then used it. We derived nWBV ground truth for that cohort with the same FastSurfer
`--seg_only` pipeline and the same nWBV definition (BrainSeg/Mask) applied to OASIS-1
and ds006557, so the external labels are directly comparable rather than proxy
measures; 10 of the 11 paired subjects yielded complete segmentation. No subject was
seen in any training, pre-training or adaptation stage. The experiment and its
per-subject outputs are released as `experiments/zenodo_external_validation/`.

The outcome does not favour our model, and we report it as the primary external
finding rather than as a caveat. The 64 mT-adapted model attains MAE = 0.0731
(r = 0.255, p = 0.477); the unadapted OASIS-1 model attains MAE = 0.0327 (r = 0.264,
p = 0.461). Both are far worse than that cohort's own constant-mean baseline
(MAE = 0.0089), and the adapted model is more than twice as far from truth as the
unadapted one — our cross-session adapter transfers negatively.

The cause is specific and measurable, and we believe it is the most useful result in
this revision. The two cohorts' nWBV distributions do not overlap: ds006557 spans
[0.752, 0.808] and the external cohort spans [0.836, 0.869]. The adapted model's
predictions span 0.008 against a true spread of 0.033 and centre on 0.7824 — within
0.002 of the adaptation cohort's mean of 0.7841 — with every prediction below every
ground-truth value. The model returns its training mean and does not extrapolate.
This is the same range-compressed behaviour we report internally in Section V-E
(r = −0.119, p = 0.590), now confirmed at an independent site on an independent
scanner.

We are careful not to over-explain the distribution shift. The external cohort is
younger (mean 30 years versus 45.1), and age accounts for part of the offset — but
only part: the age–nWBV slope observed within ds006557 accounts for about 0.012 of
the 0.071 difference, and the steeper slope implied by the OASIS-1 age span for about
0.043. A residual remains that age does not explain and that may reflect cohort or
segmentation-pipeline differences we cannot resolve with the available data. We state
this in Section V-Q rather than presenting age as a complete account. The finding
itself does not depend on the cause: the adapted model's predictions vary by 0.008
against a true spread of 0.033, so it does not track anatomy in this cohort however
the distribution shift arose.

The reviewer's requirement has therefore produced a bound on the paper's claim rather
than a confirmation of it: the protocol and adaptation procedure are reproducible and
input-dependent, but the resulting estimator is not transportable to cohorts whose
nWBV distribution differs from the adaptation set, and the revision says so in the
abstract, in Section V-Q, and in the Limitations.

**Author action:** New Section V-Q "External validation" with the full result and its
diagnosis (M30), and new Fig. 24 showing the predicted-versus-true scatter alongside
the non-overlapping label ranges. "Only public dataset" claim corrected and the second
cohort cited in Limitations (M19). External validation removed from Future Works,
since it is now a result; item (v) replaced by the constraint the validation exposed
(M20). Abstract gains an external-validation sentence (M1). Subject counts
standardised as 11 paired / 10 analysed throughout (M31). No claim of successful
generalisation appears anywhere in the revision.

## Reviewer #1, Concern #4

> *The physics-based simulation should be quantitatively validated against real 64mT
> MRI data using image similarity or distribution analysis metrics.*

**Author response:** We agree, we ran the comparison on the n = 23 paired subjects,
and the outcome does not favour our simulator — we report it as measured. Against a
Gaussian-blur-plus-histogram-matching baseline, the physics simulator attains
NCC = 0.497 ± 0.046 versus 0.516 ± 0.056 for blur (paired Wilcoxon p = 0.02 in blur's
favour) and lower SSIM (≈0.07–0.09 vs ≈0.14–0.22). Its SNR error is far larger
(p < 0.001), because the simulator deliberately targets conservative raw-acquisition
SNR (≈33) while the scanner's reconstructed output reaches ≈320–350 through averaging
and compressed sensing we do not model. The previous version called the fidelity
difference "statistically indistinguishable"; that was wrong, and wrong in the
direction that flattered us. The corrected statement is that the physics simulator is
not superior on image fidelity; its value is assessed on the downstream task, where
physics and blur are statistically equivalent (MAE 0.0146 vs 0.0148, unpaired
Mann-Whitney p = 0.81).

**Author action:** Fidelity-validation paragraph and metrics added to Methods (M9);
contribution 2 rewritten to carry the corrected result (M3); the invalid paired
statistics withdrawn throughout (M8, M21, M22).

## Reviewer #1, Concern #5

> *More ablation studies are needed to evaluate the contribution of each component:
> physics simulation, self-supervised pretraining, ViT architecture, LayerNorm + head
> adaptation.*

**Author response:** All four components are now ablated, and two of the four results
favour alternatives to our configuration — we report both rather than omitting them.
**Physics simulation:** no significant downstream advantage over blur (Mann-Whitney
p = 0.81) and not superior on fidelity (Wilcoxon p = 0.02 against us); a ±20 %
parameter-sensitivity sweep bounds the effect of any single simulation parameter at
|ΔMAE| ≤ 0.0062. **Self-supervised pretraining:** all four objectives the reviewers
named were run under the identical protocol — DINO MAE 0.0270 (r 0.903), masked
autoencoding 0.0319 (0.756), SimMIM 0.0538 (0.907), contrastive 0.0589 (0.744) —
against denoising 0.0584 (0.722); three of four beat our objective on OASIS-1, and the
strongest (DINO) was additionally carried through the full 64 mT LOOCV, where it is
worse (see Reviewer 2, Concern #3). **Architecture:** CNN3D is significantly better
than ViT3D on OASIS-1 (paired Wilcoxon p = 0.0037); Swin-UNETR (r 0.964) and UNETR
(r 0.923) are better still (Reviewer 2, Concern #2). **Adaptation:** a four-way
ablation (head-only / LayerNorm+head / LoRA / full fine-tuning) under the identical
LOOCV shows all four confidence intervals overlap; no strategy is significantly
superior (Reviewer 2, Concern #4).

**Author action:** New/updated Results subsections and figures: SSL comparators and
64 mT transfer (M13, Fig. 17), architecture comparators (M14, Fig. 17), adapter ablation
(M15, Fig. 18), physics-vs-blur corrected (M8, Fig. 6), sensitivity sweep (M23,
Fig. 2); statistical-comparisons table (M21).

## Reviewer #1, Concern #6

> *The dementia evaluation is limited because it relies on simulated data and very few
> pathological subjects. Please avoid strong clinical interpretations.*

**Author response:** We agree entirely. The dementia analysis is a failure
characterisation, not a capability claim: the CDR = 1.0 group contains two subjects,
the atrophy is physics-simulated, and the result is labelled directional wherever it
appears. The result itself is adverse — MAE rises to 0.157 against 0.016 for the
unlabelled healthy group, with predictions regressing toward the healthy-adult mean —
and it is presented as the boundary of the validated scope. The manuscript states the
model is not suitable for dementia screening, pathological morphometry, or
longitudinal clinical monitoring. We also corrected the reported effect size: the
value previously labelled Cohen's d (1.368) was in fact closest to Glass's delta;
the pooled-SD Cohen's d is 1.463 (interpretation unchanged: large effect, interpret
cautiously at n = 2).

**Author action:** Every dementia statement audited to carry "simulated, n = 2,
directional" (M17); explicit non-suitability sentence added (M17); effect size
corrected and correctly labelled (M7); scope statements bounded to healthy adults
(M18, M27).

## Reviewer #1, Concern #7

> *The uncertainty estimation results show severe miscalibration (4.3% coverage for
> 95% intervals). Please improve this analysis or clearly state its limitations.*

**Author response:** We have done both, and we additionally disclose a correction. In
auditing this analysis we found that the uncertainty figures in the previous version
had been produced by a plotting script that drew interval placeholders from a random
generator rather than from the recorded MC Dropout outputs, and that one derived
statistic was consequently wrong: interval width is not positively correlated with
error (previously r = +0.34, p = 0.108) but weakly negatively (r = −0.164,
p = 0.454). We have rebuilt all uncertainty figures from the recorded per-subject
outputs, removed a sentence that argued from the incorrect positive sign, and — since
no OASIS-1 coverage had actually been measured — ran real MC Dropout on the OASIS-1
test set (N = 100 passes): coverage 23.7 % (9/38), mean width 0.0284. The real-64 mT
result stands as measured: 4.3 % coverage at nominal 95 %, mean width 0.0286. The
miscalibration conclusion is unchanged and strengthened.

As the improvement, we added leave-one-out split-conformal prediction on the
subject-independent LOOCV residuals, which restores near-nominal coverage: 91.3 % at
nominal 90 % (width 0.0571) and 95.7 % at nominal 95 % (width 0.0664). The previous
version stated conformal calibration was infeasible in this cohort; that was
incorrect and has been replaced. We state the cost plainly: calibrated intervals are
roughly twice as wide, and the calibrated 95 % width (0.066) exceeds the cohort's
entire nWBV range (0.056) — a group-level tool, not an individual-level clinical
instrument.

**Author action:** Corrected statistics and deleted the sign-based sentence (M5, M6);
per-dataset coverage reported (M5, M16; Figs. 14, 15, 17 → revised Figs. 20, 21, 23);
split-conformal subsection added, replacing the "not feasible" passage (M11, Fig. 21);
figures rebuilt from recorded data (Figure Plan; MANIFEST).

## Reviewer #1, Concern #8

> *Statistical comparisons between models and experimental settings should be
> strengthened with appropriate significance tests.*

**Author response:** A systematic significance-test pass now covers every model and
condition comparison, summarised in a new table: T1 ViT3D vs CNN3D on OASIS-1 (paired
Wilcoxon, p = 0.0037, CNN better); T2 physics vs blur on image fidelity (paired
Wilcoxon, NCC p = 0.02, blur better; SNR error p < 0.001); T3 physics vs blur on the
downstream task (unpaired Mann-Whitney, p = 0.81, n.s.); T4 denoising vs DINO
pretraining on the real 64 mT LOOCV (paired Wilcoxon p = 0.070, n.s.; paired bootstrap
ΔMAE −0.0086 [−0.017, −0.001] favouring denoising); and the two permutation tests, P1
inter-session ICC (p = 0.0024) and P2 prediction-versus-truth (p = 0.590, n.s.). Two
comparisons favour a baseline over our configuration and one permutation result is a
null against our model; all are reported as measured. We also withdrew a test that
should never have been run: the previously cited "Wilcoxon p = 0.847" paired across
arms with different test splits, replaced by the valid unpaired T3.

**Author action:** Statistical-comparisons table added (M21); tests threaded into the
sections where each comparison lives (M8, M13, M14); invalid paired statistics
withdrawn everywhere (M8, M3, M22).

## Reviewer #1, Concern #9

> *The robustness of the model should be evaluated using multiple random seeds or
> repeated experiments.*

**Author response:** The full cross-session LOOCV protocol was repeated across five
seeds (42, 1, 7, 123, 2024): MAE = 0.0130 ± 0.0004 and ICC(3,1) = 0.644 ± 0.058, with
the headline values (0.0134, 0.615) inside the seed spread. The two quantities are
not equally stable and we say so: MAE varies by 0.0004, ICC by 0.058, with one seed
at ICC = 0.563 — the reliability estimate is the more seed-sensitive quantity and
should be read together with its wide bootstrap interval.

**Author action:** Multi-seed subsection and figure added, including the seed-level
caveat (M12, Fig. 13).

## Reviewer #1, Concern #10

> *The simulation parameters (noise, B0 distortion, relaxation values) require clearer
> justification and sensitivity analysis.*

**Author response:** On justification: the T1/T2 relaxation values are taken from
published low-field relaxometry rather than fitted to our data, and the simulated SNR
is deliberately conservative against the scanner's reconstructed effective SNR (the
un-modelled averaging/compressed-sensing chain is stated as a limitation). On
sensitivity: perturbing each parameter by ±20 % changes downstream MAE by at most
|ΔMAE| = 0.0062 (from a 20 % reduction in relaxation values); SNR and B0
perturbations change MAE by 0.0005 or less. The pipeline is not finely tuned to a
parameter choice, and the one influential parameter group (relaxation) is the one
sourced from published measurements.

**Author action:** Sensitivity paragraph and figure added to Methods (M23, Fig. 2);
sourcing and conservative-SNR rationale stated alongside (M9).

## Reviewer #1, Concern #11

> *The manuscript contains repeated discussions of MAE, ICC, and limitations. Please
> reduce redundancy and improve readability.*

**Author response:** We accept the concern and have acted on it, but we should be
straightforward about the outcome: this revision is longer than the version the
reviewer read, not shorter. Concerns #1, #3, #4, #5 and #7–#10 from this reviewer,
together with Concerns #2 and #3 from Reviewer 2, required six new experiments and
seven new results subsections. The manuscript has grown from approximately 8,500 to
12,100 words.

We therefore reduced redundancy in kind rather than in volume, and we treated the
concern as being about readability rather than page count.

The abstract was the worst case. It had stated the headline MAE three times and the
constant-mean comparison three times within a single paragraph; it is now cut by
44 %, from 456 words to 257, which also brings it back within the IEEE Access
guideline it had exceeded. Each quantity is stated once there.

The scope subsection added at Reviewer 2's request originally restated five results
given elsewhere in the paper; it now cites the sections that contain the evidence
instead. A redundant inter-session ICC sentence was removed from the Conclusion,
where the permutation result immediately preceding it makes the same point more
precisely. Discussion passages now interpret the reliability estimate rather than
repeating its numeric value, which is given in full in the Results subsection where
it is derived.

Where a quantity still appears repeatedly, most occurrences are not restatements. Of
the sentences mentioning the inter-session ICC, the majority report a different value
for a different purpose — the multi-seed spread (0.644 ± 0.058), the DINO comparator
(0.523), the permutation null, the methods definition. The same holds for the
headline MAE: most of its occurrences are table cells or contrasts against another
condition rather than assertions of the result.

Finally, the character of the repetition has changed. In the previous version the
headline MAE was repeatedly asserted on its own; every occurrence in this revision
carries the constant-mean comparison that qualifies it. We think the bare restatement
was part of what made the repetition read as insistent, and that has been removed
even where the mention itself remains.

We recognise this does not satisfy the request as literally written. We judged that
deleting material the reviewers had asked for would be the wrong trade.

**Author action:** Abstract rewritten and cut 44 % (M24 Part 1); scope subsection
converted from restatement to cross-references (M24 Part 2); redundant ICC sentence
deleted from the Conclusion (M24 Part 3); numeric restatement removed from the
Discussion's reliability passage; every assertion of the headline MAE verified to
carry its constant-mean qualifier (M24, Fix A).

---

# REVIEWER 2

## Reviewer #2, Concern #1

> *One of the most concerning findings is that the proposed ViT performs worse than
> the CNN baseline on the OASIS dataset. CNN3D achieves Pearson correlation of 0.877
> and MAE of 0.024, whereas ViT3D achieves only 0.722 correlation and MAE of 0.058.
> The authors attribute the improvement on the 64 mT dataset to the narrow value range
> rather than to the transformer architecture itself. Therefore, the experimental
> results do not convincingly demonstrate that Vision Transformers provide better
> feature learning than CNNs. Instead, the reported gains appear to result primarily
> from the adaptation procedure rather than from the transformer architecture.*

**Author response:** The reviewer is right, and the paper has been rewritten to say
so. On OASIS-1, CNN3D significantly outperforms ViT3D (MAE 0.0243 vs 0.0584, paired
Wilcoxon p = 0.0037, n = 38); the comparison remains in Table II exactly as before,
now with the significance test attached. We also accept the sharper half of the
criticism: the measured gains on 64 mT come from the adaptation procedure and the
narrow-range regime, not from the transformer architecture, and the revision adopts
that as its own statement. We removed the claims that implied otherwise — the Related
Work assertion that global self-attention is "architecturally well-suited" to this
task, and the Results claim that self-attention "produces more stable predictions",
which asserted an untested mechanism; the descriptive replacement is that ViT3D's
smaller output variance yields lower absolute error in the narrow-range regime. One
further correction, disclosed rather than hidden: CNN3D's parameter count was
misstated as ≈4.1 M; the committed definition counts 8.22 M, so the comparison was
never parameter-matched and is no longer described as such. ViT3D remains the system
studied because it is the only model carried through the complete Stage-3 protocol —
a tractability constraint now stated as such, not a design virtue (Concern #2). No
claim of transformer superiority survives anywhere in the paper.

**Author action:** T1 added beside Table II (M21); "architecturally well-suited" and
mechanism claims removed (M2, M18); Discussion §B retitled *Architecture Choice: A
Tractability Trade-off* and rewritten (M18); CNN3D parameter count corrected (M4);
contribution list rewritten to report the CNN advantage as a finding (M3);
"What This Study Does Not Claim" subsection states it once more, plainly (M18).

## Reviewer #2, Concern #2

> *The proposed ViT3D architecture is a very basic implementation consisting of only
> four transformer encoder layers with 4.23M parameters, yet the paper does not
> explain why this architecture was selected over more advanced medical Vision
> Transformer models such as Swin UNETR, UNETR, ViT-V-Net, or hierarchical
> transformers.*

**Author response:** The reviewer is right that the previous version did not justify
the choice. We supplied the justification and, where feasible, the comparison itself.
Two of the named architectures were evaluated under the identical OASIS-1 protocol:
Swin-UNETR (62.2 M parameters) attains r = 0.964, MAE = 0.0148; UNETR (92.7 M)
attains r = 0.923, MAE = 0.0198 (CPU inference 267 ms). Both substantially outperform
ViT3D on high-field data, and we report this plainly with no claim that ViT3D is the
best architecture. ViT-V-Net was not evaluated; we name the omission rather than
imply coverage. The reason ViT3D is the system carried through the paper is a
resource constraint, now stated as such: the complete Stage-3 protocol (23-fold
cross-session LOOCV with per-fold adaptation, five seeds, permutation testing,
conformal calibration, failure analysis) runs in roughly 20 hours for the
4.23 M-parameter model on our CPU-only hardware, against an estimated ~13 days per
arm for Swin-UNETR; carrying a 62–93 M-parameter model through it was not feasible
within this study. The comparators were evaluated on OASIS-1 only, and this
evaluation-depth asymmetry is stated as a limitation; carrying them through the full
64 mT protocol is named as future work. We note that the related transfer question —
whether high-field representational advantage survives the domain gap — is answered
for pretraining objectives under Concern #3, where the strongest high-field objective
was carried through the full 64 mT protocol and performed worse.

**Author action:** Architecture-comparators subsection with both models and figure
(M14, Fig. 17); architecture-selection-rationale paragraph in Methods (M25);
evaluation-depth limitation and future-work item (M14, M20). [⚠ assembly note: the
earlier draft's frozen-encoder probe sentence is removed everywhere per W1 and must
not appear in this letter either.]

## Reviewer #2, Concern #3

> *The Stage-1 pretraining uses a denoising autoencoder to reconstruct high-field MRI
> from simulated low-field MRI. However, the paper provides no evidence that this
> reconstruction objective actually learns representations useful for nWBV
> regression. Modern Vision Transformer pretraining methods, such as Masked
> Autoencoders (MAE), DINO, SimMIM, or contrastive learning, have shown superior
> feature learning for downstream medical imaging tasks. The paper never compares its
> denoising objective against these stronger self-supervised approaches, making the
> claimed advantage of the proposed pretraining strategy unconvincing.*

**Author response:** We ran all four named methods under the identical protocol
(ViT3D encoder, physics-simulated IXI input, 25 pretraining epochs, same OASIS-1
seed-42 fine-tuning and test split), and the outcome does not favour our objective on
high-field data: DINO attains MAE 0.0270 (r 0.903), masked autoencoding 0.0319
(0.756), SimMIM 0.0538 (0.907), contrastive learning 0.0589 (0.744), against
denoising at 0.0584 (0.722). Three of the four outperform the denoising objective,
which ranks fourth of five. We report this plainly and withdraw any implication of a
demonstrated pretraining advantage.

The scientifically informative result is what happened next: the strongest high-field
objective, DINO, was carried through the identical cross-session 64 mT LOOCV (same
adapter, same folds, same session assignment) and performed worse on the target
hardware — MAE 0.0220, bias +0.0085, ICC 0.523, with only 10/23 subjects below the
0.020 reference threshold, against 0.0134, −0.0022, 0.615 and 19/23 for denoising
(paired Wilcoxon p = 0.070, not significant; paired bootstrap ΔMAE −0.0086
[−0.017, −0.001]). High-field representation quality is not predictive of low-field
performance in this setting. The denoising objective is retained on three grounds,
none a superiority claim: its pretext task is the deployment domain gap itself (the
a-priori design rationale); it is the only objective carried through the complete
protocol; and the one comparator tested on target hardware was the strongest on
high-field data yet worse on target. The comparison's limits are stated: only DINO
was transferred; the DINO run is single-seed; the 25-epoch budget was not tuned per
objective.

**Author action:** SSL comparator table (all five objectives), the
*Pretraining-objective transfer to real 64 mT* subsection with the retention rationale
and limitations, and abstract/conclusion updates (M13, M1, M22, Fig. 17); T4 row in
the statistical table (M21); the earlier SimMIM-only paragraph replaced (W2).

## Reviewer #2, Concern #4

> *The proposed adaptation updates only the final LayerNorm and regression head (769
> parameters), but this design choice is based on intuition rather than experimental
> evidence. The paper states that comparisons with head-only adaptation, full
> fine-tuning, LoRA, adapters, prompt tuning, and other parameter-efficient
> transformer adaptation methods are left for future work. Since Vision Transformers
> are sensitive to adaptation strategy under domain shift, it is difficult to conclude
> that the proposed LayerNorm+head adaptation is the most effective solution.*

**Author response:** The reviewer is correct, and we no longer conclude that. The
four-way ablation is now run under the identical cross-session LOOCV, including the
LoRA arm the reviewer named: head-only (257 parameters) MAE 0.0133 [0.0099–0.0170];
LayerNorm+head (769) 0.0137 [0.0104–0.0172]; LoRA r = 4 (41,217) 0.0128
[0.0099–0.0160]; full fine-tuning (4,225,537) 0.0123 [0.0091–0.0158]. All four
confidence intervals overlap; no strategy is significantly superior at n = 23, and
LayerNorm+head is not even the lowest point estimate. The honest finding is that
adaptation strategy is not a significant performance lever in this regime;
LayerNorm+head is retained for parameter economy and training stability, and the
Methods rationale is softened accordingly (its mechanistic expectations were not
borne out as performance differences). For transparency: the headline run reports
0.0134 for LayerNorm+head while the ablation arm reports 0.0137 — independent
training runs under the same protocol and seed, within each other's intervals, both
reported. We also caution against over-reading LoRA's 0.0128, which coincides with
the leave-one-out constant-mean baseline (Concern #1 of Reviewer 1).

**Author action:** Adapter Strategy Ablation subsection with four-way table, figure
and run-provenance footnote (M15, Fig. 18); Methods rationale rewritten from
optimality to economy, with the future-work deferral sentence replaced by the
completed result (M26).

---

# REVIEWER 3

## Reviewer #3, Concern #1

> *The manuscript addresses an important and timely problem in point-of-care
> neuroimaging. Direct nWBV regression from 64mT ultra-low-field MRI without
> segmentation or super-resolution is a practically relevant research direction,
> especially for low-resource or bedside environments where conventional high-field
> MRI morphometry pipelines are difficult to deploy.*

**Author response:** We thank the reviewer; this matches the motivation of the work.

**Author action:** None required.

## Reviewer #3, Concern #2

> *The feasibility-oriented framing is appropriate and should be maintained
> consistently. The study is strongest when presented as a reproducible feasibility
> baseline rather than as a clinically deployable nWBV estimation system. The title,
> abstract, discussion, and conclusion should consistently preserve this distinction
> and avoid implying clinical readiness.*

**Author response:** We agree and have preserved the framing throughout, including in
the substantial material added for other reviewers, which was audited to the same
standard. The title is unchanged ("A Reproducible Feasibility Baseline…"). No
clinical-readiness or deployability language appears; the measured 4.53 ms CPU latency
is bounded as computational feasibility, with no GPU claimed and edge devices not
benchmarked. The
revision goes one step further and adds a "What This Study Does Not Claim" subsection
that enumerates, in one place, every claim the paper deliberately does not make.

**Author action:** Framing audit of title/abstract/discussion/conclusion after all
edits (M27); scope subsection added (M18); every new result bounded in its own
paragraph (M10–M15).

## Reviewer #3, Concern #3

> *The leakage-free subject-independent cross-session LOOCV protocol is a
> methodological strength. Training the lightweight adapter on one session from
> non-held-out subjects and testing on the second session of the held-out subject is
> an appropriate design for reducing subject/session leakage in a very small
> real-hardware cohort. This protocol is one of the strongest aspects of the
> manuscript.*

**Author response:** We thank the reviewer. The protocol is retained unchanged and is
now also the engine of the new input-dependence evidence (the permutation test uses
its paired-session predictions) and the conformal calibration (its subject-independent
residuals).

**Author action:** Protocol description retained and emphasised; no substantive
change.

## Reviewer #3, Concern #4

> *The main technical concern is that the primary real-hardware MAE is comparable to a
> constant-mean LOOCV baseline. Although the adapted ViT3D achieves MAE below the
> predefined reference threshold, the result does not clearly demonstrate superiority
> over a trivial mean predictor. The authors should better explain what evidence
> supports anatomy-dependent learning rather than range-compressed mean regression
> within a narrow healthy cohort.*

**Author response:** This is the central question, and we answer it with a test built
for it — reporting both halves of the answer. The permutation test (20,000
permutations) shows the model is input-dependent: observed inter-session
ICC = 0.6146 against a null of ≈0, p = 0.0024 — behaviour a constant or
range-compressed mean predictor cannot produce. And it shows the limit: the
reproducible per-subject signal does not track true nWBV (r = −0.119, p = 0.590),
with a prediction spread roughly a quarter of the true spread. So the honest answer
to the reviewer's question is: input-dependence, yes, demonstrated; anatomy-dependent
learning of nWBV, not demonstrated at this cohort's 0.056 range — and the paper now
says exactly that, in the same paragraph and figure. The corrected baseline
(leave-one-out 0.0128, global 0.0122; our 0.0134 above both) is stated wherever the
comparison appears, and MAE is no longer presented as evidence of anything beyond
feasibility.

**Author action:** *Input-Dependence (Permutation Test)* subsection reporting both
results with figure (M10, Fig. 12); baseline corrected throughout (M1, M3, M22);
abstract carries the two-sided statement (M1).

## Reviewer #3, Concern #5

> *Additional evidence of input-dependent learning would strengthen the manuscript.
> The reported inter-session ICC suggests some model-dependent behavior, but the
> confidence interval is wide due to the small sample size. Analyses such as adapter
> ablation, head-only versus LayerNorm+head comparison, permutation testing, or
> validation on a wider real-hardware nWBV range would make the technical claim more
> convincing.*

**Author response:** Three of the four suggested analyses are added: the permutation
test (p = 0.0024; Concern #4), the adapter ablation including the head-only versus
LayerNorm+head comparison (0.0133 vs 0.0137, intervals overlapping; Reviewer 2,
Concern #4), and multi-seed repetition (MAE 0.0130 ± 0.0004, ICC 0.644 ± 0.058). The
fourth — validation on a wider real-hardware nWBV range — cannot be satisfied with
existing data: both public paired 64 mT cohorts (ds006557 and van den Broek et al.
[42]) contain only healthy adults. We state this as a limitation and identify
prospective acquisition spanning a wider nWBV range as the necessary next step,
rather than working around it.

**Author action:** M10 (permutation), M15 (adapter), M12 (multi-seed); wider-range
limitation stated in Limitations and Future Works (M19, M20).

## Reviewer #3, Concern #6

> *The clinical generalizability should be stated more conservatively. The real 64mT
> cohort includes only healthy adults, while nWBV is clinically motivated as a
> neurodegeneration biomarker. The simulated dementia/atrophy analysis shows large
> errors and regression toward the healthy-adult mean; therefore, the current model
> should not be interpreted as suitable for dementia screening, pathological
> morphometry, or longitudinal clinical monitoring.*

**Author response:** We agree without qualification and adopt the reviewer's wording.
The validated scope is healthy adults only. The simulated-dementia analysis is
presented as failure characterisation: with n = 2 at CDR = 1.0 and simulated rather
than acquired atrophy, MAE rises to 0.157 against 0.016 for the unlabelled healthy
group, and predictions regress toward the healthy-adult mean — exactly the behaviour
the reviewer anticipates. The manuscript states verbatim that the model is not
suitable for dementia screening, pathological morphometry, or longitudinal clinical
monitoring.

**Author action:** Scope bounded to healthy adults in Discussion and Limitations;
non-suitability sentence added; every dementia result labelled simulated / n = 2 /
directional (M17, M18, M27).

## Reviewer #3, Concern #7

> *The physics simulation, uncertainty estimation, and deployment claims should be
> carefully bounded. The physics-constrained simulation pipeline is useful as a
> reproducible pre-training strategy, but its advantage over Gaussian-blur degradation
> is small and statistically non-significant. Similarly, the MC Dropout intervals are
> severely miscalibrated under domain shift and should not be used for
> individual-level clinical decisions. The reported 47 ms inference time supports
> computational feasibility, but edge-device deployment and prospective clinical
> readiness have not yet been demonstrated.*

**Author response:** All three are bounded, and in one case the correction goes
further than the reviewer indicates. **Physics simulation:** on the downstream task
the reviewer's description is right (p = 0.81, n.s.); on image fidelity the true
position is worse for us — blur is significantly better (NCC p = 0.02; SNR error
p < 0.001) — and the previous "statistically indistinguishable" wording is corrected.
The simulator is presented as a reproducible recipe, not an improvement.
**Uncertainty:** MC Dropout is retained as a transparent negative (4.3 % coverage on
real 64 mT; 23.7 % on OASIS-1, now actually measured), split-conformal calibration is
added as the remedy (91.3 % / 95.7 % coverage at ~2× the width), and no interval from
this paper is endorsed for individual-level decisions. **Deployment:** the latency claim is
withdrawn and re-measured. The reviewer quotes 47 ms, which we can no longer
substantiate: it appears in our codebase only as a hardcoded constant, and the
machine used throughout this work had no CUDA-capable GPU. Measured on CPU under a
stated protocol, a single forward pass takes 4.53 ms; this is described strictly as
computational feasibility, no GPU is claimed, and edge-device
deployment and prospective clinical readiness are stated as not demonstrated.

**Author action:** Fidelity correction (M9); contribution 2 and conclusion updated
(M3, M22); conformal subsection (M11); per-dataset MC Dropout coverage (M5, M16);
deployment language bounded (M18, M27).

---

# REVIEWER 4

## Reviewer #4, Concern #1

> *The manuscript is written in sufficiently clear and understandable English, making
> the presented work generally easy to follow.*

**Author response:** We thank the reviewer, and we have held the added material to the
same standard, supported by the readability consolidation performed for Reviewer 1,
Concern #11.

**Author action:** None beyond M24.

## Reviewer #4, Concern #2

> *The proposed methodology is described with an adequate level of clarity.*

**Author response:** We thank the reviewer. The methodology description is retained,
with additions where other reviewers requested detail (architecture selection
rationale, M25; simulation parameter justification, M23).

**Author action:** None beyond the noted additions.

## Reviewer #4, Concern #3

> *However, the blurred figures included in the manuscript limit the effective
> presentation and interpretation of the experimental results.*

**Author response:** We accept this; the cause was raster figures degraded by
word-processor image compression. The figure set has been rebuilt: every figure is
regenerated from its source result files at 600 dpi, and Figures 2–23 are inserted as
resolution-independent vector graphics (SVG), which cannot blur at any zoom or print
size; Figure 1 (MRI slice imagery) is inserted as a 600 dpi raster with compression
disabled. Six new analysis figures are added in response to other reviewers
(permutation test, conformal coverage, adapter ablation, architecture/pretraining
comparators, simulation sensitivity, multi-seed stability). During the rebuild we
also audited every figure against its data source and corrected three uncertainty
panels (disclosed under Reviewer 1, Concern #7).

**Author action:** Complete figure set regenerated and re-embedded per the figure
plan (07_FIGURE_PLAN; MANIFEST); captions carrying corrected statistics (M16, M6).
[⚠ assembly gate: confirm all 23 figures are actually placed and the export is sharp
before this block is uploaded — the claim must be true of the uploaded file.]

## Reviewer #4, Concern #4

> *Furthermore, the performance of the proposed method should be compared with
> existing state-of-the-art approaches through qualitative image-based analysis.*

**Author response:** We respectfully argue this does not apply to the task, and we
have made the reason visible in the manuscript itself rather than only here. The
model outputs a single scalar (nWBV) per scan; it performs no image synthesis or
super-resolution, so there is no output image on which a qualitative comparison
against reconstruction or enhancement methods could be made. The image-level
evaluation that does exist in this study is reported for the one image-producing
component, the physics simulator: a qualitative side-by-side of 3T, simulated 64 mT
and real 64 mT slices (Fig. 1) and a quantitative fidelity comparison against an
established degradation baseline, with paired significance tests — a comparison whose
outcome favours the baseline and is reported as measured. For the model itself, the
appropriate state-of-the-art comparison is quantitative, and it is substantially
expanded in this revision: SynthSeg+ as the segmentation-based accuracy ceiling
(r = 0.918, MAE = 0.005), CNN3D, Swin-UNETR and UNETR, with significance testing.

**Author action:** Scope-of-comparative-evaluation subsection added to the Discussion
(M28); fidelity analysis added to Methods (M9); expanded quantitative comparison
(M14, M21).

## Reviewer #4, Concern #5

> *To provide a more comprehensive evaluation of the model's performance, the
> comparative study should include a broader range of medical image modalities.*

**Author response:** This study is intentionally single-modality, and we have stated
the scope explicitly rather than extending it. The physics simulator, the
3T-to-64 mT domain-adaptation protocol and the FreeSurfer-derived ground truth are
all specific to 64 mT brain MRI; applying the method to other modalities would
require a different simulator, a different domain-shift model and different ground
truth — that is, it would answer a different scientific question from the feasibility
question posed here. The constraint is also practical: evaluation requires paired
low-field/high-field acquisitions of the same subjects, which exist publicly for very
few cohorts. We agree the direction is valuable and name multi-modality evaluation as
future work.

**Author action:** Single-modality scope stated in the Discussion scope subsection
and Limitations (M28, M19); future-work item added (M20).

## Reviewer #4, Concern #6

> *In conclusion, the proposed method is not recommended for acceptance in its current
> form. A more comprehensive evaluation is required, including both quantitative and
> qualitative comparisons with existing state-of-the-art methods across a broader
> range of image types, with particular emphasis on medical imaging applications. In
> addition, image quality comparisons with established approaches should be provided
> to more convincingly demonstrate the effectiveness, robustness, and generalizability
> of the proposed method.*

**Author response:** We have addressed every actionable component and are explicit
about the parts that fall outside the paper's scope. Addressed: figure quality
(600 dpi + vector, Concern #3); the quantitative state-of-the-art comparison, now
covering SynthSeg+, CNN3D, Swin-UNETR, UNETR and four self-supervised pretraining
objectives with significance testing; an image-quality comparison with an established
approach, applied to the study's one image-generating stage (physics simulation vs
Gaussian-blur baseline, NCC/SSIM/SNR with paired tests); and robustness evaluation
via permutation testing, conformal calibration, multi-seed repetition and parameter
sensitivity. Out of scope, with the argument now printed in the manuscript:
qualitative image comparison of a scalar-output model (Concern #4) and
broader-modality evaluation (Concern #5). We are equally explicit that several added
results do not favour our own configuration — CNN3D over ViT3D on OASIS-1, blur over
the physics simulator on fidelity, three of four SSL objectives over our denoising
objective — and we report them because the contribution is the reproducible protocol
and its transparent failure characterisation, not the optimality of any component.

**Author action:** M9, M14, M21 (added comparisons and tests); M28 (scope argument in
the body); 07_FIGURE_PLAN (figure quality); M18 (transparent-negatives framing).

---

## Cross-check table: concern → edit IDs → evidence source

| Concern | Edit IDs | Evidence (file) |
|---|---|---|
| R1.1 | M1, M3, M10, M18, M22, **M33** | `permutation_test/results.json`; `loocv_cross_session/results.json` |
| R1.2 | M11, M12, M19 | `multiseed_loocv/`; `conformal_calibration/` |
| R1.3 | M1, M19, M20, M30, M31 (+G3) | `zenodo_external_validation/` (FastSurfer nWBV, n = 10 of 11 paired, commit e389ba6); fig24 |
| R1.4 | M3, M8, M9 | `sim_validation/`; `significance_tests/` T2, T3 |
| R1.5 | M8, M13, M14, M15, M21, M23 | `ablation_gaussblur/`; `ssl_comparator_*`; `dino_headline_loocv/`; `arch_comparator_*`; `ablation_adapter/`+`ablation_lora/`; `simulation_sensitivity/` |
| R1.6 | M7, M17, M18 | `simulated_dementia/`; recomputed Cohen's d |
| R1.7 | M5, **M5b**, M6, M11, M16, **M34** | `real64mt_eval/mc_dropout_ci.json`; `oasis_mc_dropout/`; `conformal_calibration/`; `paper_statistics/failure_analysis.json` |
| R1.8 | M8, M21 | `significance_tests/`; `permutation_test/`; T4 re-derivation (02 §A) |
| R1.9 | M12 | `multiseed_loocv/` |
| R1.10 | M9, M23 | `simulation_sensitivity/`; Table I sources |
| R1.11 | M24, **M35/M35b** (11 Arabic section refs corrected) | word/repetition counts in `WORD_AGENT_PROMPT_R1.11.md` |
| R2.1 | M2, M3, M4, M18, M21 | `oasis_bootstrap/`; `significance_tests/` T1; param recount (02 §C) |
| R2.2 | M14, M20, M25, W1 | `arch_comparator_swin/`; `arch_comparator_unetr/` |
| R2.3 | M1, M13, M21, M22, W2 | `ssl_comparator_{dino,mae,simmim,contrastive}/`; `dino_headline_loocv/` |
| R2.4 | M15, M26 | `ablation_adapter/`; `ablation_lora/` |
| R3.1 | — (acknowledged) | — |
| R3.2 | M18, M27 | — (framing) |
| R3.3 | — (acknowledged) | `loocv_cross_session/` |
| R3.4 | M1, M10 | `permutation_test/` |
| R3.5 | M10, M12, M15, M19, M20 | as R1.5/R1.9 + limitation |
| R3.6 | M17, M18, M27 | `simulated_dementia/` |
| R3.7 | M3, M5, M9, M11, M16, M18, M27, M29 | `significance_tests/`; `conformal_calibration/`; `oasis_mc_dropout/` |
| R4.1 | M24 (acknowledged) | — |
| R4.2 | M23, M25 (acknowledged) | — |
| R4.3 | 07_FIGURE_PLAN, M6, M16, **M32** (full figure renumber; transposed captions fixed; orphan citation added) | `07_FINAL_FIGURES/MANIFEST.md` |
| R4.4 | M9, M14, M21, M28 | `sim_validation/`; `synthseg_output/synthseg_comparison.json` |
| R4.5 | M19, M20, M28 | — (scope) |
| R4.6 | M9, M14, M18, M21, M28, M29 + figure plan | union of the above |

No hole: all 28 concerns have a block above; every MUST edit in 05 (M1–M22, M24–M35b,
W1–W5) is cited by at least one concern (M24→R1.11; W3/W4/W5 are assembly-mechanical
and are covered under R4.3/R1.11 assembly gates); every number above appears in
`02_EVIDENCE_INVENTORY.md` with its source file. No action promises future work as a
remedy — future work appears only inside limitation statements (R1.3, R3.5, R4.5).
