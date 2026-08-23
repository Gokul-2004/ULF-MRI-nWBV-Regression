# 04 — THE FOUR HARD ONES: strongest defensible positions

For each: what the evidence actually shows, the strongest honest position, the exact
framing, and the tier of every recommended move.

---

## HARD ONE #1 — R1.1 / R3.4: MAE 0.0134 vs constant-mean, and "is there anatomy-dependent learning?"

### What the evidence actually shows (all re-verified from per-subject data)

1. Leave-one-out constant-mean MAE = **0.01281**; global-mean = 0.01225; even a LOO
   *median* predictor gets 0.01227. The model's 0.0134 is **above all three**. The
   submitted "0.0126" is not reproducible from any committed file.
2. The adapted model's predictions span an SD of **0.0041** against a true SD of
   **0.0152** — the output is compressed to ~27 % of the biological spread. Predictions
   do not correlate with truth (r = −0.119, p = 0.590) or with age (ρ = +0.232,
   p = 0.287, pending re-verification of the age file).
3. **But** the permutation test is unambiguous: inter-session ICC 0.6146 against a null
   of ≈ 0, **p = 0.0024** (20,000 permutations). The same subject gets the same
   prediction across two independent scan sessions far beyond chance. A constant or
   noise predictor cannot do that.

So: the model reads *something* real and subject-specific from a raw 64 mT scan, stably
across sessions — and that something is **not (yet) nWBV**. There is **no existing
evidence of anatomy-dependent nWBV learning on real hardware**, and no analysis in the
repo can manufacture it. The adapter ablation does not rescue this (all arms sit at the
baseline level); the ICC alone cannot (reproducibility ≠ validity).

### Strongest honest position

**Concede accuracy; claim the protocol, the input-dependence result, and the bounded
feasibility envelope.** The paper's publishable contribution is exactly what R3
enumerated when voting "Yes" on contribution: the physics-simulation recipe, the
leakage-free cross-session LOOCV protocol, and the transparent failure characterisation
— now extended with the permutation pair, which *answers R3.4's question in the
negative and says so*:

> Input-dependence: demonstrated (p = 0.0024). Anatomy-tracking: not demonstrated
> (r = −0.119, p = 0.590). The model is reproducible without being accurate in the
> biological direction; establishing anatomical tracking requires a cohort whose nWBV
> range exceeds the current 0.056.

This is not spin. It is the only claim the data supports, and it is the claim R3
explicitly said would make the paper acceptable ("the study is strongest when presented
as a reproducible feasibility baseline"). The abstract, contribution 4, §V-E, and the
conclusion must all carry the two-sided statement; nothing anywhere may imply the model
"captures anatomy" or "learns morphometry" on real hardware.

Why a reviewer should still find this publishable (the letter's argument): a documented,
reproducible, subject-independent benchmark showing that a 4.23 M-parameter
segmentation-free regressor achieves threshold-level MAE *without* exceeding a cohort
prior — with the failure modes mapped — is the honest starting line every future
64 mT-morphometry paper needs, and no such baseline exists in the literature (the
"first direct nWBV regression from 64 mT under subject-independent LOOCV" novelty claim
survives untouched).

**Tier: 1** (reframing downward — exactly what R1.1 and R3.2 request). The permutation
subsection itself is **Tier 2** (promoting an existing-but-unused experiment;
justification sentence in the letter: "added at the direct request of Reviewers 1 and 3").

### What must NOT be done

- Do not lean on `loocv_cross_session_v2` (MAE 0.0129) — better MAE, but ICC collapses
  to 0.330, which would destroy the only discriminative evidence the paper has.
- Do not present the ICC permutation as evidence of *accuracy* — R3 will see through it.
- Do not surface the unadapted model's age correlation (−0.49): it comes from the
  high-variance unadapted output and was already purged once (July plan Exposure 2).

---

## HARD ONE #2 — R2.1: ViT3D loses to CNN3D on OASIS. Does the ViT survive as headline?

### What the evidence actually shows

- OASIS-1 (n = 38): CNN3D r 0.877 / MAE 0.0243 vs ViT3D r 0.722 / MAE 0.0584 — and now
  formally **p = 0.0037** (paired Wilcoxon, T1). CNN wins. Further: Swin-UNETR
  (r 0.9636) and UNETR (r 0.9226) beat both. On high-field data the paper's
  architecture is demonstrably fourth of four.
- Real 64 mT, no adaptation (n = 23): ViT MAE 0.0403 (bias +0.0403) vs CNN 0.0763
  (bias −0.0763). ViT's error is almost purely a uniform offset (corrigible by a
  769-parameter adapter); CNN's is larger and of opposite sign. Whether the same
  adapter would fix CNN3D **was never tested** — the submitted paper already admits
  this, and it must stay admitted.
- Newly verified: CNN3D is **8.22 M parameters**, not the "≈ 4.1 M" in the manuscript.
  The "matched-parameter comparator" framing was never true; CNN3D has ~2× the ViT's
  parameters.

### Decision: ViT3D stays the headline **system**; "ViT is better" dies completely

The honest causal statement, which the paper must adopt verbatim in spirit: **the
measured gains on 64 mT come from the adaptation procedure and the narrow-range regime,
not from the transformer architecture.** That is R2's own sentence, and the data agrees
with him.

Why not switch the headline to CNN3D (the authorised Tier-3 option)? Because it is
impossible with existing evidence: CNN3D was never carried through Stage-3 LOOCV,
multi-seed, permutation, conformal, or failure analysis. A CNN-headlined paper would
rest on a single unadapted n = 23 evaluation — thinner than what R2 already rejected.
The only system with a complete evidence trail is ViT3D. So the defensible structure is:

1. **ViT3D = "the system studied", never "the architecture recommended".** Methods gains
   a selection-rationale paragraph: chosen a priori for compactness and CPU-tractability
   of the *full* protocol (≈ 20 h vs an estimated ≈ 13 days per arm for Swin-UNETR on
   the study's CPU-only hardware); explicitly *not* selected on accuracy.
2. **Report the losses prominently.** Table II stays; T1 goes next to it; the
   architecture-comparators subsection reports Swin/UNETR beating ViT on OASIS; the SSL
   subsection reports DINO/MAE/SimMIM beating denoising on OASIS. The paper stops
   defending its components and starts using their defeats as findings.
3. **The transfer finding is the scientific payoff** — and it must rest only on sound
   evidence: DINO, the strongest high-field objective, carried through the identical
   LOOCV, gets **worse** on target hardware (0.0220 vs 0.0134; ICC 0.523 vs 0.615;
   10/23 vs 19/23 below threshold; paired Wilcoxon p ≈ 0.070 n.s., bootstrap ΔMAE
   −0.0086 [−0.017, −0.001]). "High-field representation quality does not predict
   64 mT performance" is a genuinely useful negative result and turns R2's concern into
   the paper's most interesting new finding. The unsound frozen-encoder probe
   (0.0112/0.0117) is removed — resting this argument on it would hand R2 a
   methodological kill (he is the reviewer most likely to notice an 8-dim vs 256-dim
   probe).
4. **Discussion §B rewritten** from "why ViT3D's adapter correction matters /
   domain-specific not general" (which still reads as advocacy, and cites untested
   texture-bias mechanisms) to "Architecture choice: a tractability trade-off" — the
   choice is a constraint, the comparators are the context, the mechanism sentences go.

**Tier: 2** (demoting the ViT from contribution to studied system; rewriting the
contribution list; justifying sentence in the letter: "the reviewer is right that the
gains come from the adaptation procedure; we have rewritten the paper to say so").
Removing the WIP-era transfer probe is **Tier 0** (it is an error correction, and it
never appeared in the version reviewers saw).

### The trap, honoured

The OASIS CNN-vs-ViT numbers R2 quoted (0.877/0.024 vs 0.722/0.058) remain in Table II
in exactly the form he read them, now with a significance test attached. Nothing is
deleted, softened, or moved to supplementary.

---

## HARD ONE #3 — R1.3: external validation on an independent dataset

### What exists, honestly

- `zenodo_validation/` proves a second public paired 64 mT/3T cohort exists (van den
  Broek et al. 2025, Zenodo, 11 paired subjects) — which makes the submitted paper's
  "the only public paired 64 mT/3T brain MRI dataset" **factually false**.
- The analysis in that folder cannot serve as external validation: it evaluated an
  earlier multi-biomarker model against intensity-threshold proxies (BTF/VBR/TCR/MCI,
  GT ≈ 0.30–0.42 — a different quantity from FreeSurfer nWBV ≈ 0.78). Presenting it as
  validation of this paper's pipeline would be fabricated comparability.
- No other paired ULF data exists in the repo. External validation of the nWBV pipeline
  **cannot be produced from existing results, full stop.**

### Strongest honest position (three moves, in this order)

1. **Self-correct before the reviewer can.** The letter opens the R1.3 block by
   correcting the prior version's false claim and citing the Zenodo cohort. This
   converts a latent integrity bomb into demonstrated diligence.
2. **State the precise, checkable reason it is not used** — and only that reason: this
   study's ground truth is FreeSurfer/FastSurfer segmentation of the paired 3T scan; no
   such segmentation exists for the Zenodo cohort; deriving it is a separate
   segmentation study — *feasible in principle, a scope decision, not an impossibility*
   (never claim the dataset is unusable; that would be a second overclaim).
3. **Offer the internal generalisation evidence at its true value**: cross-session
   LOOCV across independent acquisitions, permutation p = 0.0024, multi-seed
   0.0130 ± 0.0004 — explicitly labelled "internal but genuine, and not a substitute
   for external validation". Future work names the Zenodo cohort with the ground-truth
   derivation it requires.

This is the one concern the resubmission openly fails to satisfy as literally worded.
The mitigation is that R1 wrote ten other concerns, all now closed with evidence, and
the editor's letter explicitly invites reasoned disagreement. The letter must not
hedge: "We have not performed external validation in this revision" appears in plain
words.

**Tier: 1** for the limitation reframe; the factual correction is **Tier 0**
(non-optional integrity fix). Drafted wording: edit M19/M20 and letter block R1.3.

---

## HARD ONE #4 — R4.4 / R4.5: qualitative SOTA image comparison, "broader medical modalities"

### Is it in scope? No — and the argument must live in the manuscript

R4's requests pattern-match to an image-synthesis/enhancement paper. This paper's model
emits **one scalar per scan**. There is no output image, no synthesis stage, no
super-resolution stage; a "qualitative image comparison with established approaches"
has no object to compare. Likewise "broader medical image modalities" would swap out
the physics simulator, the domain-shift model, the ground-truth pipeline and the
research question — i.e., a different study (changing scope is Tier 4 and forbidden
anyway).

But `MANUSCRIPT_AUDIT.md` D9 is right that the current WIP answers R4.4–4.6 **only in
the letter**, which reviewers are told to cross-check against the paper. The scope
argument must be printed in the manuscript body (new short subsection in Discussion or
Limitations — edit M28), so R4 can see his concern produced a visible change.

### What we give R4 instead (all existing evidence)

1. **Everything image-like that the study legitimately has**: fig01 (3T vs simulated
   64 mT vs real 64 mT axial slices — a qualitative image-level comparison of the one
   image-producing component, the simulator) plus the new quantitative fidelity table
   (NCC / SSIM / SNR-error vs the Arnold-style blur baseline, with paired tests) — an
   *image-quality comparison with an established approach*, exactly in the spirit of
   his request, applied to the only image-generating stage that exists.
2. **The quantitative SOTA comparison, expanded**: SynthSeg+ ceiling (r 0.918 /
   MAE 0.005), CNN3D, Swin-UNETR, UNETR, four SSL objectives, all with significance
   testing — the "comprehensive evaluation" half of R4.6, delivered.
3. **A visibly transformed figure set** (his concrete, actionable complaint): 600 dpi +
   vector SVG throughout, six new analysis figures.

### Residual risk, named

R4 answered Yes/Yes on contribution and soundness; his substantive lever was figures
(fixed) plus the scope demand (refused with argument). If he insists on multi-modality
evaluation as an acceptance condition, no revision within this study can satisfy him —
that risk is accepted and quantified in `08_RISK_ASSESSMENT.md`. The mitigation is the
editor: the response letter's R4.5 block states plainly that the request would change
the scientific question, which is precisely the class of disagreement the decision
letter invites authors to argue.

**Tier: 2** for the manuscript scope subsection (justifying sentence: "added so the
scope boundary is stated in the paper itself, not only in this response"); the letter
pushback itself is **Tier 1** (bounding claims, which R3.2 independently demands).
