# 08 — RISK ASSESSMENT: where this can still fail

One resubmission. Binary review. Same AE, likely the same four reviewers, who will
receive the response letter and check the manuscript against it. Ranked honestly.

---

## Risk 1 — Reviewer 2 rejects the "honest negatives" trade (highest residual risk)

**The gamble, named:** we answer all four of R2's concerns with run experiments, and
three of the four outcomes go *against* the paper's own configuration (CNN > ViT on
OASIS; Swin/UNETR > ViT on OASIS; DINO/MAE/SimMIM > denoising on OASIS). Our bet is
that R2 — whose stated objection was *unsupported claims*, not *this particular
architecture* — accepts "every claim now has evidence, and the evidence is reported
even when unfavourable" as technical soundness. The DINO-does-not-transfer result
(0.0220 vs 0.0134 on real hardware) is the one finding that runs in our favour, and
it is genuinely interesting.

**How it fails:** R2 reads the same table and concludes "then the paper studied the
wrong system — retrain with Swin/DINO and come back." No revision without new
training can preempt that reading.

**Mitigation in place:** the tractability rationale is stated as a constraint, not a
virtue (M25); the evaluation-depth asymmetry is a named limitation; the contribution
list no longer contains any component-optimality claim for him to strike at (M3); and
the transfer subsection reframes "wrong component" into "component choice is not
predictable from high-field performance — measured". Probability R2 still votes
reject: meaningful but reduced; this is the risk we accept because the alternative
(retraining) is outside the resubmission's possibility space.

**Sub-risk 1a — the removed probe.** R2 saw nothing about the frozen-encoder probe in
Review 1 (it was added post-rejection), so removing it (W1) costs nothing with him.
But if any draft shown to the AE or any figure retains "0.0112/0.0117" or "capacity
gives no benefit", we would be publishing a comparison we know to be unsound. W1 and
the fig05 rebuild are therefore *blocking*; their verification step is cheap and must
run.

## Risk 2 — REVISED: external validation was run, and it failed

**Superseded 2026-08-23.** The original Risk 2 was that R1 would treat "external
validation is required" as binary and reject an argued limitation. That risk is gone:
the validation was run (`experiments/zenodo_external_validation/`, commit e389ba6) with
FastSurfer ground truth derived by the paper's own pipeline. R1.3 is answered.

**The replacement risk is the result.** External MAE is 0.0731 against that cohort's
constant-mean baseline of 0.0089, the adapted model is worse than the unadapted one,
and predictions collapse to the adaptation cohort's mean. R1 or R3 may read this as
proof that the method does not work at all, rather than as a bounded and diagnosed
limitation.

**Why we still expect this to help more than it hurts.** First, it is not a new
failure: it is the internal Section V-E finding (r = −0.119) reproduced at an
independent site, so the paper gains consistency rather than acquiring a contradiction.
Second, the cause is measured, not speculated — the cohorts' nWBV ranges do not
overlap and the model returns its training mean, which is ordinary
out-of-distribution behaviour rather than an unexplained collapse. Third, and
decisively, the repository is public and the experiment is already committed: a
resubmission whose Limitations declined external validation while the public code
contained a completed one would not survive a reviewer who looked.

**Residual risk: moderate, and now unavoidable.** The honest presentation is also the
only presentation available.

**Sub-risk 2a — the negative-adaptation finding.** That the adapter transfers
negatively (0.0731 vs 0.0327 unadapted) is uncomfortable, because the Stage-3 adapter
is the paper's headline contribution. It must not be buried: it is the single most
useful thing this revision tells the field about adapters on 23-subject cohorts.

## Risk 3 — Reviewer 4's modality demand is refused

**The gamble:** we give R4 everything actionable (sharp vector figures, expanded
quantitative SOTA, an image-quality comparison for the simulator, and the scope
argument printed in the manuscript) and refuse the two requests that would change the
research question. R4 voted Yes/Yes on contribution and soundness; his lever was
comprehensiveness. If he insists on multi-modality evaluation as an acceptance
condition, no version of this study satisfies him.

**Mitigation:** the manuscript-body scope subsection (M28) ensures he cannot say the
concern was ignored; the letter's R4.6 block enumerates what was added. Residual
risk: low-moderate — R4 reads as a form-driven reviewer, and his concrete complaint
(figures) is fully fixed.

## Risk 4 — Self-inflicted: the integrity corrections draw attention

**The gamble:** the preamble discloses seven author-initiated corrections (five statistical, plus the withdrawn 47 ms GPU latency claim and the added external validation), including
that three figures were built from placeholder synthetic intervals. Disclosure is the
right call — reviewers will see fig20/21/22 change wholesale and the r-value flip
sign; a silent swap discovered by a reviewer would be fatal, while a disclosed
correction with the conclusion *strengthened* (miscalibration worse than reported)
reads as diligence. But it is still an admission that the v1 figures contained
fabricated intervals, and an unsympathetic reviewer or AE could weight that heavily.

**Mitigation:** the corrections all move against the paper's interest (baseline
0.0128 > model comparison worsens; blur significantly better; CNN twice the claimed
size; coverage still catastrophic), which is the signature of honest correction, and
the preamble says so in one place instead of scattering it. Residual risk: low, but
non-zero, and accepted deliberately.

## Risk 5 — Execution risk in the Word document (mechanical, but real)

The working document has absorbed five editing passes plus queued prompts. Failure
modes: a stale claim survives (0.0126, p = 0.847, +0.34, "only public dataset",
4.1 M, "LoRA recovers", Swin "r = 0.939" pre-retrain values, the probe's
0.0112/0.0117); a placeholder marker ships; the response letter asserts something the
PDF doesn't show (the exact failure IEEE binary review punishes).

**Mitigation — a final grep-style gate on the exported clean PDF text; all counts
must be zero:** `0.0126` · `p = 0.847` / `p=0.847` (bare `0.847` is LEGITIMATE — Fisher CI bound of ViT3D's r = 0.722 [0.524–0.847]; must NOT be scrubbed) · `+0.34` · `1.368` · `only public` · `4.1M` /
`4.1 M` (CNN context) · `0.0112` · `0.0117` · `FIGURE PLACEHOLDER` · `Notcom` ·
`recovers most of the full-fine-tuning benefit` · **`47 ms`** · **`standard GPU`** · **`4.1M`/`4.1 M`** · **`only public`**. And four counts must be non-zero:
`0.0128` (≥4) · `23.7` (≥1) · `−0.164` or `-0.164` (≥2) · `van den Broek` (≥2) · **`0.0731`** (≥2) · **`8.22`** (≥1, CNN3D parameter count).
Also verify gates G1 (age correlations re-derived on the dataset machine), G2 (47 ms
provenance), G3 (Zenodo citation), G4 (fig22 rebuild) before upload.

## Risk 6 — The paper's honest core: it still doesn't beat the mean

Strip everything away and the headline remains: MAE 0.0134 vs constant-mean 0.0128,
prediction–truth r negative. Any reviewer may conclude a feasibility baseline whose
model has not demonstrated anatomical signal is not a contribution. Our answer is
R3's own review (which found exactly this paper contributory when framed honestly),
the permutation result (the model is measurably input-dependent — a non-obvious,
reproducible finding at this field strength), the first-of-kind protocol, and the
transfer finding. That is the paper. If the panel doesn't buy it, there was no
version of this resubmission that they would have bought — only a different study.

## Least likely to be satisfied, ranked

1. **R2** (Risk 1) — the make-or-break reviewer, as in round 1.
2. **R1** (Risk 2) — R1.3 is now closed with an experiment; the residual risk is how R1 reads its unfavourable outcome.
3. **R4** (Risk 3) — fixable complaint fixed; scope refusal argued.
4. **R3** — should be satisfied; the revision is nearly a transcription of his review.

## The one-line gamble

We are betting the paper's acceptance on reviewers rewarding *completed evidence plus
disciplined honesty about unfavourable results* over component-level performance —
because with no new experiments permitted, honesty is the only axis on which this
manuscript can still improve, and it is the axis Reviewer 3 already voted for.
