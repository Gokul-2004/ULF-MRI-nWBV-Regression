# Prompt for Fable 5 — Review-2 resubmission audit

Paste everything below the line.

---

You are auditing a rejected IEEE Access manuscript before its ONE permitted
resubmission. If the resubmission fails, the paper is dead — no further chances.

Working directory: `/Users/gokulkrishnan.nair/Desktop/ULF-MRI-nWBV-Regression`
Source package: `FABLE - 5    FINAL ONE/` (note: four spaces in the folder name)

## YOUR ROLE: ANALYST, NOT EDITOR

Do NOT modify, rewrite, or regenerate anything. Do not touch the manuscript, the
figures, the scripts, or any experiment. Do not run experiments or training. You
are READ-ONLY everywhere except one output folder.

Your entire output is a new folder: `FABLE - 5    FINAL ONE/PLAN/`

Everything you conclude goes in there as documents describing changes to be made
by someone else. Specify changes precisely enough that they can be applied
without you — quote the exact "before" text and write out the exact "after" text.
Write nothing outside `PLAN/`.

## THE CONSTRAINT THAT SHAPES EVERYTHING

Find the best achievable path to ACCEPTANCE using ONLY the experiments and results
that already exist in this repository. Assume no new experiment will be run, no new
data acquired, no model retrained.

So the question is never "what experiment would satisfy this reviewer?" — it is:

> Given the evidence we already have, what is the strongest honest response, and
> how must the paper be reframed so that what we have is sufficient?

Where existing evidence genuinely cannot close a concern, the answer is a reasoned
limitation, a scope boundary, or a defensible pushback to the editor — never a
promised future experiment. Say so explicitly and draft the wording.

## HOW NUCLEAR YOU MAY GO (resubmission blast radius)

Context that bounds every recommendation: this is a RESUBMISSION of manuscript
Access-2026-28453 to the same Associate Editor, and IEEE Access resubmissions
typically go back to the ORIGINAL reviewers — who are shown the response document
and will check whether their specific concern was actually addressed in the paper.
It is not a fresh submission. It must read as the same study, honestly corrected —
not as a different paper wearing the same manuscript ID.

Work within these tiers, and label every recommendation you make with its tier.

**TIER 0 — free, do it without argument.**
Figure quality and captions. Redundancy trimming (R1.11). Correcting the wrong
statistics and the 4.3 % → 23.7 % caption. Adding limitations. Language polish.
All changes must still be yellow-highlighted in the resubmitted PDF.

**TIER 1 — encouraged, a reviewer explicitly asked for it.**
Reframing claims DOWNWARD. Removing overstatement. Bounding clinical language.
Rewriting abstract / discussion / conclusion to foreground the feasibility
baseline. R3.2 and R1.1 ask for exactly this — moving in this direction reads as
responsive, never as evasive.

**TIER 2 — allowed, but each needs one justifying sentence in the response letter.**
Retitling, IF the new title is more conservative and more accurate. Rewriting the
contribution list. Demoting the ViT from "better architecture" to "one compact
architecture evaluated, reported with its shortfall". Adding a "what this paper
does not claim" subsection. Promoting existing-but-unused experiments into the
main text. Moving figures to supplementary.

**TIER 3 — risky, recommend only with an explicit argument for why it helps.**
Changing which model is the headline result (e.g. foregrounding CNN3D).
Restructuring the results narrative around a negative finding. Reordering sections
substantially. These can read as responsive OR as a bait-and-switch — if you
recommend one, state plainly how the response letter makes it land as the former,
and name the risk if it doesn't.

**TIER 4 — forbidden, do not propose.**
Changing the study scope, datasets, or research question. Adding or removing
authors (needs a separate formal Editor request). Presenting any experiment we did
not actually run. Softening a reviewer's concern by deleting the evidence that
provoked it.

**THE ONE TRAP, STATED EXPLICITLY.** Do not propose removing or burying the OASIS
CNN3D-vs-ViT3D comparison to make R2.1 go away. Reviewer 2 read those exact
numbers and will look for them. Deleting an inconvenient result is the single
fastest route to a final rejection. The only viable move is to keep the
comparison, report it prominently, and reframe what it means.

**CALIBRATION.** Reviewers reject resubmissions for under-reacting far more often
than for over-reacting. A timid revision that argues the paper was fine loses.
Prefer Tier 1–2 boldness with airtight honesty over Tier 0 cosmetics. But every
tier above 1 must be earned by an argument in the response letter, not just
executed.

## READ THESE FIRST, IN ORDER

1. `FABLE - 5    FINAL ONE/00_README_START_HERE.md`
2. `FABLE - 5    FINAL ONE/02_Reviewer_Comments_VERBATIM_Review1.md` — all 4 reviewers, raw
3. `FABLE - 5    FINAL ONE/04_Editor_Decision_Letter_and_Resubmission_Rules.md`
4. `FABLE - 5    FINAL ONE/01_Manuscript_AS_SUBMITTED_Review1.pdf` — what they reviewed
5. `FABLE - 5    FINAL ONE/07_FINAL_FIGURES/MANIFEST.md` and the figure set (fig01–fig23, plus fig22D)
6. `FABLE - 5    FINAL ONE/08_FIGURE_DATA_INTEGRITY_REPORT.md`
7. `experiments/*/results.json` — EVERY one. This is your evidence inventory.
8. `Review - 2/` — all of it: prior plans, draft letters, audits, WIP manuscript
9. `CLAUDE.md` — pipeline, scripts, conventions

**FORM YOUR OWN JUDGEMENT.** `Review - 2/` holds the strategy we built on the
first pass plus draft response letters. Read them as a record of what was already
attempted — NOT as a plan to continue. Deliberately set that strategy aside and
reason from the raw reviewer comments up. If our earlier approach was wrong or
misprioritised, say so loudly and say why.

## DELIVERABLES IN `PLAN/`

### `01_DIAGNOSIS.md`
Why was this rejected? Reviewer 2 answered "No" to every question. Sort every
concern into: genuinely fatal / closable with existing evidence / framing-and-
presentation only. Be blunt about which criticisms are simply correct. Do not
defend the paper reflexively.

### `02_EVIDENCE_INVENTORY.md`
What we actually have. One row per existing experiment: what it tested, the real
numbers (read from `results.json`, not from the manuscript), which figure shows
it, and which reviewer concern it can serve. Include experiments the paper never
used. This is the raw material for everything else.

### `03_COVERAGE_MATRIX.md`
Every numbered concern from all 4 reviewers, one row each:

| concern | existing evidence (cite exact results.json / figure) | closable with what we have? YES / PARTIAL / NO | how it gets closed |

Where the answer is NO, state it plainly. Never invent a number.

### `04_THE_FOUR_HARD_ONES.md`
For each, the strongest defensible position reachable with current evidence:

- **R1.1 / R3.4** — MAE 0.0134 vs constant-mean 0.0126. Is there real evidence of
  anatomy-dependent learning in what we already ran (permutation tests, ICC,
  adapter ablation)? If not, how must the contribution be reframed so the paper is
  still publishable and still honest?
- **R2.1** — ViT3D loses to CNN3D on OASIS (r 0.722 vs 0.877). Does the ViT
  survive as the headline model, or does the story change? You are explicitly
  authorised to recommend abandoning the "ViT is better" claim entirely if that is
  the honest read — including retitling or restructuring the contribution (Tier 2/3).
- **R1.3** — external validation on an independent dataset. If no existing data
  supports it, draft the limitation/pushback instead.
- **R4.4 / R4.5** — SOTA qualitative comparison across "broader medical
  modalities". Judge whether this is in scope for a 64 mT nWBV feasibility paper at
  all, and draft either the comparison plan or the reasoned scope pushback.

The editor explicitly invites disagreement — but it must be argued, not asserted.

### `05_MANUSCRIPT_CHANGES.md` — the edit spec, for whoever applies it
Numbered edit list. Give every edit a stable ID (M1, M2, M3…) — the response letter
will cite these IDs. For each edit:

- **ID** | **Tier (0–4)** | **Location** (section name + page/figure number in the submitted PDF)
- **BEFORE:** the current text, quoted exactly as it appears
- **AFTER:** the replacement text, written out in full and ready to paste
- **WHY:** which reviewer concern(s) this serves, by number (e.g. R1.7, R3.6)
- **PRIORITY:** MUST / SHOULD / OPTIONAL

Cover title, abstract, contributions, and conclusion if the framing changes.
Include every correction from `08_FIGURE_DATA_INTEGRITY_REPORT.md`.

Do not write "revise this paragraph to be more cautious" — write the paragraph.
Anyone applying this file must never have to invent wording.

### `06_RESPONSE_TO_REVIEWERS.md` — ready to paste into the IEEE template
Follow the exact structure of `05_TEMPLATE_Response_to_Reviewers_IEEE.docx`:

> **Reviewer#1, Concern #1:** *(quote the concern verbatim)*
> **Author response:** the substance — what we found, what the evidence shows, and
> where we agree with the reviewer. Agree first where they are right; it buys
> credibility for the pushbacks.
> **Author action:** what changed in the paper, citing the edit IDs from 05 and the
> new section/page/figure where the reviewer can see it.

All four reviewers, every numbered concern, none skipped — including the ones where
our answer is a limitation or a scope pushback. Reviewer 3's positive points
(3.1, 3.2, 3.3) still get a short acknowledgement; do not silently drop them.

### `07_FIGURE_PLAN.md`
Which figures go in, in what order, with what numbering, which format per figure
(per `MANIFEST.md` — SVG for most, PNG for Figure 1), and which captions change.
Note any figure that should be dropped or moved to supplementary — Reviewer 1
complained about redundancy, Reviewer 4 about blurred figures.

### `08_RISK_ASSESSMENT.md`
Where is this resubmission still most likely to fail, honestly? Which reviewer is
least likely to be satisfied, and what is the residual risk we are choosing to
accept? One resubmission only — name the gamble.

## CONSISTENCY RULE

`05_MANUSCRIPT_CHANGES.md` and `06_RESPONSE_TO_REVIEWERS.md` are one artefact in
two halves:

- Every "Author action" in 06 must cite at least one edit ID that exists in 05.
- Every MUST edit in 05 must be referenced by at least one concern in 06.
- Every number quoted in 06 must trace to a `results.json` recorded in 02.
- No "Author action" may promise future work as the remedy. If nothing changed
  because nothing could change, say that in the response and explain why the
  existing evidence or an explicit limitation is the appropriate answer.

End 06 with a short cross-check table: concern → edit IDs → evidence source. If
that table has a hole, you have not finished.

## NON-NEGOTIABLES

- **Verify before asserting.** Read the actual `results.json`. Do not trust numbers
  in the manuscript or in the old draft letters — prior versions contained real
  errors: two figures were built from `np.random.normal` instead of real data, and
  two reported statistics were wrong (see `08_FIGURE_DATA_INTEGRITY_REPORT.md`).
- **Never fabricate** a result, a metric, or a citation. "We do not have this" is an
  acceptable and expected finding.
- **Keep the honest feasibility-baseline framing.** Reviewer 3 praised it explicitly
  and it is the paper's strongest asset. Never overstate to appease a reviewer.
- **Every concern must be answerable in BOTH** the response letter and the manuscript
  body — reviewers check that the paper changed, not just the letter.
- **Three uploads are required** at resubmission: the point-by-point response, a
  yellow-highlighted manuscript PDF, and a clean manuscript (Word/LaTeX + PDF).
  Your plan must be applicable to all three.
- **Write nothing outside `PLAN/`.**
