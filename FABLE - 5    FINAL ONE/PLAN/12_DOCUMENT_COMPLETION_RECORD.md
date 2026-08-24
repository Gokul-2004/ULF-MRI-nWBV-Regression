# 12 — Document completion record (2026-08-24)

All manuscript text edits are applied to `Manuscript_v2_WORKING.docx`. The LaTeX
sources are abandoned; the .docx is the sole submission source.

## Applied

M1–M35b plus W-series, 40 edits in total. W1 and W3 were no-ops — they targeted
WIP artifacts absent from the original submission, which is what the working
document turned out to be.

## Found during application, not in the original plan

Six defects surfaced by the Word pass's own auditing rather than by any planned edit.
Each is recorded because the class matters more than the instance: quote-matched
edits cannot find a defect nobody thought to look for.

| # | Defect | How it was found |
|---|---|---|
| M33 | The Results-section `0.0126` (para 210) — no edit covered it | forbidden-value sweep |
| M5b | `Both observations` left with one antecedent after M5 deleted the other | reading the sentence after the edit |
| M34 | Conformal calibration deferred as "future work" in §VI while reported as a result in §V | sweeping the phrase "identified as future work" |
| M35/M35b + 8 | Eleven Arabic section references (`Section 5.9`, `Section 6.4`) in a roman-numbered paper | pattern sweep after the first instance |
| — | Transposed figure captions: failure-analysis in §V-N, MC-calibration in §V-O | building the physical caption inventory |
| — | Duplicate `TABLE VII` and out-of-order `TABLE VI` | building the physical table inventory |
| Fix C | §VI-B appealed to "the global self-attention argument below" — an argument M18 had just deleted | tracing why "like-for-like advantage" read badly |

Two further corrections were to the plan itself, not the document:

- **The Phase 9 gate listed bare `0.847` as must-be-zero.** Two occurrences are the
  upper bound of a Fisher CI (`r = 0.722 [0.524–0.847]`). Following the gate would
  have deleted real data in a revision whose premise is data integrity.
- **The abstract was 456 words against a 250-word ceiling** after M1. Not redundancy —
  a compliance failure, created by this plan. M24 brought it to 257.

## Cross-reference integrity

The submitted manuscript had four independent cross-referencing defects: eleven
Arabic section references, transposed figure captions, a duplicate table number, and
an orphaned figure citation. Reviewer 4's "not presented in a comprehensive manner"
was registering something real.

All now resolved:
- Figures auto-number 1–24 by position; every one cited; zero dangling callouts
- Tables renumbered I–X by position; every one cited; zero Arabic references
- Section cross-references all roman, all verified by content rather than arithmetic
- Google-Docs anchor hyperlinks stripped where display text was wrong or targets junk

## Verification state at completion

**Forbidden — all zero:** `0.0126` · `47ms` · `standard GPU` · `4.1M` · `p = 0.847` ·
`+0.34` · `1.368` · `only public` · `0.0112` · `0.0117` · `Notcom` · `Section 5.x` ·
`Section 6.x` · `like-for-like` · `@@`

**Required — all present:** `0.0128` ×5 · `−0.164` ×4 · `23.7` ×2 · `8.22` ×1 ·
`4.53` ×7 · `0.0731` ×2 · `van den Broek` ×4 · `[42]`

**Abstract:** 257 words (was 456; ceiling 250).

## Outstanding

1. Nine `[[INSERT FIGURE: filename]]` placeholders need images from
   `Review - 2/figures_to_insert/`. PNG for Figure 1, SVG for 2–24.
2. `06_RESPONSE_TO_REVIEWERS.md` must be reconciled against the finished document —
   all section, figure and table numbers changed during application.
3. The three uploads: response document, highlighted PDF, clean .docx + PDF.
