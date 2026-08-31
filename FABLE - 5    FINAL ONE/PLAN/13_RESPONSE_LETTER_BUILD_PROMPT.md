# 13 — Prompt for building the response-to-reviewers document

Open `05_TEMPLATE_Response_to_Reviewers_IEEE.docx` in Word. Paste the master prompt
below **once**, then feed the letter in five pieces from `06_RESPONSE_TO_REVIEWERS.md`:

1. the preamble (everything above "# REVIEWER 1")
2. Reviewer 1 — 11 concerns
3. Reviewer 2 — 4 concerns
4. Reviewer 3 — 7 concerns
5. Reviewer 4 — 6 concerns, then the cross-check table

Do not paste all 28 at once — the assistant will compress or paraphrase, and the
verbatim reviewer quotes are what the editor checks first.

---

## MASTER PROMPT — paste once

```
You are helping me assemble the response-to-reviewers document for the ONE permitted
resubmission of IEEE Access manuscript Access-2026-28453. The same four reviewers
will read this alongside the revised manuscript and check each claim against it.

The open document is the official IEEE Access response template. I will give you the
content in five pieces. For each, format it into the template's structure.

ABSOLUTE RULES:

1. Use my text VERBATIM. Do not rewrite, condense, soften, or "improve" it. Every
   number has been verified against a source file and every section reference against
   the revised manuscript. Rewording risks breaking one.

2. The reviewer's concern is quoted EXACTLY as the reviewer wrote it, including any
   typos or odd phrasing. Never clean up a reviewer's words.

3. Each concern becomes one block in the template's format:
       Reviewer #N, Concern #M:   <the reviewer's words, verbatim, italic or quoted
                                   per the template>
       Author response:           <my response text, verbatim>
       Author action:             <my action text, verbatim>

4. Never invent an action, a section number, a figure number, or a statistic. If any
   piece looks incomplete, stop and ask me.

5. Keep every concern, including ones where the response is an acknowledgement of a
   positive comment (Reviewer 3's Concerns 1 and 3, Reviewer 4's Concerns 1 and 2).
   Skipping them is noticed.

FORMATTING:
- Match the template's existing heading and body styles. If the template has a worked
  example, follow its layout exactly and delete the example when done.
- Bold the "Author response:" and "Author action:" labels as the template does.
- Preserve paragraph breaks inside responses; several run to multiple paragraphs.
- No highlighting in this document — highlighting is only for the manuscript PDF.
- Keep en-dashes, ±, ≈, and Greek letters as they appear. Do not let autocorrect
  convert them.

Confirm you understand, then wait for the first piece.
```

---

## The five pieces

Paste from `06_RESPONSE_TO_REVIEWERS.md`, in order. Prefix each with:

```
PIECE <n> of 5 — <what it is>. Format into the template. Verbatim.
```

**Piece 1 — preamble.** Everything before `# REVIEWER 1`. This carries the seven
author-initiated corrections and the cross-referencing disclosure. It goes before the
first concern block, as an introductory statement.

**Piece 2 — Reviewer 1**, Concerns 1–11.

**Piece 3 — Reviewer 2**, Concerns 1–4.

**Piece 4 — Reviewer 3**, Concerns 1–7.

**Piece 5 — Reviewer 4**, Concerns 1–6, then the cross-check table.

---

## On the cross-check table

The table at the end of `06` (concern → edit IDs → evidence source) is **an internal
verification tool, not part of the submission.** It exists so we can confirm every
claimed action maps to a real edit and a real result file.

Do not paste it into the response document unless you want it there. If you do,
strip the edit IDs (M1, W3, …) first — they are meaningless to a reviewer and reveal
internal process.

---

## Before you upload

- Every "Author action" names a location a reviewer can turn to in the revised PDF.
- No action promises future work as the remedy. Future work appears only inside
  limitation statements (R1.3, R3.5, R4.5).
- The numbers in the letter match the manuscript: 0.0128, −0.164, 23.7 %, 8.22 M,
  4.53 ms, 0.0731, and the van den Broek citation as reference [42].
- Section references match the final lettering: external validation is **V-Q**,
  Uncertainty Quantification is **V-N**, Limitations is **VI-F**.
