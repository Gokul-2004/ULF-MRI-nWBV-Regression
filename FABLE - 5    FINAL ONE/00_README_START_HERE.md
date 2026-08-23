# Review-1 Source Package — START HERE

Everything IEEE Access sent back on **Review Round 1** for manuscript
**Access-2026-28453**, plus the exact manuscript that was reviewed.

These are **source documents only** — raw inputs, no analysis, no plan.

## Files

| File | What it is |
|---|---|
| `01_Manuscript_AS_SUBMITTED_Review1.pdf` | The manuscript exactly as the four reviewers saw it. 13 pages, IEEE Access two-column format. This is the "before" state. Read-only reference — the editable Word source lives in `../Review - 2/Manuscript_Revision_WIP/`. |
| `02_Reviewer_Comments_VERBATIM_Review1.md` / `.docx` | All four reviewer reports, transcribed verbatim from the decision email — comments plus their Additional Questions ratings. No summarizing or regrouping. |
| `03_Reviewer4_Comments_ORIGINAL_ATTACHMENT.pdf` | Reviewer 4's report as a standalone PDF attachment from the email (same text as in file 02, kept for provenance). |
| `04_Editor_Decision_Letter_and_Resubmission_Rules.md` | Prof. Sandra Costanzo's decision letter, verbatim, plus the three required resubmission uploads as a checklist. |
| `05_TEMPLATE_Response_to_Reviewers_IEEE.docx` | Official IEEE Access template — concern / author response / author action, per comment. |
| `06_TEMPLATE_Request_for_Byline_Change_IEEE.docx` | Official IEEE form. Only needed if the author list changes. |
| `07_FINAL_FIGURES/` | **The current final figure set** for the resubmission — 23 figures, each as 600 dpi PNG **and** true-vector SVG. Read its `MANIFEST.md` first: it specifies which format to insert per figure, flags four regenerated figures, and records a caption number that must change (4.3 % → 23.7 % on Fig 20). `_superseded/` holds a quarantined old Fig 20 — do not insert it. |
| `08_FIGURE_DATA_INTEGRITY_REPORT.md` | The audit behind those regenerations: layout defects found and fixed, plus a data-integrity finding (two figures had used `np.random.normal` intervals instead of real data) and two manuscript statistics that must be corrected. |

## Hard constraints on the resubmission

- **Decision: Reject with one resubmission permitted.** No second chance.
- IEEE Access uses **binary peer review** — an article is rejected even for minor edits.
- Every reviewer concern must be visibly addressed in *both* the response document *and* the manuscript body.
- Three uploads required: point-by-point response, yellow-highlighted manuscript PDF, clean manuscript (Word/LaTeX + PDF).

## Reviewer disposition at a glance

| Reviewer | Contributes to knowledge? | Technically sound? | Comprehensive? |
|---|---|---|---|
| 1 | Partially | Partially | Yes |
| 2 | **No** | **No** | **No** |
| 3 | Yes | Partially yes | Yes, in general |
| 4 | Yes | Yes | **No** |

## Where the rest of the work lives (outside this folder)

- `../experiments/` — one subfolder per experiment, each with `results.json`. The record of every number in the paper, including the new post-Review-1 runs.
- `../Review - 2/` — all revision work in progress: `Manuscript_Revision_WIP/`, prior draft response letters, `RESUBMISSION_PLAN.md`, `MANUSCRIPT_AUDIT.md`. The figures in `07_FINAL_FIGURES/` were copied from here (`figures_to_insert/`).
- `../standalone_paper/` — manuscript source and prior `response_to_reviewers.md`.
- `../paper_figures/`, `../Pictures/` — generated figures.
- `../CLAUDE.md` — repository guide: pipeline structure, key scripts, how to reproduce.
