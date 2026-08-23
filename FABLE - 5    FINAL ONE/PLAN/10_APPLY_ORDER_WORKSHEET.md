# 10 — Application worksheet: one top-to-bottom pass through `Manuscript_v2_WORKING.docx`

**Source of truth: `Review - 2/Manuscript_Revision_WIP/Manuscript_v2_WORKING.docx`.**
The LaTeX files (`standalone_paper/manuscript.tex`, `manuscript_WORKING.tex`) are
ABANDONED — do not edit them, do not copy from them. The CNN3D fix in commit 02ba580
went into the dead `.tex`; it is reproduced here as M4 and must be applied to the .docx.

IEEE requires the main manuscript as **Word or LaTeX plus a PDF**. The .docx is the
deliverable; the PDF is exported from it.

Full BEFORE/AFTER text for every edit is in `05_MANUSCRIPT_CHANGES.md` — this sheet is
the running order and the tick-list. Highlight every change in yellow as you go; the
highlighted PDF is a required upload.

---

## Phase 0 — before touching prose (do these first, they are global)

| ✓ | ID | Action |
|---|---|---|
| ☐ | **W1** | Remove the frozen-encoder transfer probe **everywhere** — abstract, results, discussion, fig05 panel B. Search: `0.0112`, `0.0117`, `capacity gives no benefit`, `frozen-encoder`, `ridge probe`. It is methodologically unsound (8-dim vs 256-dim features) and must not appear. |
| ☐ | **M31** | Subject-count discipline: external cohort is **11 paired, 10 analysable**. First use spells both out; all results say n = 10. |
| ☐ | **W3** | Mechanical: kill all six `FIGURE PLACEHOLDER` markers; add the missing fig17 citation in §V; unit spacing (`64 mT`, `0.020 nWBV`). |

## Phase 1 — front matter

| ✓ | ID | Location | Note |
|---|---|---|---|
| ☐ | **M1** | Abstract | Read the **REVISED 2026-08-23** note first: strike the 47 ms clause, add the external-validation sentence. |
| ☐ | **W5** | Abstract | Comparator clause consistency. |
| ☐ | **M3** | §I Contributions | Item 1 is scoped so it does not imply generalisation. |
| ☐ | **M2** | §II-C ViTs in Medical Imaging | |

## Phase 2 — methods

| ✓ | ID | Location |
|---|---|---|
| ☐ | **M9** | §III-A simulation — add fidelity validation |
| ☐ | **M23** | §III-A — parameter sensitivity *(SHOULD)* |
| ☐ | **M25** | §III-B ViT3D — architecture selection rationale |
| ☐ | **M26** | §III-C — LayerNorm+head rationale |
| ☐ | **M4** | §IV-B Baselines — CNN3D **8.22M**, explicitly not a matched-parameter comparator |

## Phase 3 — results (existing subsections)

| ✓ | ID | Location |
|---|---|---|
| ☐ | **M7** | §V-B CDR stratification — Cohen's d 1.463 |
| ☐ | **M8** | §V-C physics vs blur — the paired test was invalid; Mann-Whitney p = 0.81 |
| ☐ | **M5** | §V-I Uncertainty — r = −0.164, p = 0.454; delete the direction sentence |
| ☐ | **M6** | §V-J Failure Analysis panel D — **r = −0.164, p = 0.454**, never +0.34 |
| ☐ | **M11** | §V-I — conformal calibration |
| ☐ | **M16** | Fig. 20 caption — coverage **23.7 %** (9/38), not 4.3 % |

## Phase 4 — results (new subsections)

Insert in this order so the section letters stay sequential.

| ✓ | ID | New subsection |
|---|---|---|
| ☐ | **M10** | after §V-E — permutation test / input-dependence |
| ☐ | **M12** | after the ICC subsection — multi-seed stability |
| ☐ | **M13** | SSL pretraining comparators + 64 mT transfer |
| ☐ | **W2** | (folds into M13 — four objectives, not SimMIM alone) |
| ☐ | **M14** | architecture comparators |
| ☐ | **M15** | adapter strategy ablation — all four CIs overlap |
| ☐ | **M21** | statistical comparisons summary table |
| ☐ | **M30** | **NEW §V-J External validation** + insert **fig24** |

## Phase 5 — discussion, limitations, conclusion

| ✓ | ID | Location |
|---|---|---|
| ☐ | **M18** | §VI Discussion — rewrite subsection B, add scope subsection |
| ☐ | **M28** | §VI — scope subsection answering Reviewer 4 |
| ☐ | **M17** | dementia wording audit (abstract, §V-G, §VI, conclusion) |
| ☐ | **M19** | §VI-E Limitations — read the REVISED note: it now points to the validation, not away from it |
| ☐ | **M20** | §VI-F Future Works — adapter ablation, conformal, and external validation all LEAVE (they are results now) |
| ☐ | **M22** | §VII Conclusion |

## Phase 6 — blocked on the latency benchmark

| ✓ | ID | Note |
|---|---|---|
| ☐ | **M29** | Five locations. `experiments/inference_latency/results.json` gives: **median 4.53 ms**, IQR 1.37, N = 100, 6 threads, Intel i7-8750H, 64³, batch 1, cold full-volume pass. Use the **cold** figure. Remove "on a standard GPU" everywhere. |
| ☐ | **W4** | Reconcile every remaining latency number against that file; delete any that cannot be traced. |

## Phase 7 — run LAST, in this order

| ✓ | ID | Note |
|---|---|---|
| ☐ | **M24** | Redundancy consolidation (R1.11). Must run after all content edits or it will re-introduce cuts. |
| ☐ | **M27** | Framing audit — title, abstract, discussion, conclusion. Final consistency pass. |

## Phase 8 — figures

24 figures. Format per `MANIFEST.md`: **PNG for Fig. 1** (MRI slices), **SVG for 2–24**.
Test with fig02 first; if Word substitutes fonts badly, fall back to 600 dpi PNG
throughout — both satisfy R4.3.

Blocked: **fig10** (needs the M29 number) and **fig22** (panel C needs
`participants.tsv`; assemble with `fig22D_ci_width_vs_error_CORRECTED.*`, **r = −0.164**).

## Phase 9 — the gate (all counts must be ZERO)

`0.0126` · `p = 0.847` / `p=0.847` (bare `0.847` is LEGITIMATE — Fisher CI bound of ViT3D's r = 0.722 [0.524–0.847]; must NOT be scrubbed) · `+0.34` · `1.368` · `only public` · `4.1M` / `4.1 M` · `0.0112` ·
`0.0117` · `47 ms` · `standard GPU` · `FIGURE PLACEHOLDER` · `Notcom` ·
`recovers most of the full-fine-tuning benefit`

Must be NON-zero: `0.0128` (≥4) · `23.7` (≥1) · `−0.164`/`-0.164` (≥2) ·
`van den Broek` (≥2) · `0.0731` (≥2) · `8.22` (≥1) · `4.53` (≥1)

## Phase 10 — the three uploads

1. Response to reviewers — from `06_RESPONSE_TO_REVIEWERS.md`, in the IEEE template
2. Highlighted PDF — every change in yellow
3. Clean manuscript — .docx **and** PDF exported from it
