# Figure insertion checklist

Files in this folder are named by their **figure number in the manuscript**. Insert
`Figure_NN_*` at Figure NN. No mapping needed.

**Both formats are present for every figure, all 24 verified at 600 dpi.** Use
either — pick one and stay consistent:

- **PNG throughout (simplest).** All 24 are 600 dpi, well above IEEE's 300 dpi
  minimum for halftones. Immune to font-substitution problems. Larger .docx.
- **SVG for 2–24, PNG for Figure 1.** True vector, so Word's compression cannot
  touch it, and the file stays smaller. Figure 1 must stay PNG regardless — it is
  MRI slice imagery, and its SVG is just wrapped bitmaps.

If you use PNG throughout, **image compression must be disabled** — Word will
otherwise downsample to 220 ppi on export, which is exactly what caused R4.3.
Vector SVG is not affected by that setting; PNG is.

**All 24 must be replaced**, not only the nine placeholders. The fifteen figures that
still hold their original embedded image are at 220 ppi after Word's compression —
that is Reviewer 4's complaint (R4.3), and it is not fixed by leaving them.

Before you start: **disable image compression.** Right-click any picture →
Compress Pictures → highest quality → apply to all. Word downsampled the originals;
it will do it again.

| ✓ | Fig | How it appears now | Caption should mention |
|---|---|---|---|
| ☐ | 1 | embedded image | 3T / simulated / real axial slices |
| ☐ | 2 | **placeholder** | max \|ΔMAE\| = 0.0062, ±20 % sweep |
| ☐ | 3 | embedded — **LAYOUT WAS BROKEN**, must replace | three-stage pipeline, 5 boxes in a row |
| ☐ | 4 | embedded image | predicted vs true, OASIS-1 |
| ☐ | 5 | embedded — **LABELS OVERLAPPED**, must replace | true nWBV by CDR stage |
| ☐ | 6 | embedded image | physics vs blur, Mann-Whitney p = 0.81 |
| ☐ | 7 | embedded image | SynthSeg+ / CNN3D / ViT3D, no adaptation |
| ☐ | 8 | embedded — **SHOWS 47 ms / 95 s**, must replace | 4.53 ms measured, 11.9 ms CNN3D |
| ☐ | 9 | embedded image | ρ = −0.778 truth, ρ = +0.232 predictions |
| ☐ | 10 | embedded image | cross-session LOOCV scatter |
| ☐ | 11 | embedded image | bootstrap MAE, bias −0.0022 |
| ☐ | 12 | **placeholder** | ICC 0.6146 vs null ≈ 0, p = 0.0024 |
| ☐ | 13 | **placeholder** | 5 seeds, 0.0130 ± 0.0004, seed 7 ICC 0.563 |
| ☐ | 14 | embedded image | ICC 0.615 [0.236–0.866] |
| ☐ | 15 | embedded image | CDR-stratified, simulated, n = 2 directional |
| ☐ | 16 | embedded image | pseudo-label ablation, +0.005 penalty |
| ☐ | 17 | **placeholder** | 8 comparators, CNN3D 8.22M, DINO transfer |
| ☐ | 18 | **placeholder** | all four CIs overlap |
| ☐ | 19 | embedded — **SHOWS 4.3 % (SYNTHETIC)**, must replace | 23.7 % (9/38), width 0.0284 |
| ☐ | 20 | embedded — **SHOWS r = +0.34 (FABRICATED)**, must replace | r = −0.164, p = 0.454, unadapted |
| ☐ | 21 | **placeholder** | conformal 91.3 % @ 90 %, 95.7 % @ 95 % |
| ☐ | 22 | **placeholder** | reliability curve, 4.3 % at nominal 95 % |
| ☐ | 23 | **placeholder** | failure analysis, panel D r = −0.164 |
| ☐ | 24 | **placeholder** | external validation, ranges do not overlap |

## The five that are not optional

Figures **3, 5, 8, 19, 20** currently display content that contradicts their own
captions or was drawn from placeholder random data:

- **Fig 8** shows `47 ms` and `95 s` — both withdrawn as unmeasured
- **Fig 19** shows `4.3 %` coverage from a synthetic MC-Dropout run; the real OASIS
  value is `23.7 %`, and the caption already says so
- **Fig 20** shows `r = +0.34` — the fabricated statistic this revision discloses and
  corrects. The caption beneath it already reads `r = −0.164`

Shipping any of these would mean the figure disproves the caption above it.

## After inserting

1. Export to PDF, zoom to 400 % on Figure 1's slices — R4.3 is only fixed if they
   are sharp in the exported file.
2. Check Figures 8, 19, 20 by eye: no `47 ms`, no `95 s`, no `4.3 %` on the OASIS
   panel, no `+0.34` anywhere.
3. Confirm the abstract still reads 258 words.
