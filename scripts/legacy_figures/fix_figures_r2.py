"""
Fix the three defective manuscript figures + rebuild the MC Dropout figures from
REAL data.  Writes 600 dpi PNG + SVG straight into "Review - 2/figures_to_insert"
using the manuscript figure numbering.

What this fixes
---------------
Fig 3  (fig1_architecture)   layout bug: the ViT3D box was drawn at y=0.6 while every
                             other box sits at y=1.2, so it dropped out of the row and
                             the arrow chain broke.  Sub-captions were also x-offset by
                             +0.25 from their boxes.
Fig 6  (fig3_cdr_boxplot)    the mu= labels were drawn on top of the data points and the
                             y-axis spanned 0.58-1.08 for data occupying 0.645-0.810.
Fig 21 (fig8b_ci_real64mt)   REBUILT FROM REAL DATA.  The shipped figure drew its
                             intervals from np.random.normal(0.0145, 0.003) -- see
                             generate_all_figures.py:474.  Real per-subject MC Dropout
                             output exists in experiments/real64mt_eval/mc_dropout_ci.json
                             (N=100 passes) and is used here instead.
Fig 22D                      REBUILT FROM REAL DATA.  generate_all_figures.py:532 drew
                             ci_widths from np.random.normal and then computed
                             pearsonr(ci_widths, errors) from those random numbers,
                             producing the r=+0.34, p=0.108 reported in the manuscript.
                             The real correlation is r=-0.164, p=0.454.

NOT fixed here
--------------
Fig 20 (fig8a_ci_oasis)      Cannot be rebuilt: no OASIS-1 MC Dropout output exists
                             anywhere in the repo.  The shipped figure's intervals are
                             np.random.normal(0.0145, 0.004) and its "Coverage: 4.3%"
                             subtitle is the real-64mT number applied to an OASIS panel.
                             Either run MC Dropout on the OASIS-1 test set, or drop the
                             figure and the OASIS interval-width claim.

Run:  python scripts/fix_figures_r2.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "Review - 2" / "figures_to_insert"
EXP  = ROOT / "experiments"
DPI  = 600

OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "savefig.dpi":        DPI,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
    "savefig.facecolor":  "white",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "lines.linewidth":    1.5,
})

BLUE, ORANGE, GREEN, RED, PURPLE, GRAY = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f")
THRESH = 0.020


def save(fig, name):
    fig.savefig(str(OUT / f"{name}.png"), format="png", dpi=DPI)
    fig.savefig(str(OUT / f"{name}.svg"), format="svg")
    plt.close(fig)
    print(f"  wrote {name}.png ({DPI} dpi) + {name}.svg")


# ── FIG 3 — three-stage training pipeline ────────────────────────────────────
def fig03_pipeline():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")

    BOX_Y, BOX_H, BOX_W = 1.15, 0.95, 1.5
    MID = BOX_Y + BOX_H / 2                      # every box on one centre line

    boxes = [
        (0.20, "IXI\nPre-training\n($n$=156)",        BLUE),
        (2.15, "OASIS-1\nFine-tuning\n($n$=375)",     ORANGE),
        (4.10, "LOOCV\nAdaptation\n($n$=23)",         GREEN),
        (6.05, "ViT3D\n4.23M params\n64³ → nWBV",     PURPLE),
        (8.00, "Inference\n47 ms\nno FreeSurfer",     RED),
    ]
    subs = [
        "Physics 64 mT\nsimulation",
        "nWBV labels\n(FreeSurfer)",
        "Cross-session;\nFastSurfer GT (3T)",
        "LayerNorm + head\n(769 params)",
        "standard GPU",
    ]

    for (x, label, color), sub in zip(boxes, subs):
        ax.add_patch(plt.Rectangle((x, BOX_Y), BOX_W, BOX_H, linewidth=1.5,
                                   edgecolor=color, facecolor=color, alpha=0.88))
        ax.text(x + BOX_W / 2, MID, label, ha="center", va="center",
                fontsize=9, color="white", fontweight="bold", linespacing=1.45)
        # sub-caption centred on its own box
        ax.text(x + BOX_W / 2, BOX_Y - 0.16, sub, ha="center", va="top",
                fontsize=8, color=GRAY, style="italic", linespacing=1.3)

    # arrows between every consecutive pair, on the box centre line
    for (x0, _, _), (x1, _, _) in zip(boxes, boxes[1:]):
        ax.annotate("", xy=(x1 - 0.06, MID), xytext=(x0 + BOX_W + 0.06, MID),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.6))

    # stage banners over the three training stages named in the caption
    for (x, _, color), tag in zip(boxes[:3], ("STAGE 1", "STAGE 2", "STAGE 3")):
        ax.text(x + BOX_W / 2, BOX_Y + BOX_H + 0.13, tag, ha="center", va="bottom",
                fontsize=8.5, color=color, fontweight="bold")

    ax.set_title("Three-Stage Training Pipeline", fontsize=13,
                 fontweight="bold", pad=18)
    save(fig, "fig03_three_stage_training_pipeline")


# ── FIG 6 — nWBV by CDR stage ────────────────────────────────────────────────
def fig06_cdr_boxplot():
    oasis = json.loads((EXP / "oasis_finetune" / "finetune_results.json").read_text())
    groups = {0.0: [], 0.5: [], 1.0: []}
    for s in oasis["per_subject_test"]:
        if s["cdr"] in groups:
            groups[s["cdr"]].append(s["true_nwbv"])
    keys = [0.0, 0.5, 1.0]
    data = [groups[k] for k in keys]
    ns   = [len(g) for g in data]
    print(f"    CDR group sizes: {dict(zip(keys, ns))}")

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    colors = ["#4da6ff", "#1a6ed8", "#0a3d8f"]

    # showfliers=False: every raw point is drawn once, by the jittered scatter below
    bp = ax.boxplot(data, patch_artist=True, widths=0.50, showfliers=False,
                    medianprops=dict(color="white", lw=2.6),
                    whiskerprops=dict(lw=1.6, color="#333333"),
                    capprops=dict(lw=1.6, color="#333333"),
                    boxprops=dict(linewidth=1.6))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.82)
        patch.set_edgecolor("#1a1a1a"); patch.set_linewidth(1.6)

    # jitter the raw points sideways with larger spread for visibility
    rng = np.random.default_rng(42)
    for i, g in enumerate(data, 1):
        xj = i + rng.uniform(-0.14, 0.14, len(g))
        ax.scatter(xj, g, s=48, facecolor="white", edgecolor=colors[i - 1],
                   linewidth=1.5, zorder=4, alpha=0.92)

    lo = min(min(g) for g in data)
    hi = max(max(g) for g in data)
    pad = (hi - lo) * 0.12
    ymin, ymax = lo - pad * 0.8, hi + pad * 5.0      # more headroom for labels

    # mu labels ABOVE each group, larger and clearer
    label_y = hi + pad * 0.70
    for i, (g, c) in enumerate(zip(data, colors), 1):
        ax.text(i, label_y, f"$\\mu$={np.mean(g):.3f}", ha="center", va="bottom",
                fontsize=12, color="white", fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.40", facecolor=c,
                          edgecolor="white", linewidth=1.4, alpha=0.98))

    # significance bracket above the labels — clearer rendering
    by = hi + pad * 2.8
    tick = pad * 0.32
    ax.plot([2, 2, 3, 3], [by - tick, by, by, by - tick], color="#1a1a1a", lw=1.6, zorder=5)
    ax.text(2.5, by + tick * 0.8, "$p = 0.022$  (CDR 0.5 vs 1.0, one-sided)",
            ha="center", va="bottom", fontsize=9.5, color="#1a1a1a", fontweight="bold")

    ax.set_ylim(ymin, ymax)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f"CDR {k}\n($n={n}$)" for k, n in zip(keys, ns)], fontsize=11.5)
    ax.set_ylabel("True nWBV (FastSurfer)", fontsize=12, fontweight="bold")
    ax.set_title("nWBV by CDR Stage — OASIS-1 Test Set ($n=18$)",
                 fontsize=13.5, fontweight="bold", pad=16)
    ax.grid(axis="y", alpha=0.20, linestyle=":", linewidth=0.9)
    ax.set_axisbelow(True)

    # literature reference — larger, clearer positioning
    ax.text(0.5, -0.18, "Literature: 5–10% nWBV reduction per CDR stage",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, style="italic", color="#444444", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5f8ff",
                      edgecolor="#8899cc", linewidth=1.2, alpha=0.96))

    fig.tight_layout()
    save(fig, "fig06_true_nwbv_by_cdr_stage")


# ── MC Dropout real data, shared by fig 21 and fig 22D ───────────────────────
def _mc():
    d = json.loads((EXP / "real64mt_eval" / "mc_dropout_ci.json").read_text())
    ps = sorted(d["per_subject"], key=lambda s: s["gt"])
    return d, (np.array([s["gt"] for s in ps]),
               np.array([s["mean"] for s in ps]),
               np.array([s["ci_lo"] for s in ps]),
               np.array([s["ci_hi"] for s in ps]),
               np.array([s["ci_width"] for s in ps]))


# ── FIG 21 — MC Dropout intervals, real 64 mT (REAL DATA) ────────────────────
def fig21_mc_real64mt():
    d, (gt, mu, lo, hi, w) = _mc()
    err = np.abs(mu - gt)
    r, p = stats.pearsonr(w, err)
    cov = np.mean((gt >= lo) & (gt <= hi)) * 100
    print(f"    n={len(gt)}  width={w.mean():.4f}  coverage={cov:.1f}%  "
          f"r={r:+.3f} p={p:.3f}")

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    x = np.arange(len(gt))

    ax.fill_between(x, lo, hi, alpha=0.28, color=BLUE,
                    label=f"MC Dropout 95% CI (mean width {w.mean():.3f})")
    ax.plot(x, mu, "o-", color=BLUE, ms=4, lw=1.2, label="ViT3D prediction (no adapt.)")
    ax.plot(x, gt, "s-", color=RED, ms=4, lw=1.2, label="True nWBV (FastSurfer)")
    fail = np.where(err > THRESH)[0]
    ax.scatter(fail, gt[fail], marker="x", color=RED, s=70, zorder=5,
               linewidths=1.6, label=f"|error| > {THRESH:g}  ({len(fail)}/{len(gt)})")

    ymin = min(lo.min(), gt.min()); ymax = max(hi.max(), gt.max())
    pad = (ymax - ymin) * 0.10
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlim(-0.6, len(gt) - 0.4)
    ax.set_xlabel("Subject (sorted by true nWBV)")
    ax.set_ylabel("nWBV")
    ax.set_title(f"MC Dropout Uncertainty — Real 64 mT ($n$={len(gt)}, "
                 f"$N$=100 passes)\nEmpirical coverage {cov:.1f}% at nominal 95%; "
                 "intervals not suitable for clinical use",
                 fontsize=10.5, fontweight="bold")
    ax.grid(alpha=0.22, linestyle=":")
    ax.set_axisbelow(True)
    # legend BELOW the axes — never over the data
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, frameon=False, fontsize=8.5)

    fig.tight_layout()
    save(fig, "fig21_mcdropout_intervals_real64mt")


# ── FIG 22 panel D — CI width vs error (REAL DATA) ───────────────────────────
def fig22d_ci_width_vs_error():
    d, (gt, mu, lo, hi, w) = _mc()
    err = np.abs(mu - gt)
    r, p = stats.pearsonr(w, err)

    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    ax.scatter(w, err, s=46, facecolor=BLUE, edgecolor="#123", alpha=0.8, zorder=3)
    m, b = np.polyfit(w, err, 1)
    xs = np.linspace(w.min(), w.max(), 100)
    ax.plot(xs, m * xs + b, "--", color="#666666", lw=1.3, zorder=2)
    ax.axhline(THRESH, color=RED, ls=":", lw=1.3, label=f"threshold {THRESH:g}")

    ax.set_xlabel("MC Dropout 95% CI width (nWBV)")
    ax.set_ylabel("Absolute prediction error (nWBV)")
    ax.set_title(f"CI width vs. error\n$r = {r:+.3f}$, $p = {p:.3f}$ "
                 f"($n$={len(w)}, n.s.)", fontsize=10.5, fontweight="bold")
    ax.grid(alpha=0.22, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    fig.tight_layout()
    save(fig, "fig22D_ci_width_vs_error_CORRECTED")
    return r, p


if __name__ == "__main__":
    print("Fig 3  — three-stage pipeline (layout fix)")
    fig03_pipeline()
    print("Fig 6  — CDR boxplot (label occlusion + axis range fix)")
    fig06_cdr_boxplot()
    print("Fig 21 — MC Dropout real 64 mT (REBUILT FROM REAL DATA)")
    fig21_mc_real64mt()
    print("Fig 22D — CI width vs error (REBUILT FROM REAL DATA)")
    r, p = fig22d_ci_width_vs_error()
    print()
    print("=" * 72)
    print(f"  MANUSCRIPT CORRECTION REQUIRED")
    print(f"  reported : r = +0.34,  p = 0.108   (from np.random.normal widths)")
    print(f"  actual   : r = {r:+.3f}, p = {p:.3f}   (from mc_dropout_ci.json)")
    print(f"  The sign flips.  Any sentence arguing from a POSITIVE r is now false.")
    print("=" * 72)
    print("  Fig 20 NOT rebuilt — no OASIS-1 MC Dropout data exists in this repo.")
