"""
Rebuild fig22_failure_analysis (4 panels) from REAL DATA — integrity fix
========================================================================
The prior fig22_failure_analysis had a FABRICATED Panel D: its CI-width-vs-error
correlation (r = +0.34) came from np.random.normal placeholder intervals. The
real value, computed from raw per-subject MC Dropout data, is r = -0.164
(p = 0.454) — the SIGN FLIPS. Negative means wider intervals do NOT accompany
larger errors: no calibration-consistent claim.

This assembler builds all four panels from committed results.json files, with
NO np.random anywhere:
  A  Residuals vs true nWBV        — oasis_finetune/finetune_results.json
  B  MAE by nWBV tertile           — same
  C  Error vs age (LOOCV)          — loocv_cross_session + participants.tsv (real ages)
  D  CI width vs |error| (CORRECT) — real64mt_eval/mc_dropout_ci.json  (r = -0.164)

Spec: PLAN/07_FIGURE_PLAN.md (fig22, gate G4). Panel-D caption text per M6.

    python scripts/rebuild_fig22_failure_analysis.py

Output -> Review - 2/figures_to_insert/fig22_failure_analysis.{png,svg}  (600 dpi + SVG)
"""

import sys, json, csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

ROOT = Path(__file__).parent.parent
EXP = ROOT / "experiments"
OUT = ROOT / "Review - 2" / "figures_to_insert"
DPI = 600
THRESH = 0.020

BLUE, ORANGE, GREEN, RED, PURPLE, GRAY = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": DPI, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "savefig.transparent": False,
})


def main():
    # ── load real data ───────────────────────────────────────────────────────
    oasis = json.loads((EXP / "oasis_finetune" / "finetune_results.json").read_text())["per_subject_test"]
    o_true = np.array([s["true_nwbv"] for s in oasis])
    o_pred = np.array([s["pred_nwbv"] for s in oasis])
    o_res = o_pred - o_true

    loocv = json.loads((EXP / "loocv_cross_session" / "results.json").read_text())["per_subject"]
    ages = {}
    for row in csv.DictReader(open(ROOT / "data" / "ds006557_data" / "participants.tsv"), delimiter="\t"):
        ages["sub-" + row["participant_id"]] = float(row["age"])
    l_ids = [s["subject"] for s in loocv]
    l_true = np.array([s["true_nwbv"] for s in loocv])
    l_pred = np.array([s["pred_hfe"] for s in loocv])
    l_err = np.abs(l_pred - l_true)
    l_age = np.array([ages[i] for i in l_ids])

    mc = json.loads((EXP / "real64mt_eval" / "mc_dropout_ci.json").read_text())["per_subject"]
    mc = sorted(mc, key=lambda s: s["gt"])
    ci_w = np.array([s["ci_width"] for s in mc])
    ci_err = np.abs(np.array([s["mean"] for s in mc]) - np.array([s["gt"] for s in mc]))

    # ── figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.30)

    # Panel A — residuals vs true nWBV (OASIS)
    ax = fig.add_subplot(gs[0, 0])
    cols = [RED if t < 0.76 else BLUE for t in o_true]
    ax.scatter(o_true, o_res, c=cols, s=40, alpha=0.8, edgecolors="white", linewidths=0.4)
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.axhline(THRESH, color=RED, lw=1, ls=":", alpha=0.6)
    ax.axhline(-THRESH, color=RED, lw=1, ls=":", alpha=0.6)
    ax.axvline(0.76, color=ORANGE, lw=1.2, ls="--", alpha=0.7, label="nWBV=0.76 (atrophy)")
    ax.set_xlabel("True nWBV"); ax.set_ylabel("Residual (pred − true)")
    ax.set_title("(A) Residuals — OASIS Test", fontweight="bold")
    ax.legend(fontsize=8.5)

    # Panel B — MAE by nWBV tertile (OASIS)
    ax = fig.add_subplot(gs[0, 1])
    t33, t67 = np.percentile(o_true, 33), np.percentile(o_true, 67)
    masks = [o_true < t33, (o_true >= t33) & (o_true < t67), o_true >= t67]
    labels = [f"Low\n(atrophy)\n<{t33:.2f}", f"Mid\n{t33:.2f}–{t67:.2f}", f"High\n(normal)\n>{t67:.2f}"]
    maes = [np.mean(np.abs(o_pred[m] - o_true[m])) for m in masks]
    ns = [int(m.sum()) for m in masks]
    bars = ax.bar(labels, maes, color=[RED, ORANGE, GREEN], alpha=0.8, width=0.5)
    ax.axhline(THRESH, color=RED, lw=1.5, ls="--", label=f"Threshold ({THRESH:g})")
    for bar, v, n in zip(bars, maes, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{v:.3f}\n($n={n}$)", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("MAE (nWBV)"); ax.set_title("(B) MAE by nWBV Tertile", fontweight="bold")
    ax.legend(fontsize=8.5); ax.set_ylim(0, max(maes) * 1.35)

    # Panel C — error vs age, LOOCV (REAL ages)
    ax = fig.add_subplot(gs[1, 0])
    fail = l_err > THRESH
    ax.scatter(l_age[~fail], l_err[~fail], c=BLUE, s=45, alpha=0.8,
               edgecolors="white", linewidths=0.4, label="Pass (≤0.020)")
    ax.scatter(l_age[fail], l_err[fail], c=RED, s=60, marker="D", alpha=0.9,
               edgecolors="white", linewidths=0.4, label="Fail (>0.020)")
    for i, a, e in zip(l_ids, l_age, l_err):
        if e > THRESH:
            ax.annotate(i.replace("sub-HYPE", "HY"), (a, e),
                        textcoords="offset points", xytext=(5, 3), fontsize=8, color=RED)
    ax.axhline(THRESH, color=RED, lw=1.5, ls="--", alpha=0.7)
    rho, prho = stats.spearmanr(l_age, l_err)
    ax.text(0.05, 0.93, f"$\\rho={rho:.3f}$, $p={prho:.3f}$", transform=ax.transAxes,
            fontsize=9, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85))
    ax.set_xlabel("Subject Age (years)"); ax.set_ylabel("|Error| (nWBV)")
    ax.set_title(f"(C) Error vs Age — LOOCV ($n$={len(l_age)})", fontweight="bold")
    ax.legend(fontsize=8.5)

    # Panel D — CI width vs |error| (REAL DATA, r = -0.164) — sign is the point
    ax = fig.add_subplot(gs[1, 1])
    r, p = stats.pearsonr(ci_w, ci_err)
    ax.scatter(ci_w, ci_err, s=46, facecolor=PURPLE, edgecolor="#123", alpha=0.8, zorder=3)
    m, b = np.polyfit(ci_w, ci_err, 1)
    xs = np.linspace(ci_w.min(), ci_w.max(), 100)
    ax.plot(xs, m * xs + b, "--", color=GRAY, lw=1.3, zorder=2)
    ax.text(0.05, 0.93, f"$r = {r:+.3f}$, $p = {p:.3f}$\n(n.s. at $n$={len(ci_w)})",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85))
    ax.set_xlabel("MC Dropout CI Width (nWBV)"); ax.set_ylabel("|Error| (nWBV)")
    ax.set_title("(D) CI Width vs Error\n(Calibration Check)", fontweight="bold")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig22_failure_analysis.png", format="png", dpi=DPI)
    fig.savefig(OUT / "fig22_failure_analysis.svg", format="svg")
    plt.close(fig)

    print("Rebuilt fig22_failure_analysis (4 panels, real data)")
    print(f"  Panel C: Spearman rho={rho:.3f}, p={prho:.3f} (real ages, n={len(l_age)})")
    print(f"  Panel D: r={r:+.4f}, p={p:.4f}  <-- CORRECTED (was fabricated +0.34)")
    print(f"  Saved -> {OUT/'fig22_failure_analysis.png'} + .svg")
    assert r < 0, "Panel D correlation must be NEGATIVE — abort if not."
    print("  Sign check: r < 0 confirmed (wider intervals do NOT track larger errors).")


if __name__ == "__main__":
    main()
