"""
Rebuild fig4_age_nwbv as the two-panel figure the manuscript describes
=====================================================================
Manuscript Figure 11 (\\label{fig:age}) caption:

  (A) Ground-truth nWBV vs. age: strong decline (rho = -0.778, p < 0.001)
      reflects cohort biology.
  (B) LOOCV-adapted ViT3D predictions vs. age: no significant correlation
      (rho = +0.232, p = 0.287), indicating the model does not reliably
      capture age-related decline at n = 23.
  Both panels: real 64 mT data, n = 23.

The existing fig4_age_nwbv() in generate_all_figures.py is single-panel and
prints a HARDCODED rho = -0.597 that no data in this repo produces. It even
computes the correct value at runtime and then discards it. That literal is the
sole reason the figure and the manuscript disagreed.

EVERY STATISTIC HERE IS COMPUTED FROM THE COMMITTED DATA AT PLOT TIME.
Nothing is hardcoded. The number drawn on the figure is by construction the
number the data supports, so figure and caption cannot drift apart again.

Run on the machine with data/ds006557_data/participants.tsv:

    python scripts/rebuild_fig4_age.py

Output -> figures_rebuilt/fig4_age_nwbv.{png,svg}  (600 dpi + vector)

Sources
-------
  ages        data/ds006557_data/participants.tsv          participant_id, age
  GT nWBV     experiments/loocv_cross_session/results.json per_subject[].true_nwbv
  adapted     experiments/loocv_cross_session/results.json per_subject[].pred_hfe

NOTE: do NOT use experiments/real64mt_eval/predictions.json for panel B. That
file's age_correlation (-0.49) is computed on UN-ADAPTED oasis_finetuned
predictions — a different quantity, and the value withdrawn from the paper.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "figures_rebuilt"
DPI  = 600

LOOCV = ROOT / "experiments" / "loocv_cross_session" / "results.json"
PARTS = ROOT / "data" / "ds006557_data" / "participants.tsv"

# tab10 — matches the other 22 figures in the set
BLUE, ORANGE, GRAY = "#1f77b4", "#ff7f0e", "#7f7f7f"

# Manuscript values, for drift detection only — never plotted.
EXPECT = {"gt": -0.778, "pred": +0.232}
TOL = 0.01

plt.rcParams.update({
    "font.family":         "sans-serif",
    # DejaVu first: the set uses  ± × μ ≈ ≤ →  and Helvetica/Arial on macOS
    # render several as empty tofu boxes.
    "font.sans-serif":     ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size":           9,
    "axes.titlesize":      10,
    "axes.labelsize":      9.5,
    "legend.fontsize":     8,
    "axes.linewidth":      0.9,
    "svg.fonttype":        "none",
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
    "savefig.dpi":         DPI,
    "savefig.bbox":        "tight",
    "savefig.facecolor":   "white",
    "savefig.transparent": False,
})


def load():
    if not PARTS.exists():
        sys.exit(f"STOP: {PARTS} not found.\n"
                 "  Subject ages are real data. Do not synthesise them.\n"
                 "  Run on the machine that holds data/ds006557_data/.")

    ages = {}
    with open(PARTS) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pid = row["participant_id"]
            pid = pid if pid.startswith("sub-") else f"sub-{pid}"
            ages[pid] = float(row["age"])

    per = json.loads(LOOCV.read_text())["per_subject"]

    rows, missing = [], []
    for e in per:
        if e["subject"] in ages:
            rows.append((e["subject"], ages[e["subject"]],
                         e["true_nwbv"], e["pred_hfe"]))
        else:
            missing.append(e["subject"])

    if missing:
        print(f"  WARNING: no age for {len(missing)} subject(s): {', '.join(missing)}")
    if not rows:
        sys.exit("STOP: no subjects matched between participants.tsv and results.json.")

    rows.sort(key=lambda r: r[0])
    subj = [r[0] for r in rows]
    return subj, np.array([r[1] for r in rows]), \
           np.array([r[2] for r in rows]), np.array([r[3] for r in rows])


def panel(ax, age, y, colour, title, ylabel):
    """Scatter + least-squares trend, with Spearman rho/p computed here."""
    rho, p = stats.spearmanr(age, y)

    ax.scatter(age, y, s=46, c=colour, edgecolors="white", linewidths=0.8,
               zorder=3, alpha=0.92)
    m, b = np.polyfit(age, y, 1)
    xs = np.linspace(age.min() - 1, age.max() + 1, 100)
    ax.plot(xs, m * xs + b, "--", color=GRAY, lw=1.3, zorder=2)

    ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text(0.04, 0.05, f"ρ = {rho:+.3f}, {ptxt}\nn = {len(age)}",
            transform=ax.transAxes, fontsize=8.5, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="0.6", linewidth=0.8))

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Subject age (years)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    return rho, p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subj, age, true, pred = load()
    print(f"Loaded n = {len(subj)} subjects (age {age.min():.0f}-{age.max():.0f})")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))

    rg, pg = panel(axes[0], age, true, BLUE,
                   "(A) Ground-truth nWBV vs. age", "Ground-truth nWBV")
    rp, pp = panel(axes[1], age, pred, ORANGE,
                   "(B) LOOCV-adapted prediction vs. age", "Predicted nWBV")

    # Shared y-range so B's flatness reads visually against A's decline.
    lo = min(true.min(), pred.min()); hi = max(true.max(), pred.max())
    pad = 0.06 * (hi - lo)
    for ax in axes:
        ax.set_ylim(lo - pad, hi + pad)

    fig.tight_layout()
    fig.savefig(OUT / "fig4_age_nwbv.png", format="png", dpi=DPI)
    fig.savefig(OUT / "fig4_age_nwbv.svg", format="svg")
    plt.close(fig)

    # ── report + drift check against the manuscript ──────────────────────────
    print("\nComputed (Spearman, plotted verbatim on the figure):")
    print(f"  (A) ground-truth nWBV vs age : rho = {rg:+.4f}  p = {pg:.5f}")
    print(f"  (B) adapted pred   vs age    : rho = {rp:+.4f}  p = {pp:.5f}")

    drift = False
    for lbl, got, want in (("A", rg, EXPECT["gt"]), ("B", rp, EXPECT["pred"])):
        if abs(got - want) > TOL:
            print(f"  DRIFT: panel {lbl} computed {got:+.4f} but the manuscript "
                  f"states {want:+.3f} — the manuscript needs correcting.")
            drift = True
    if not drift:
        print("  Both match the manuscript within tolerance. No text change needed.")

    print(f"\nWrote {OUT/'fig4_age_nwbv.png'} ({DPI} dpi) + .svg")
    print("Check: SVG text selectable; no tofu in 'ρ'; both panels share a y-range.")


if __name__ == "__main__":
    main()
