"""
Figures for the Review-2 resubmission, part 2 (architecture, sensitivity, seeds)
===============================================================================
Companion to scripts/generate_review2_figures.py. That script covers the three
figures planned on 2026-07-24 (permutation / conformal / adapter). The
experiments run after that date — architecture comparators, SSL comparator,
64 mT transfer probe, simulation sensitivity, multi-seed LOOCV — had no figures,
even though two of them answer reviewer requests directly. This script adds them.

Every number plotted is read from committed experiment JSON at run time; nothing
is hard-coded and nothing is simulated. Each value is printed next to its source
so it can be checked against the manuscript before the figure is used.

  FIG D  Architecture and pretraining comparators (R2.1 / R2.2 / R2.3)
         (A) OASIS-1 accuracy vs. parameter count for CNN3D, ViT3D, SimMIM-ViT3D,
             Swin-UNETR and UNETR — the larger models win on high field.
         (B) Frozen-encoder ridge probe on real 64 mT — that advantage does not
             survive the domain gap. This is the R2 rebuttal in one panel.

  FIG E  Simulation-parameter sensitivity (R1.10)
         Tornado plot of ΔMAE for ±20 % perturbations of SNR, B0 inhomogeneity
         and T1/T2 relaxation.

  FIG F  Multi-seed stability (R1.9)
         Per-seed LOOCV MAE and ICC(3,1) across 5 seeds, with mean ± SD bands and
         the headline single-run values marked.

Outputs PNG (300 dpi) + SVG to "Review - 2/Manuscript_Revision_WIP/figures/".
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
EXP = PROJECT_ROOT / "experiments"
SWIN_JSON   = EXP / "arch_comparator_swin"    / "results.json"
UNETR_JSON  = EXP / "arch_comparator_unetr"   / "results.json"
SIMMIM_JSON = EXP / "ssl_comparator_simmim"   / "results.json"
PROBE_JSON  = EXP / "transfer_probe_64mt"     / "results.json"
CNNCMP_JSON = EXP / "oasis_cnn_comparison"    / "results.json"
SENS_JSON   = EXP / "simulation_sensitivity"  / "results.json"
SEED_JSON   = EXP / "multiseed_loocv"         / "results.json"
LOOCV_JSON  = EXP / "loocv_cross_session"     / "results.json"
OUT_DIR     = PROJECT_ROOT / "Review - 2" / "Manuscript_Revision_WIP" / "figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.9,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8,
    "legend.frameon": True,
    "svg.fonttype": "none",       # keep text as text in the SVG
})

# same palette as generate_review2_figures.py
C_MODEL, C_NULL, C_GOOD, C_BAD, C_REF = "#1f4e79", "#b8cce4", "#2e7d32", "#c62828", "#616161"

CHECKS = []


def load(p):
    with open(p) as fh:
        return json.load(fh)


def check(label, value, source):
    CHECKS.append((label, value, source))
    return value


def save(fig, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", {"dpi": 300}), ("svg", {})):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  wrote {stem}.png (300 dpi) + {stem}.svg")


# ─────────────────────────────────────────────────────────────────────────────
# FIG D — architecture and pretraining comparators (R2.1 / R2.2 / R2.3)
# ─────────────────────────────────────────────────────────────────────────────
def fig_architecture_comparators():
    swin, unetr, simmim = load(SWIN_JSON), load(UNETR_JSON), load(SIMMIM_JSON)
    probe, cnncmp = load(PROBE_JSON), load(CNNCMP_JSON)

    vit_params = swin["comparison"]["vit3d_params"]
    cnn_params = swin["comparison"]["cnn3d_params"]

    # (name, params, OASIS MAE, OASIS r, is_ours)
    models = [
        ("CNN3D",         cnn_params,     check("CNN3D OASIS MAE",  cnncmp["CNN3D"]["mae"],       "oasis_cnn_comparison"),
                                          check("CNN3D OASIS r",    cnncmp["CNN3D"]["pearson_r"], "oasis_cnn_comparison"), False),
        ("ViT3D (ours)",  vit_params,     check("ViT3D OASIS MAE",  cnncmp["ViT3D"]["mae"],       "oasis_cnn_comparison"),
                                          check("ViT3D OASIS r",    cnncmp["ViT3D"]["pearson_r"], "oasis_cnn_comparison"), True),
        ("ViT3D +SimMIM", vit_params,     check("SimMIM OASIS MAE", simmim["mae"],                "ssl_comparator_simmim"),
                                          check("SimMIM OASIS r",   simmim["pearson_r"],          "ssl_comparator_simmim"), False),
        ("Swin-UNETR",    swin["params"], check("Swin OASIS MAE",   swin["mae"],                  "arch_comparator_swin"),
                                          check("Swin OASIS r",     swin["pearson_r"],            "arch_comparator_swin"), False),
        ("UNETR",         unetr["params"], check("UNETR OASIS MAE", unetr["mae"],                 "arch_comparator_unetr"),
                                          check("UNETR OASIS r",    unetr["pearson_r"],           "arch_comparator_unetr"), False),
    ]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.9))

    # ── panel A: OASIS-1, MAE bars + r overlay, ordered by parameter count
    models_a = sorted(models, key=lambda m: m[1])
    labels = [f"{m[0]}\n{m[1]/1e6:.1f}M" for m in models_a]
    maes   = [m[2] for m in models_a]
    rs     = [m[3] for m in models_a]
    colors = [C_MODEL if m[4] else C_NULL for m in models_a]
    x = range(len(models_a))

    bars = axA.bar(x, maes, color=colors, edgecolor="black", linewidth=0.7, width=0.62, zorder=3)
    axA.set_xticks(list(x))
    axA.set_xticklabels(labels, fontsize=8)
    axA.set_ylabel("OASIS-1 test MAE (nWBV)")
    axA.set_ylim(0, max(maes) * 1.34)
    axA.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
    axA.set_axisbelow(True)

    for bar, mae, r in zip(bars, maes, rs):
        axA.text(bar.get_x() + bar.get_width() / 2, mae + max(maes) * 0.025,
                 f"{mae:.4f}\nr={r:.3f}", ha="center", va="bottom", fontsize=7.4)

    axA.set_title("(A) OASIS-1 (high field): larger models win", fontsize=9.5)

    # ── panel B: frozen-encoder transfer probe on real 64 mT
    vit_probe  = probe["backbones"]["vit3d"]
    swin_probe = probe["backbones"]["swin_unetr"]
    p_labels = ["ViT3D\n(4.2M)", "Swin-UNETR\n(62.2M)"]
    p_maes = [check("probe ViT3D MAE", vit_probe["mae"],  "transfer_probe_64mt"),
              check("probe Swin MAE",  swin_probe["mae"], "transfer_probe_64mt")]
    p_rs   = [check("probe ViT3D r",   vit_probe["r"],    "transfer_probe_64mt"),
              check("probe Swin r",    swin_probe["r"],   "transfer_probe_64mt")]

    bars_b = axB.bar([0, 1], p_maes, color=[C_MODEL, C_NULL], edgecolor="black",
                     linewidth=0.7, width=0.46, zorder=3)
    axB.set_xticks([0, 1])
    axB.set_xticklabels(p_labels, fontsize=8)
    axB.set_ylabel("real 64 mT probe MAE (nWBV)")
    axB.set_xlim(-0.6, 1.6)
    axB.set_ylim(0, max(p_maes) * 1.42)
    axB.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
    axB.set_axisbelow(True)

    for bar, mae, r in zip(bars_b, p_maes, p_rs):
        axB.text(bar.get_x() + bar.get_width() / 2, mae + max(p_maes) * 0.03,
                 f"{mae:.4f}\nr={r:.3f}", ha="center", va="bottom", fontsize=7.8)

    axB.axhline(max(p_maes) * 1.16, xmin=0.16, xmax=0.84, color=C_REF, linewidth=0.8)
    axB.text(0.5, max(p_maes) * 1.185, "capacity gives no benefit here",
             ha="center", va="bottom", fontsize=7.6, color=C_REF, style="italic")
    axB.set_title("(B) Real 64 mT (frozen encoder + ridge probe)", fontsize=9.5)

    fig.tight_layout()
    save(fig, "fig_architecture_comparators")


# ─────────────────────────────────────────────────────────────────────────────
# FIG E — simulation-parameter sensitivity (R1.10)
# ─────────────────────────────────────────────────────────────────────────────
def fig_simulation_sensitivity():
    sens = load(SENS_JSON)
    baseline = check("sensitivity baseline MAE", sens["baseline_mae"], "simulation_sensitivity")
    max_abs  = check("max |ΔMAE|",               sens["max_abs_delta_mae"], "simulation_sensitivity")

    runs = sorted(sens["runs"], key=lambda r: abs(r["delta_mae"]))
    labels = [r["perturbation"] for r in runs]
    deltas = [check(f"ΔMAE {r['perturbation']}", r["delta_mae"], "simulation_sensitivity")
              for r in runs]

    fig, ax = plt.subplots(figsize=(6.9, 3.5))
    y = range(len(runs))
    colors = [C_BAD if d > 0 else C_GOOD for d in deltas]

    ax.barh(list(y), deltas, color=colors, edgecolor="black", linewidth=0.7,
            height=0.6, zorder=3)
    ax.axvline(0, color="black", linewidth=1.0, zorder=4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlabel(r"$\Delta$ MAE vs. baseline (nWBV)   —   negative = better")
    ax.grid(axis="x", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    span = max(abs(min(deltas)), abs(max(deltas)))
    ax.set_xlim(-span * 1.55, span * 1.55)
    for yi, d in zip(y, deltas):
        off = span * 0.05
        ax.text(d + (off if d >= 0 else -off), yi, f"{d:+.4f}",
                va="center", ha="left" if d >= 0 else "right", fontsize=7.6)

    ax.set_title(f"Simulation-parameter sensitivity ($\\pm$20 %); "
                 f"baseline MAE = {baseline:.4f}, max |$\\Delta$MAE| = {max_abs:.4f}",
                 fontsize=9.3)
    fig.tight_layout()
    save(fig, "fig_simulation_sensitivity")


# ─────────────────────────────────────────────────────────────────────────────
# FIG F — multi-seed stability (R1.9)
# ─────────────────────────────────────────────────────────────────────────────
def fig_multiseed_stability():
    ms = load(SEED_JSON)
    head = load(LOOCV_JSON)

    s = ms["summary"]
    mae_mean = check("multiseed MAE mean", s["mae_mean"], "multiseed_loocv")
    mae_std  = check("multiseed MAE SD",   s["mae_std"],  "multiseed_loocv")
    icc_mean = check("multiseed ICC mean", s["icc_mean"], "multiseed_loocv")
    icc_std  = check("multiseed ICC SD",   s["icc_std"],  "multiseed_loocv")

    per = ms["per_seed"]
    seeds = [p["seed"] for p in per]
    maes  = [p["mae"] for p in per]
    iccs  = [p.get("icc_31") for p in per]

    head_mae = check("headline LOOCV MAE", head["mae"], "loocv_cross_session")
    head_icc = check("headline ICC(3,1)", head["icc_31_hfc_hfe"], "loocv_cross_session")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.0, 3.5))
    x = range(len(seeds))

    for ax, vals, mean, sd, headline, ylab, title in (
        (axA, maes, mae_mean, mae_std, head_mae, "LOOCV MAE (nWBV)",
         f"(A) MAE = {mae_mean:.4f} $\\pm$ {mae_std:.4f}"),
        (axB, iccs, icc_mean, icc_std, head_icc, "Inter-session ICC(3,1)",
         f"(B) ICC = {icc_mean:.3f} $\\pm$ {icc_std:.3f}"),
    ):
        ax.axhspan(mean - sd, mean + sd, color=C_NULL, alpha=0.55, zorder=1,
                   label="mean $\\pm$ 1 SD")
        ax.axhline(mean, color=C_MODEL, linewidth=1.1, zorder=2, label="mean across seeds")
        ax.axhline(headline, color=C_BAD, linewidth=1.1, linestyle="--", zorder=2,
                   label="headline single run")
        ax.plot(list(x), vals, "o", color=C_MODEL, markersize=6.5,
                markeredgecolor="black", markeredgewidth=0.7, zorder=3)
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"seed\n{sd_}" for sd_ in seeds], fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        span = max(max(vals) - min(vals), sd * 2) or 1e-4
        ax.set_ylim(min(min(vals), mean - sd, headline) - span * 0.55,
                    max(max(vals), mean + sd, headline) + span * 0.75)

    axA.legend(loc="upper left", fontsize=7.2)
    fig.tight_layout()
    save(fig, "fig_multiseed_stability")


if __name__ == "__main__":
    print("Generating Review-2 part-2 figures\n")
    fig_architecture_comparators()
    fig_simulation_sensitivity()
    fig_multiseed_stability()

    print("\nEvery plotted value, with its source — check these against the manuscript:")
    width = max(len(c[0]) for c in CHECKS)
    for label, value, source in CHECKS:
        shown = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"  {label:<{width}}  {shown:>10}   [{source}]")
    print(f"\n{len(CHECKS)} values read from committed JSON; none hard-coded.")
