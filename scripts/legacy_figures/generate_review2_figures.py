"""
Figures for the Review-2 resubmission (permutation, conformal, adapter ablation)
===============================================================================
Regenerates the three NEW manuscript figures from committed experiment outputs.
Every number plotted is recomputed here from the same committed JSON the
published results.json files were produced from — nothing is hard-coded and
nothing is simulated.

  FIG A  Permutation tests (R1.1 / R3.4 / R3.5)
         Real 20,000-permutation null distributions, reproduced with the exact
         estimator and RNG sequence of scripts/permutation_test_loocv.py.
         (A) inter-session ICC(3,1)   (B) prediction-vs-truth Pearson r

  FIG B  Split-conformal calibration (R1.7)
         Coverage swept across nominal levels with the exact leave-one-out
         split-conformal routine of scripts/conformal_calibration.py; MC-Dropout
         coverage recomputed from the committed per-subject posterior SDs.
         (A) reliability curve   (B) per-subject 90% conformal intervals

  FIG C  Adapter-strategy ablation (R2.4)
         head-only / LN+head / LoRA / full fine-tune, bootstrap 95% CIs, against
         the leave-one-out constant-mean baseline.

Outputs PNG (300 dpi) + SVG to "Review - 2/Manuscript_Revision_WIP/figures/".
"""

import json
import math
import random
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
LOOCV_JSON  = PROJECT_ROOT / "experiments" / "loocv_cross_session"   / "results.json"
PERM_JSON   = PROJECT_ROOT / "experiments" / "permutation_test"      / "results.json"
CONF_JSON   = PROJECT_ROOT / "experiments" / "conformal_calibration" / "results.json"
LORA_JSON   = PROJECT_ROOT / "experiments" / "ablation_lora"         / "results.json"
MCDROP_JSON = PROJECT_ROOT / "experiments" / "real64mt_eval"         / "mc_dropout_ci.json"
OUT_DIR     = PROJECT_ROOT / "Review - 2" / "Manuscript_Revision_WIP" / "figures"

N_PERM, SEED = 20_000, 42

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

C_MODEL, C_NULL, C_GOOD, C_BAD, C_REF = "#1f4e79", "#b8cce4", "#2e7d32", "#c62828", "#616161"


# ── estimators (identical to the committed scripts) ─────────────────────────────

def mean(x):
    return sum(x) / len(x)


def pearson(a, b):
    ma, mb = mean(a), mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da * db > 0 else 0.0


def icc31(a, b):
    """ICC(3,1): two-way mixed, single measures. Mirrors permutation_test_loocv.py."""
    n = len(a)
    data = [[a[i], b[i]] for i in range(n)]
    gm = mean([v for row in data for v in row])
    row_means = [mean(row) for row in data]
    col_means = [mean([data[i][j] for i in range(n)]) for j in range(2)]
    ss_r = 2 * sum((row_means[i] - gm) ** 2 for i in range(n))
    ss_c = n * sum((col_means[j] - gm) ** 2 for j in range(2))
    ss_t = sum((data[i][j] - gm) ** 2 for i in range(n) for j in range(2))
    ss_e = ss_t - ss_r - ss_c
    ms_r = ss_r / (n - 1)
    ms_e = ss_e / (n - 1)
    return (ms_r - ms_e) / (ms_r + ms_e) if (ms_r + ms_e) != 0 else 0.0


def conformal_halfwidth(cal_residuals, alpha):
    """Mirrors conformal_calibration.py."""
    m = len(cal_residuals)
    s = sorted(cal_residuals)
    k = math.ceil((m + 1) * (1 - alpha))
    return (s[-1], True) if k > m else (s[k - 1], False)


def conformal_evaluate(preds, trues, alpha):
    n = len(preds)
    residuals = [abs(preds[i] - trues[i]) for i in range(n)]
    out = []
    for i in range(n):
        cal = [residuals[j] for j in range(n) if j != i]
        q, _ = conformal_halfwidth(cal, alpha)
        out.append({"pred": preds[i], "true": trues[i],
                    "lo": preds[i] - q, "hi": preds[i] + q,
                    "covered": preds[i] - q <= trues[i] <= preds[i] + q})
    return out


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", {"dpi": 300}), ("svg", {})):
        p = OUT_DIR / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", facecolor="white", **kw)
        print(f"   saved  {p.relative_to(PROJECT_ROOT)}")
    plt.close(fig)


# ── FIGURE A — permutation null distributions ──────────────────────────────────

def figure_permutation(true, hfe, hfc):
    rng = random.Random(SEED)

    obs_icc = icc31(hfc, hfe)
    null_icc = []
    for _ in range(N_PERM):
        perm = hfe[:]
        rng.shuffle(perm)
        null_icc.append(icc31(hfc, perm))
    p_icc = (sum(v >= obs_icc for v in null_icc) + 1) / (N_PERM + 1)

    obs_r = pearson(hfe, true)
    null_r = []
    for _ in range(N_PERM):
        perm = true[:]
        rng.shuffle(perm)
        null_r.append(abs(pearson(hfe, perm)))
    p_r = (sum(v >= abs(obs_r) for v in null_r) + 1) / (N_PERM + 1)

    ref = json.load(open(PERM_JSON))
    print(f"   ICC  observed={obs_icc:.4f} (json {ref['test1_intersession_icc']['observed_icc']})"
          f"  p={p_icc:.5f} (json {ref['test1_intersession_icc']['p_value']})")
    print(f"   r    observed={obs_r:+.4f} (json {ref['test2_anatomical_tracking']['observed_r']})"
          f"  p={p_r:.5f} (json {ref['test2_anatomical_tracking']['p_value']})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # (A) inter-session ICC — the evidence FOR input-dependence
    n_, bins_, patches_ = ax1.hist(null_icc, bins=70, color=C_NULL,
                                   edgecolor="white", linewidth=0.25)
    for b, patch in zip(bins_[:-1], patches_):
        if b >= obs_icc:
            patch.set_facecolor(C_BAD)
    ax1.axvline(obs_icc, color=C_MODEL, lw=1.8)
    ax1.annotate(f"observed\nICC = {obs_icc:.3f}",
                 xy=(obs_icc, ax1.get_ylim()[1] * 0.62),
                 xytext=(obs_icc - 0.46, ax1.get_ylim()[1] * 0.80),
                 fontsize=8, color=C_MODEL, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_MODEL, lw=1.1))
    ax1.text(0.97, 0.96, f"$p$ = {p_icc:.4f}", transform=ax1.transAxes,
             ha="right", va="top", fontsize=9, fontweight="bold", color=C_MODEL,
             bbox=dict(boxstyle="round,pad=0.35", fc="#eaf1f8", ec=C_MODEL, lw=0.8))
    ax1.set_xlabel("ICC(3,1) under label permutation")
    ax1.set_ylabel("Permutations")
    ax1.set_title("(A) Inter-session reproducibility", fontweight="bold")

    # (B) anatomical tracking — the honest counter-evidence
    ax2.hist(null_r, bins=70, color=C_NULL, edgecolor="white", linewidth=0.25)
    ax2.axvline(abs(obs_r), color=C_REF, lw=1.8, ls="--")
    ax2.annotate(f"observed\n|r| = {abs(obs_r):.3f}",
                 xy=(abs(obs_r), ax2.get_ylim()[1] * 0.55),
                 xytext=(abs(obs_r) + 0.16, ax2.get_ylim()[1] * 0.78),
                 fontsize=8, color=C_REF, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_REF, lw=1.1))
    ax2.text(0.97, 0.96, f"$p$ = {p_r:.2f}  (n.s.)", transform=ax2.transAxes,
             ha="right", va="top", fontsize=9, fontweight="bold", color=C_REF,
             bbox=dict(boxstyle="round,pad=0.35", fc="#f2f2f2", ec=C_REF, lw=0.8))
    ax2.set_xlabel("|Pearson $r$| under label permutation")
    ax2.set_ylabel("Permutations")
    ax2.set_title("(B) Anatomical tracking", fontweight="bold")

    for ax in (ax1, ax2):
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout()
    save(fig, "fig_permutation_tests")


# ── FIGURE B — split-conformal calibration ─────────────────────────────────────

def figure_conformal(true, hfe):
    levels = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    conf_cov = []
    for lv in levels:
        res = conformal_evaluate(hfe, true, 1 - lv)
        conf_cov.append(sum(r["covered"] for r in res) / len(res))

    # MC-Dropout coverage at each level, from the committed per-subject posterior SDs
    mc = json.load(open(MCDROP_JSON))["per_subject"]
    mc_cov = []
    for lv in levels:
        z = NormalDist().inv_cdf(0.5 + lv / 2)
        hit = sum(abs(s["gt"] - s["mean"]) <= z * s["std"] for s in mc)
        mc_cov.append(hit / len(mc))

    ref = json.load(open(CONF_JSON))
    print(f"   conformal 90% = {conf_cov[4]:.4f} (json {ref['conformal']['level_90']['empirical_coverage']})")
    print(f"   conformal 95% = {conf_cov[5]:.4f} (json {ref['conformal']['level_95']['empirical_coverage']})")
    print(f"   MC-Dropout 95% = {mc_cov[5]:.4f} (json {ref['mc_dropout_baseline']['empirical_coverage']})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # (A) reliability curve
    ax1.plot([0, 1], [0, 1], ls=":", color="black", lw=1.1, label="Perfect calibration")
    ax1.plot(levels, conf_cov, "o-", color=C_GOOD, lw=1.8, ms=5,
             label="Split conformal (this work)")
    ax1.plot(levels, mc_cov, "s-", color=C_BAD, lw=1.8, ms=5,
             label="MC Dropout (manuscript)")
    ax1.annotate(f"{conf_cov[5]*100:.1f}% at 95%", xy=(0.95, conf_cov[5]), xytext=(0.79, 0.70),
                 fontsize=8, color=C_GOOD, fontweight="bold", ha="center",
                 arrowprops=dict(arrowstyle="->", color=C_GOOD, lw=1.0))
    ax1.annotate(f"{mc_cov[5]*100:.1f}% at 95%", xy=(0.95, mc_cov[5]), xytext=(0.70, 0.22),
                 fontsize=8, color=C_BAD, fontweight="bold", ha="center",
                 arrowprops=dict(arrowstyle="->", color=C_BAD, lw=1.0))
    ax1.set_xlabel("Nominal confidence level")
    ax1.set_ylabel("Empirical coverage")
    ax1.set_title("(A) Reliability curve", fontweight="bold")
    ax1.set_xlim(0.45, 1.0)
    ax1.set_ylim(-0.03, 1.05)
    ax1.legend(loc="upper left", fontsize=7.5)
    ax1.grid(alpha=0.25, lw=0.6)
    ax1.set_axisbelow(True)

    # (B) per-subject 90% conformal intervals
    res90 = conformal_evaluate(hfe, true, 0.10)
    order = sorted(range(len(res90)), key=lambda i: res90[i]["true"])
    xs = range(len(order))
    for x, i in zip(xs, order):
        r = res90[i]
        ax2.vlines(x, r["lo"], r["hi"], color=C_NULL, lw=3.4, zorder=1)
    ax2.plot(list(xs), [res90[i]["pred"] for i in order], "_", color=C_MODEL,
             ms=7, mew=1.6, label="Prediction", zorder=3)
    cov_x = [x for x, i in zip(xs, order) if res90[i]["covered"]]
    cov_y = [res90[i]["true"] for i in order if res90[i]["covered"]]
    unc_x = [x for x, i in zip(xs, order) if not res90[i]["covered"]]
    unc_y = [res90[i]["true"] for i in order if not res90[i]["covered"]]
    ax2.plot(cov_x, cov_y, "o", color=C_GOOD, ms=3.6, label="True (covered)", zorder=4)
    ax2.plot(unc_x, unc_y, "X", color=C_BAD, ms=6, label="True (not covered)", zorder=5)
    n_cov = len(cov_x)
    hi_max = max(r["hi"] for r in res90)
    lo_min = min(r["lo"] for r in res90)
    ax2.set_ylim(lo_min - 0.002, hi_max + 0.011)
    ax2.set_xlabel("Subject (sorted by true nWBV)")
    ax2.set_ylabel("nWBV")
    ax2.set_title(f"(B) 90% conformal intervals — {n_cov}/{len(res90)} covered",
                  fontweight="bold")
    ax2.legend(loc="upper center", fontsize=7, ncol=3, columnspacing=1.0,
               handletextpad=0.4, borderpad=0.35)
    ax2.grid(axis="y", alpha=0.25, lw=0.6)
    ax2.set_axisbelow(True)

    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout()
    save(fig, "fig_conformal_calibration")


# ── FIGURE C — adapter-strategy ablation ───────────────────────────────────────

def figure_adapter(true):
    tbl = json.load(open(LORA_JSON))["combined_table"]
    keys  = ["head_only", "ln_head", "lora", "full_ft"]
    names = ["Head only", "LayerNorm\n+ head", "LoRA\n($r$ = 4)", "Full\nfine-tune"]
    maes  = [tbl[k]["mae"] for k in keys]
    los   = [tbl[k]["mae_ci_95"][0] for k in keys]
    his   = [tbl[k]["mae_ci_95"][1] for k in keys]
    prms  = [tbl[k]["trainable_params"] for k in keys]

    # leave-one-out constant-mean baseline, recomputed from the LOOCV ground truth
    n = len(true)
    loo = [mean([true[j] for j in range(n) if j != i]) for i in range(n)]
    baseline = mean([abs(true[i] - loo[i]) for i in range(n)])
    ref_base = json.load(open(PERM_JSON))["constant_mean_baseline"]["leave_one_out_mean_mae"]
    print(f"   LOO constant-mean baseline = {baseline:.4f} (json {ref_base})")
    print(f"   MAEs {[f'{m:.4f}' for m in maes]}")

    def fmt_params(p):
        if p >= 1_000_000:
            return f"{p/1_000_000:.2f} M"
        if p >= 1_000:
            return f"{p/1_000:.1f} K"
        return str(p)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = range(len(keys))
    colors = [C_MODEL if k == "ln_head" else "#7fa8cc" for k in keys]
    ax.bar(x, maes, width=0.62, color=colors, edgecolor="white", lw=0.8, zorder=2)
    ax.errorbar(x, maes,
                yerr=[[m - l for m, l in zip(maes, los)],
                      [h - m for m, h in zip(maes, his)]],
                fmt="none", ecolor="#243b53", elinewidth=1.2, capsize=4, zorder=3)

    ax.axhline(baseline, color=C_REF, ls="--", lw=1.3, zorder=4,
               label=f"Constant-mean baseline ({baseline:.4f})")

    for xi, m in enumerate(maes):
        ax.text(xi, his[xi] + 0.0004, f"{m:.4f}", ha="center", fontsize=8,
                fontweight="bold", color="#243b53")

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{nm}\n{fmt_params(p)}" for nm, p in zip(names, prms)])
    ax.set_ylabel("Cross-session LOOCV MAE (nWBV)")
    ax.set_xlabel("Adapter strategy (trainable parameters)")
    ax.set_title("Adapter strategy — all 95% CIs overlap ($n$ = 23)", fontweight="bold")
    ax.set_ylim(0, 0.0215)
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.65, 3.65)
    ax.tick_params(axis="x", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    save(fig, "fig_adapter_ablation")


def main():
    recs = json.load(open(LOOCV_JSON))["per_subject"]
    true = [r["true_nwbv"] for r in recs]
    hfe  = [r["pred_hfe"]  for r in recs]
    hfc  = [r["pred_hfc"]  for r in recs]

    print("=" * 70)
    print(f"REVIEW-2 FIGURES  (n = {len(recs)} subjects)")
    print("=" * 70)
    print("\nFIG A — permutation tests")
    figure_permutation(true, hfe, hfc)
    print("\nFIG B — split-conformal calibration")
    figure_conformal(true, hfe)
    print("\nFIG C — adapter ablation")
    figure_adapter(true)
    print(f"\nAll figures written to {OUT_DIR.relative_to(PROJECT_ROOT)}/  (PNG 300 dpi + SVG)")


if __name__ == "__main__":
    main()
