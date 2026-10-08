"""
Every figure in the paper, drawn only from compute_all() output.

Rules (checked by lint.py): no random numbers, no result values typed into this
file — every plotted or printed value comes from `numbers`, `tables` or
`figdata`. Palette: validated categorical slots (blue, orange, aqua) + neutral
grey for baselines; each group also has its own marker shape, so identity is
never colour alone.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .stats import THRESHOLD  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GROUP = {"vit": ("#2a78d6", "o", "ViT3D"), "cnn": ("#eb6834", "s", "CNN3D"),
         "swin": ("#1baf7a", "D", "Swin-UNETR"), "baseline": ("#8a8984", "^", "Baseline")}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "svg.fonttype": "none",
})
COL1, COL2 = 3.5, 7.16  # IEEE single / double column width, inches


def _save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    return name


def _v(numbers, key):
    return numbers[key]["value"]


def _identity(ax, lo, hi):
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.8, color=INK2, zorder=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")


# ── simulation and pipeline ───────────────────────────────────────────────────

def fig_sensitivity(numbers, figdata, out):
    runs = figdata["misc"]["sensitivity"]["runs"]
    fig, ax = plt.subplots(figsize=(COL1, 1.9))
    labels = [r["perturbation"] for r in runs]
    d = [r["delta_mae"] for r in runs]
    ax.barh(range(len(d)), d, color=GROUP["vit"][0], height=0.6)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(d)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Change in OASIS-1 test MAE (nWBV)")
    ax.set_title(f"±20% simulation-parameter sweep, max |ΔMAE| = {_v(numbers, 'sensitivity.max_abs_delta_mae'):.4f}", loc="left")
    return _save(fig, out, "fig_simulation_sensitivity")


def fig_pipeline(numbers, figdata, out):
    fig, ax = plt.subplots(figsize=(COL2, 1.6))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 26)
    boxes = [
        (1, "Stage 1", "IXI · 104 train / 26 val", "Simulated 64 mT input\nRegress 4 segmentation-\nderived proxy targets"),
        (26, "Stage 2", "OASIS-1 · 300 / 37 / 38", "Simulated 64 mT input\nFull fine-tune to nWBV\n(OASIS-supplied labels)"),
        (51, "Stage 3", "ds006557 · n = 23", "Real 64 mT, cross-session\nFit on HFC of 22 subjects\nPredict held-out HFE"),
        (76, "External", "van den Broek · n = 10", "Real 64 mT, never seen\nReadouts unchanged"),
    ]
    for x, head, sub, body in boxes:
        ax.add_patch(plt.Rectangle((x, 1), 22.5, 24, fc="#f4f3f0", ec=INK2, lw=0.8))
        ax.text(x + 1.2, 23.2, head, fontsize=7.4, weight="bold", va="top", color=INK)
        ax.text(x + 1.2, 19.6, sub, fontsize=6.6, va="top", color=INK2)
        ax.text(x + 1.2, 14.6, body, fontsize=6.3, va="top", color=INK, linespacing=1.4)
    for x in (23.7, 48.7, 73.7):
        ax.annotate("", xy=(x + 2.1, 13), xytext=(x, 13), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9))
    return _save(fig, out, "fig_pipeline")


# ── OASIS ─────────────────────────────────────────────────────────────────────

def fig_oasis(numbers, tables, figdata, out):
    oa = figdata["oasis"]
    y = oa["y"]
    keys = [("vit3d_headline", "vit"), ("cnn3d", "cnn"), ("swin_unetr", "swin")]
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.5))
    lo, hi = min(y.min(), *(oa["models"][k]["pred"].min() for k, _ in keys)) - 0.01, max(y.max(), *(oa["models"][k]["pred"].max() for k, _ in keys)) + 0.01
    for ax, (k, g) in zip(axes, keys):
        m = oa["models"][k]
        c, mk, _ = GROUP[g]
        _identity(ax, lo, hi)
        ax.scatter(y, m["pred"], s=14, color=c, marker=mk, edgecolor="white", lw=0.5, zorder=3)
        ax.set_title(f"{m['label'].split(' (')[0]}\nMAE {m['mae_mean']:.4f} ± {m['mae_sd']:.4f}", loc="left", fontsize=7.5)
        ax.set_xlabel("True nWBV (OASIS)")
    axes[0].set_ylabel("Predicted nWBV")
    return _save(fig, out, "fig_oasis_scatter")


# ── physics vs blur ───────────────────────────────────────────────────────────

def fig_physics_blur(numbers, figdata, out):
    summ = figdata["physics_blur"]["summary"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.3), gridspec_kw={"width_ratios": [1, 1.2]})
    seeds = ["42", "1", "7"]
    for i, s in enumerate(seeds):
        r = summ["per_seed"][s]["loocv_real_64mt"]
        a.plot([r["mean_abs_err_physics"], r["mean_abs_err_blur"]], [i, i], color=INK2, lw=1, zorder=1)
        a.scatter(r["mean_abs_err_physics"], i, color=GROUP["vit"][0], marker="o", s=28, zorder=3)
        a.scatter(r["mean_abs_err_blur"], i, color=GROUP["cnn"][0], marker="s", s=28, zorder=3)
    a.set_yticks(range(3), [f"seed {s}" for s in seeds]); a.invert_yaxis()
    a.set_xlabel("Real 64 mT LOOCV MAE (nWBV)")
    a.scatter([], [], color=GROUP["vit"][0], marker="o", label="physics simulation")
    a.scatter([], [], color=GROUP["cnn"][0], marker="s", label="Gaussian blur")
    a.set_ylim(3.1, -0.5)
    a.legend(loc="lower right", fontsize=7, ncol=2)
    a.set_title(f"A  Real 64 mT: pooled Δ {_v(numbers, 'e4.pooled.real64mt.delta_mae'):+.4f}, "
                f"p = {_v(numbers, 'e4.pooled.real64mt.wilcoxon_p'):.2f}", loc="left")
    po = summ["pooled_over_seeds"]
    cats = ["Tested on physics-\nsimulated inputs", "Tested on\nblurred inputs"]
    ph = [po["oasis_test_on_physics_inputs"]["mean_abs_err_physics"], po["oasis_test_on_blur_inputs"]["mean_abs_err_physics"]]
    bl = [po["oasis_test_on_physics_inputs"]["mean_abs_err_blur"], po["oasis_test_on_blur_inputs"]["mean_abs_err_blur"]]
    x = np.arange(2)
    b.bar(x - 0.19, ph, 0.36, color=GROUP["vit"][0], label="pretrained on physics")
    b.bar(x + 0.19, bl, 0.36, color=GROUP["cnn"][0], label="pretrained on blur")
    b.set_xticks(x, cats); b.set_ylabel("OASIS-1 test MAE (nWBV)")
    b.set_ylim(0, max(ph + bl) * 1.45)
    b.legend(fontsize=7, loc="upper left")
    b.set_title("B  OASIS-1: each arm wins only on its own degradation", loc="left")
    return _save(fig, out, "fig_physics_vs_blur")


# ── real 64 mT readouts ───────────────────────────────────────────────────────

def fig_readouts(numbers, figdata, out):
    rows = figdata["readouts"]["rows"]
    keys = [k for k in rows if k != "constant_loo_mean"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 4.2), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
    for i, k in enumerate(keys):
        r = rows[k]
        c, mk, _ = GROUP[r["group"]]
        lo, hi = r["mse_skill_ci95"]
        lo_c = max(lo, -1.5)
        a.plot([lo_c, hi], [i, i], color=c, lw=1.4, solid_capstyle="round")
        if lo < -1.5:
            a.annotate("", xy=(-1.5, i), xytext=(-1.35, i), arrowprops=dict(arrowstyle="-|>", color=c, lw=1))
        if r["mse_skill"] >= -1.5:
            a.scatter(r["mse_skill"], i, color=c, marker=mk, s=26, edgecolor="white", lw=0.6, zorder=3)
        b.plot(r["mae_ci95"], [i, i], color=c, lw=1.4, solid_capstyle="round")
        b.scatter(r["mae"], i, color=c, marker=mk, s=26, edgecolor="white", lw=0.6, zorder=3)
    a.axvline(0, color=INK, lw=0.9)
    a.set_xlim(-1.55, 1)
    a.set_yticks(range(len(keys)), [rows[k]["label"] for k in keys]); a.invert_yaxis()
    a.set_xlabel("MSE skill vs leave-one-out mean (> 0 beats the constant)")
    a.set_title("A  Skill over the constant predictor [95% CI]", loc="left")
    const = rows["constant_loo_mean"]["mae"]
    b.axvline(const, color=INK, lw=0.9, ls="--", label="leave-one-out constant")
    b.axvline(THRESHOLD, color=INK2, lw=0.8, ls=":", label=f"MAE = {THRESHOLD:.3f}")
    b.set_xlabel("MAE (nWBV) [95% CI]")
    b.set_title("B  Absolute error", loc="left")
    for g in ("vit", "cnn", "swin", "baseline"):
        c, mk, lab = GROUP[g]
        b.scatter([], [], color=c, marker=mk, label=lab)
    b.legend(loc="center right", fontsize=7)
    return _save(fig, out, "fig_readouts_real64mt")


def fig_readout_scatter(numbers, figdata, out):
    ro = figdata["readouts"]
    y = ro["y"]
    keys = [("vit_ln_head_headline", "ViT3D + LN/head adapter"), ("ridge_swin_gap", "Swin-UNETR frozen + ridge"),
            ("age_only_ridge", "Age only (ridge)")]
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.5))
    lo, hi = y.min() - 0.012, y.max() + 0.012
    for ax, (k, lab) in zip(axes, keys):
        r = ro["rows"][k]
        c, mk, _ = GROUP[r["group"]]
        _identity(ax, lo, hi)
        ax.scatter(y, r["pred"], s=16, color=c, marker=mk, edgecolor="white", lw=0.5, zorder=3)
        ax.set_title(f"{lab}\nr = {r['pearson_r']:+.2f}, skill = {r['mse_skill']:+.2f}", loc="left")
        ax.set_xlabel("True nWBV (3 T FastSurfer)")
    axes[0].set_ylabel("Predicted nWBV (64 mT, HFE)")
    return _save(fig, out, "fig_readout_scatter")


def fig_reproducibility(numbers, figdata, out):
    rep = figdata["repro"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.4), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.42})
    labels = ["LOO constant\n(same fold)", "Headline\n(same fold)", "Frozen\nsingle model", "Cross-seed\n(20 pairs)"]
    vals = [1.0, _v(numbers, "repro.icc_headline_same_fold"), _v(numbers, "repro.icc_frozen_single_model"),
            _v(numbers, "repro.icc_cross_seed_mean")]
    cols = [GROUP["baseline"][0], GROUP["vit"][0], GROUP["vit"][0], GROUP["vit"][0]]
    a.bar(range(4), vals, color=cols, width=0.62)
    lo, hi = _v(numbers, "repro.icc_cross_seed_range")
    a.plot([3, 3], [lo, hi], color=INK, lw=1)
    for i, v in enumerate(vals):
        top = hi if i == 3 else v
        a.text(i, top + 0.03, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    a.set_xticks(range(4), labels, fontsize=6.8); a.set_ylim(0, 1.15); a.set_ylabel("ICC(C,1), HFC vs HFE")
    a.set_title("A  Same-fold ICC is inflated by a shared fold offset", loc="left")
    z, (_, hfe) = rep["z_hfe"], rep["headline"]
    r, p = _v(numbers, "repro.headline_pred_vs_z_offset")
    b.scatter(z, hfe, s=16, color=GROUP["vit"][0], edgecolor="white", lw=0.5)
    k, c0 = np.polyfit(z, hfe, 1)
    xx = np.linspace(z.min(), z.max(), 2)
    b.plot(xx, c0 + k * xx, color=INK2, lw=0.9)
    b.set_xlabel("Head position in field of view, superior–inferior (mm)")
    b.set_ylabel("Headline prediction (HFE)")
    b.set_title(f"B  Predictions track head position: r = {r:+.2f}, p = {p:.4f}", loc="left")
    return _save(fig, out, "fig_reproducibility")


def fig_multiseed(numbers, figdata, out):
    seeds = [42, 1, 7, 123, 2024]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL1 * 1.4, 1.8))
    a.bar(range(5), [_v(numbers, f"multiseed.seed{s}.mae") for s in seeds], color=GROUP["vit"][0], width=0.6)
    a.axhline(_v(numbers, "ds006557.constant_loo_mean.mae"), color=INK, ls="--", lw=0.9)
    a.set_xticks(range(5), seeds); a.set_ylabel("MAE"); a.set_title("A  MAE (dashed: constant)", loc="left")
    b.bar(range(5), [_v(numbers, f"multiseed.seed{s}.icc_same_fold") for s in seeds], color=GROUP["vit"][0], width=0.6)
    b.axhline(_v(numbers, "repro.icc_cross_seed_mean"), color=INK, ls="--", lw=0.9)
    b.set_xticks(range(5), seeds); b.set_ylabel("ICC(C,1)"); b.set_title("B  Same-fold ICC (dashed: cross-seed)", loc="left")
    for ax in (a, b):
        ax.set_xlabel("Seed")
    return _save(fig, out, "fig_multiseed")


def fig_conformal(numbers, figdata, out):
    cf = figdata["conformal"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.2))
    labels = [f"{lab.split(' ')[0]}\n{coh}, {lv}%" for lab, coh, lv, *_ in cf]
    cols = [GROUP["vit"][0] if lab.startswith("ViT") else GROUP["swin"][0] for lab, *_ in cf]
    x = np.arange(len(cf))
    a.bar(x, [c[3] for c in cf], color=cols, width=0.62)
    a.axhline(1, color=INK, ls="--", lw=0.9)
    a.set_xticks(x, labels, fontsize=6.2); a.set_ylabel("Interval width / cohort nWBV range")
    a.set_title("A  Width (dashed: entire cohort range)", loc="left")
    b.bar(x, [c[4] * 100 for c in cf], color=cols, width=0.62)
    for i, c in enumerate(cf):
        b.text(i, c[4] * 100 + 2, f"cov {c[5] * 100:.0f}%", ha="center", fontsize=6.2, color=INK)
    b.set_xticks(x, labels, fontsize=6.2); b.set_ylabel("Cohort inside each interval (%)"); b.set_ylim(0, 115)
    b.set_title("B  Discrimination (low = intervals separate subjects)", loc="left")
    return _save(fig, out, "fig_conformal")


def fig_external(numbers, figdata, out):
    ex = figdata["external"]
    y, P = ex["y"], ex["preds"]
    lo_ds, hi_ds = ex["ds_range"]
    order = np.argsort(y)
    fig, ax = plt.subplots(figsize=(COL2, 2.5))
    ax.axhspan(lo_ds, hi_ds, color="#efeee9", zorder=0)
    ax.text(len(y) - 0.4, hi_ds - 0.002, "ds006557 label range", ha="right", va="top", fontsize=6.8, color=INK2)
    xs = np.arange(len(y))
    ax.scatter(xs, y[order], color=INK, marker="_", s=120, lw=1.6, label="true nWBV (3 T FastSurfer)", zorder=4)
    for k, g, lab in (("vit_ln_head_seedmean", "vit", "ViT3D + adapter"), ("vit_unadapted", "baseline", "ViT3D unadapted"),
                      ("ridge_swin_gap", "swin", "Swin-UNETR + ridge")):
        c, mk, _ = GROUP[g]
        ax.scatter(xs, P[k][order], color=c, marker=mk, s=18, edgecolor="white", lw=0.5,
                   label=f"{lab}: MAE {_v(numbers, f'external.{k}.mae'):.4f}", zorder=3)
    ax.set_xticks(xs, [f"{i + 1}" for i in xs]); ax.set_xlabel("External subject (sorted by true nWBV)")
    ax.set_ylabel("nWBV")
    ax.legend(fontsize=6.8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title(f"External cohort: 0 of {_v(numbers, 'external.n')} labels inside the adaptation range; "
                 f"leave-one-out mean MAE {_v(numbers, 'external.loo_mean_mae'):.4f}", loc="left")
    return _save(fig, out, "fig_external")


def build_all(numbers, tables, figdata, out: Path):
    return [fig_pipeline(numbers, figdata, out), fig_sensitivity(numbers, figdata, out),
            fig_oasis(numbers, tables, figdata, out), fig_physics_blur(numbers, figdata, out),
            fig_readouts(numbers, figdata, out), fig_readout_scatter(numbers, figdata, out),
            fig_reproducibility(numbers, figdata, out), fig_multiseed(numbers, figdata, out),
            fig_conformal(numbers, figdata, out), fig_external(numbers, figdata, out)]
