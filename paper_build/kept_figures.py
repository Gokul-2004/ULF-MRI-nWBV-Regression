"""
Figures 2-18 and 20-23 of the manuscript, in the layout of the round-2 figures,
drawn only from kept.compute_kept() output and the shared numbers dict.

Labels that the revision corrected are corrected here too: OASIS-1 labels are
"OASIS-supplied" (not FreeSurfer/FastSurfer), 0.020 nWBV is a "0.020 reference"
(not a clinical threshold), ICC is ICC(C,1), Stage 1 is proxy-target regression
on generically degraded inputs, and every time is the measured one.

Written to out/figures/manuscript/Figure_NN_<name>.png (600 dpi) and .svg.
Fig. 1 (MRI slices from the raw scans) and Fig. 19 (OASIS MC Dropout, per-subject
values not saved) are not rebuilt; Fig. 24 is figures.fig24_external.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from .stats import THRESHOLD  # noqa: E402

BLUE, ORANGE, GREEN, RED, PURPLE, NAVY, GREY = "#3a86c3", "#f7943b", "#4cae4c", "#d9443f", "#9b6fc3", "#1f4e79", "#616161"
LIGHT = "#b8cce4"
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 9,
         "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 9.5,
         "axes.labelsize": 9, "legend.fontsize": 7.5, "legend.frameon": True, "savefig.dpi": 600,
         "savefig.bbox": "tight", "svg.fonttype": "none", "axes.grid": False}
REF_LABEL = "0.020 reference"


def _save(fig, out, name):
    d = out / "manuscript"
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(d / f"{name}.{ext}")
    plt.close(fig)
    return f"manuscript/{name}"


def _v(numbers, key):
    return numbers[key]["value"]


def _ref(ax, horizontal=True):
    (ax.axhline if horizontal else ax.axvline)(THRESHOLD, color=RED, ls="--", lw=1.2, label=REF_LABEL)


def _box(ax, text, loc=(0.03, 0.97)):
    ax.text(*loc, text, transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.6))


# ── Fig. 2 ─────────────────────────────────────────────────────────────────────

def fig02(numbers, k, out):
    sens = k["sensitivity"]
    runs = sorted(sens["runs"], key=lambda r: -abs(r["delta_mae"]))
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    d = [r["delta_mae"] for r in runs]
    ax.barh(range(len(d)), d, color=[RED if x > 0 else GREEN for x in d], height=0.6)
    for i, x in enumerate(d):
        ax.text(x + (1 if x >= 0 else -1) * max(abs(v) for v in d) / 30, i, f"{x:+.4f}", va="center", ha="left" if x >= 0 else "right", fontsize=7.5)
    ax.axvline(0, color="black", lw=0.9)
    ax.set_yticks(range(len(d)), [r["perturbation"] for r in runs]); ax.invert_yaxis()
    lim = max(abs(x) for x in d) * 1.6
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Δ MAE vs. baseline (nWBV)  —  negative = better")
    ax.set_title(f"Simulation-parameter sensitivity (±20 %); baseline MAE = {sens['baseline_mae']:.4f}, "
                 f"max |ΔMAE| = {sens['max_abs_delta_mae']:.4f}", fontweight="normal")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, out, "Figure_02_simulation_parameter_sensitivity")


# ── Fig. 3 ─────────────────────────────────────────────────────────────────────

def fig03(numbers, k, out):
    fig, ax = plt.subplots(figsize=(7.4, 1.75))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(1.5, 30)
    fwd, e2e = _v(numbers, "time.forward_ms"), _v(numbers, "time.end_to_end_ms")
    boxes = [("STAGE 1", BLUE, "IXI\nPre-training\n(proxy targets)", "Generic degradation;\n104 training volumes"),
             ("STAGE 2", ORANGE, "OASIS-1\nFine-tuning\n(n=375)", "Physics-simulated input;\nOASIS-supplied nWBV"),
             ("STAGE 3", GREEN, "LOOCV\nAdaptation\n(n=23)", "Cross-session;\nFastSurfer labels (3 T)"),
             ("", PURPLE, "ViT3D\n4.23M\nparams", "64³ → nWBV;\nLayerNorm + head (769)"),
             ("", RED, f"Inference\n{fwd:.2f} ms\n(CPU)", f"forward pass;\n{e2e:.0f} ms end-to-end")]
    w, gap = 15.0, 5.2
    for i, (stage, col, body, sub) in enumerate(boxes):
        x = 1 + i * (w + gap)
        if stage:
            ax.text(x + w / 2, 27.5, stage, ha="center", fontsize=8, weight="bold", color=col)
        ax.add_patch(FancyBboxPatch((x, 10.5), w, 13, boxstyle="square,pad=0", fc=col, ec="none"))
        ax.text(x + w / 2, 17, body, ha="center", va="center", fontsize=7.2, weight="bold", color="white", linespacing=1.3)
        ax.text(x + w / 2, 7.8, sub, ha="center", va="top", fontsize=6.5, style="italic", color=GREY, linespacing=1.3)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + w + gap - 0.6, 17), xytext=(x + w + 0.6, 17), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.4))
    ax.set_title("Three-Stage Training Pipeline", fontsize=11)
    return _save(fig, out, "Figure_03_three_stage_training_pipeline")


# ── Fig. 4 ─────────────────────────────────────────────────────────────────────

def _cdr_key(c):
    return "None" if c is None else f"{float(c):.1f}"


def fig04(numbers, k, out):
    o = k["oasis"]
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    cols = {"None": BLUE, "0.5": ORANGE, "0.0": GREEN, "1.0": RED}
    keys = [_cdr_key(c) for c in o["cdr"]]
    for key in ("None", "0.5", "0.0", "1.0"):
        m = np.array([q == key for q in keys])
        ax.scatter(o["y"][m], o["pred"][m], s=26, color=cols[key], edgecolor="white", lw=0.5, label=f"CDR: {key}", zorder=3)
    lo, hi = 0.635, 0.88
    ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=0.9, label="Identity")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    _box(ax, f"r = {o['r']:.3f} [{o['r_ci'][0]:.3f}–{o['r_ci'][1]:.3f}]\nMAE = {o['mae']:.3f} [{o['mae_ci'][0]:.3f}–{o['mae_ci'][1]:.3f}]")
    ax.legend(loc="lower right")
    ax.set_xlabel("True nWBV (OASIS-supplied)"); ax.set_ylabel("Predicted nWBV (ViT3D)")
    ax.set_title(f"OASIS-1 Test Set (n = {len(o['y'])})")
    return _save(fig, out, "Figure_04_predicted_vs_true_oasis")


# ── Fig. 5 ─────────────────────────────────────────────────────────────────────

def fig05(numbers, k, out):
    c = k["cdr"]
    groups = [(0.0, "CDR 0.0", "#64b5f6"), (0.5, "CDR 0.5", "#2f7fd0"), (1.0, "CDR 1.0", NAVY)]
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    data = [c["truth_by_cdr"][g] for g, _, _ in groups]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, medianprops=dict(color="white", lw=1.6))
    for patch, (_, _, col) in zip(bp["boxes"], groups):
        patch.set_facecolor(col); patch.set_edgecolor(col)
    for i, d in enumerate(data):
        x = np.linspace(-0.12, 0.12, len(d)) + i + 1
        ax.scatter(x, d, s=22, facecolor="white", edgecolor=groups[i][2], lw=1.1, zorder=3)
        ax.text(i + 1, 0.83, f"μ = {np.mean(d):.3f}", ha="center", fontsize=9, color="white", weight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc=groups[i][2], ec="none"))
    ax.plot([2, 2, 3, 3], [0.855, 0.86, 0.86, 0.855], color="black", lw=0.9)
    ax.text(2.5, 0.864, f"p = {c['mw_p']:.3f} (CDR 0.5 vs 1.0, one-sided)", ha="center", fontsize=7.5)
    ax.set_xticks([1, 2, 3], [f"{lab}\n(n = {len(d)})" for (_, lab, _), d in zip(groups, data)])
    ax.set_ylim(0.63, 0.9)
    ax.set_ylabel("True nWBV (OASIS-supplied)")
    ax.set_title(f"nWBV by CDR Stage — OASIS-1 Test Set (n = {sum(len(d) for d in data)})")
    ax.text(0.5, -0.2, "Literature: 5–10% nWBV reduction per CDR stage", transform=ax.transAxes, ha="center", fontsize=7.5,
            style="italic", bbox=dict(boxstyle="round,pad=0.3", fc="#eef2f8", ec="#9fb3d1"))
    return _save(fig, out, "Figure_05_true_nwbv_by_cdr_stage")


# ── Fig. 6 ─────────────────────────────────────────────────────────────────────

def fig06(numbers, k, out):
    b = k["blur"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
    for ax, key, title, col in ((axes[0], "physics_sim", "Physics sim", BLUE), (axes[1], "gaussian_blur", "Gaussian blur", ORANGE)):
        a = b[key]
        ax.scatter(a["true"], a["pred"], s=14, color=col, edgecolor="white", lw=0.4, zorder=3)
        lo, hi = min(a["true"].min(), a["pred"].min()) - 0.01, max(a["true"].max(), a["pred"].max()) + 0.01
        ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=0.8); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"{title}\nr = {a['r']:.3f} [{a['r_ci'][0]:.3f}–{a['r_ci'][1]:.3f}]  MAE = {a['mae']:.3f}", fontsize=8)
        ax.set_xlabel("True nWBV"); ax.set_ylabel("Predicted nWBV")
    dr = b["physics_sim"]["r"] - b["gaussian_blur"]["r"]
    fig.suptitle(f"Physics Simulation vs. Gaussian Blur Pre-training (earlier unpaired comparison)\n"
                 f"(n_test = {len(b['physics_sim']['true'])}, separate test splits; Δr = {dr:+.3f}; Mann–Whitney p = {b['mw_p']:.2f})",
                 fontsize=9, weight="bold")
    fig.tight_layout()
    return _save(fig, out, "Figure_06_physics_vs_gaussian_blur")


# ── Figs. 7 and 8 ──────────────────────────────────────────────────────────────

def fig07(numbers, k, out):
    u = k["unadapted"]
    labels = ["SynthSeg+", "CNN3D\n(no adapt.)", "ViT3D\n(no adapt.)"]
    keys, cols = ["synthseg", "CNN3D", "ViT3D"], [GREEN, ORANGE, BLUE]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.1))
    m = [u[x]["mae"] for x in keys]; r = [u[x]["r"] for x in keys]
    a.bar(labels, m, color=cols, width=0.55); _ref(a); a.legend(loc="upper right")
    for i, v in enumerate(m):
        a.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
    a.set_ylabel("MAE (nWBV)"); a.set_title("Mean Absolute Error"); a.set_ylim(0, 0.1)
    b.bar(labels, r, color=cols, width=0.55)
    for i, v in enumerate(r):
        b.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    b.set_ylabel("Pearson r"); b.set_title("Pearson Correlation"); b.set_ylim(0, 1)
    fig.suptitle(f"Method Comparison on Real 64 mT Hardware (n = {numbers['ds006557.n']['value']})", fontsize=10, weight="bold")
    fig.tight_layout()
    return _save(fig, out, "Figure_07_method_comparison_real64mt")


def fig08(numbers, k, out):
    u = k["unadapted"]; ds = k["ds"]; lat = u["latency_ms"]
    labels = ["SynthSeg+", "CNN3D\n(no adapt.)", "ViT3D\n(no adapt.)", "ViT3D\n(LOOCV adapt.)"]
    cols = ["#d0d0d0", LIGHT, "#8e24aa", NAVY]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    m = [u["synthseg"]["mae"], u["CNN3D"]["mae"], u["ViT3D"]["mae"], ds["mae"]]
    a.bar(range(4), m, color=cols, edgecolor="black", lw=0.5)
    for i, v in enumerate(m):
        a.text(i, v + max(m) / 50, f"{v:.4f}", ha="center", fontsize=6.8)
    a.set_xticks(range(4), labels, fontsize=7); a.set_ylabel("real 64 mT MAE (nWBV)", fontsize=8)
    a.set_title("(A) Accuracy on real 64 mT hardware", fontsize=8, loc="left", fontweight="normal")
    t = [lat["synthseg"], lat["cnn"], lat["vit"], lat["vit"]]
    b.bar(range(4), t, color=cols, edgecolor="black", lw=0.5)
    txt = [f"{_v(numbers, 'kept.latency.synthseg_min'):.1f} min", f"{lat['cnn']:.1f} ms", f"{lat['vit']:.2f} ms", f"{lat['vit']:.2f} ms"]
    for i, (v, s) in enumerate(zip(t, txt)):
        b.text(i, v * 1.4, s + "\n(measured)", ha="center", fontsize=6.5)
    b.set_yscale("log"); b.set_ylim(1, 1e7)
    b.set_xticks(range(4), labels, fontsize=7); b.set_ylabel("time per volume (ms, log scale)", fontsize=8)
    b.set_title("(B) Inference cost", fontsize=8, loc="left", fontweight="normal")
    fig.text(0.5, -0.04, "All times measured on the study CPU (Intel Core i7-8750H, 6 threads). SynthSeg+: median per scan with 4 jobs in parallel; "
             "CNN3D and ViT3D: cold full-volume forward pass, batch 1.", ha="center", fontsize=6.3, color=GREY)
    fig.tight_layout()
    return _save(fig, out, "Figure_08_fourway_comparison_with_latency")


# ── Figs. 9-11 ─────────────────────────────────────────────────────────────────

def fig09(numbers, k, out):
    d = k["ds"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, yv, col, title, res in ((a, d["y"], BLUE, "(A) Ground-truth nWBV vs. age", d["age_truth"]),
                                    (b, d["hfe"], ORANGE, "(B) LOOCV-adapted prediction vs. age", d["age_pred"])):
        ax.scatter(d["age"], yv, s=16, color=col, zorder=3)
        sl, ic = np.polyfit(d["age"], yv, 1)
        xs = np.array([d["age"].min(), d["age"].max()])
        ax.plot(xs, sl * xs + ic, "--", color="grey", lw=0.9)
        pt = "p < 0.001" if res.pvalue < 0.001 else f"p = {res.pvalue:.3f}"
        ax.text(0.03, 0.04, f"ρ = {res.statistic:+.3f}, {pt}\nn = {len(yv)}", transform=ax.transAxes, fontsize=7,
                bbox=dict(boxstyle="round", fc="white", ec="#999999"))
        ax.set_title(title, fontsize=8); ax.set_xlabel("Subject age (years)", fontsize=8)
    a.set_ylabel("Ground-truth nWBV", fontsize=8); b.set_ylabel("Predicted nWBV", fontsize=8)
    b.yaxis.set_tick_params(labelleft=True)
    fig.tight_layout()
    return _save(fig, out, "Figure_09_age_vs_nwbv")


def fig10(numbers, k, out):
    d = k["ds"]
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    lo, hi = 0.745, 0.813
    xs = np.array([lo, hi])
    ax.fill_between(xs, xs - d["mae"], xs + d["mae"], color=PURPLE, alpha=0.1, label=f"±MAE ({d['mae']:.4f})")
    ax.plot(xs, xs, "--", color="grey", lw=0.9, label="Identity")
    ok = d["err"] <= THRESHOLD
    ax.scatter(d["y"][ok], d["hfe"][ok], s=34, color=PURPLE, label=f"|error| ≤ 0.020 ({ok.sum()}/{len(ok)})", zorder=3)
    ax.scatter(d["y"][~ok], d["hfe"][~ok], s=50, marker="D", color=RED, label=f"|error| > 0.020 ({(~ok).sum()}/{len(ok)})", zorder=3)
    for i in np.where(~ok)[0]:
        ax.annotate(d["subjects"][i].replace("sub-HYPE", "HY"), (d["y"][i], d["hfe"][i]), xytext=((5, -9) if d["y"][i] > 0.78 else (4, 6)),
                    textcoords="offset points", color=RED, fontsize=7.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    _box(ax, f"MAE = {d['mae']:.4f} [{d['mae_ci'][0]:.3f}–{d['mae_ci'][1]:.3f}]\nBias = {d['bias']:+.4f}\nsame-fold ICC(C,1) = {d['icc']:.3f}")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlabel("True nWBV (FastSurfer, 3 T)"); ax.set_ylabel("Predicted nWBV (LOOCV-adapted ViT3D)")
    ax.set_title(f"Cross-Session LOOCV — Real 64 mT (n = {len(d['y'])})\nTrain on HFC of 22 subjects → Test on HFE of held-out", fontsize=9)
    return _save(fig, out, "Figure_10_cross_session_loocv_scatter")


def fig11(numbers, k, out):
    d = k["ds"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.1))
    a.hist(d["boot"], bins=50, color="#4e9ac7", edgecolor="white", lw=0.3)
    a.axvline(d["mae"], color=NAVY, lw=1.4, label=f"MAE = {d['mae']:.4f}")
    for x in d["mae_ci"]:
        a.axvline(x, color=NAVY, ls="--", lw=1)
    a.plot([], [], ls="--", color=NAVY, label=f"95% CI [{d['mae_ci'][0]:.3f}–{d['mae_ci'][1]:.3f}]")
    a.axvline(THRESHOLD, color=RED, lw=1.2, label=REF_LABEL)
    a.set_xlim(0, 0.021); a.legend(loc="center left", fontsize=6.5)
    a.set_xlabel("Bootstrap MAE (nWBV)", fontsize=8); a.set_ylabel("Frequency", fontsize=8)
    a.set_title(f"(A) Bootstrap MAE Distribution\nLOOCV n = {len(d['y'])}, {len(d['boot']):,} resamples", fontsize=8)
    ba = d["ba"]
    out_loa = (ba["diff"] < ba["loa"][0]) | (ba["diff"] > ba["loa"][1])
    b.scatter(ba["mean"][~out_loa], ba["diff"][~out_loa], s=16, color=BLUE, label="Within LoA")
    b.scatter(ba["mean"][out_loa], ba["diff"][out_loa], s=30, marker="D", color=RED, label="Outside LoA")
    for i in np.where(out_loa)[0]:
        b.annotate(d["subjects"][i].replace("sub-HYPE", "HY"), (ba["mean"][i], ba["diff"][i]), xytext=(4, 2), textcoords="offset points", fontsize=6.5, color=RED)
    b.axhline(ba["bias"], color=BLUE, lw=1.2, label=f"Bias = {ba['bias']:+.4f}")
    for x, lab in zip(ba["loa"], ("−1.96SD", "+1.96SD")):
        b.axhline(x, color=GREY, ls="--", lw=0.9, label=f"{lab} = {x:+.4f}")
    b.legend(fontsize=6.3, loc="upper right")
    b.set_xlabel("Mean of Predicted and True nWBV", fontsize=8); b.set_ylabel("Predicted − True nWBV", fontsize=8)
    b.set_title("(B) Bland–Altman Analysis\nLOOCV-adapted ViT3D", fontsize=8)
    fig.tight_layout()
    return _save(fig, out, "Figure_11_bootstrap_mae_bland_altman")


# ── Figs. 12-14 ────────────────────────────────────────────────────────────────

def fig12(numbers, k, out):
    pm = k["ds"]["perm"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    a.hist(pm["null_icc"], bins=60, color=LIGHT, edgecolor="white", lw=0.2)
    a.axvline(pm["obs_icc"], color=NAVY, lw=1.6)
    a.annotate(f"observed\nICC = {pm['obs_icc']:.3f}", xy=(pm["obs_icc"], a.get_ylim()[1] * 0.6), xytext=(-60, 15), textcoords="offset points",
               fontsize=7.5, color=NAVY, weight="bold", arrowprops=dict(arrowstyle="->", color=NAVY))
    a.text(0.97, 0.95, f"p = {pm['p_icc']:.4f}", transform=a.transAxes, ha="right", va="top", fontsize=8, weight="bold", color=NAVY,
           bbox=dict(boxstyle="round", fc="white", ec=NAVY))
    a.set_xlabel("ICC(C,1) under label permutation (same fold model)", fontsize=8); a.set_ylabel("Permutations", fontsize=8)
    a.set_title("(A) Inter-session agreement, same fold model", fontsize=8.5)
    b.hist(pm["null_abs_r"], bins=60, color=LIGHT, edgecolor="white", lw=0.2)
    b.axvline(abs(pm["obs_r"]), color=GREY, ls="--", lw=1.6)
    b.annotate(f"observed\n|r| = {abs(pm['obs_r']):.3f}", xy=(abs(pm["obs_r"]), b.get_ylim()[1] * 0.55), xytext=(25, 15), textcoords="offset points",
               fontsize=7.5, color=GREY, arrowprops=dict(arrowstyle="->", color=GREY))
    b.text(0.97, 0.95, f"p = {pm['p_r']:.2f}  (n.s.)", transform=b.transAxes, ha="right", va="top", fontsize=8,
           bbox=dict(boxstyle="round", fc="white", ec=GREY))
    b.set_xlabel("|Pearson r| under label permutation", fontsize=8); b.set_ylabel("Permutations", fontsize=8)
    b.set_title("(B) Anatomical tracking", fontsize=8.5)
    fig.tight_layout()
    return _save(fig, out, "Figure_12_permutation_tests")


def fig13(numbers, k, out):
    seeds = k["ms"]["seeds"]; d = k["ds"]
    s = [str(x[0]) for x in seeds]; m = np.array([x[1] for x in seeds]); icc = np.array([x[2] for x in seeds])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, vals, head, lab, fmt in ((a, m, d["mae"], "LOOCV MAE (nWBV)", "{:.4f} ± {:.4f}"), (b, icc, d["icc"], "same-fold ICC(C,1)", "{:.3f} ± {:.3f}")):
        mu, sd = vals.mean(), vals.std(ddof=1)
        ax.axhspan(mu - sd, mu + sd, color=LIGHT, alpha=0.6, label="mean ± 1 SD")
        ax.axhline(mu, color=NAVY, lw=1, label="mean across seeds")
        ax.axhline(head, color=RED, ls="--", lw=1, label="headline single run")
        ax.scatter(range(len(vals)), vals, color=NAVY, s=28, zorder=3)
        ax.set_xticks(range(len(vals)), [f"seed\n{x}" for x in s], fontsize=7)
        ax.set_ylabel(lab, fontsize=8)
        ax.set_title(("(A) MAE = " if ax is a else "(B) same-fold ICC(C,1) = ") + fmt.format(mu, sd), fontsize=8, fontweight="normal")
    a.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    return _save(fig, out, "Figure_13_multiseed_loocv_stability")


def fig14(numbers, k, out):
    d = k["ds"]
    order = np.argsort(d["err"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    cols = [RED if d["err"][i] > THRESHOLD else BLUE for i in order]
    a.barh(range(len(order)), d["err"][order], color=cols, height=0.75)
    a.set_yticks(range(len(order)), [d["subjects"][i].replace("sub-HYPE", "HY") for i in order], fontsize=6)
    a.axvline(THRESHOLD, color=RED, ls="--", lw=1.1, label=REF_LABEL)
    a.axvline(d["mae"], color=BLUE, lw=1.1, label=f"MAE = {d['mae']:.4f}")
    a.legend(fontsize=6.5, loc="lower right"); a.set_xlabel("|Error| (nWBV)", fontsize=8)
    a.set_title(f"(A) Per-Subject Absolute Error\nLOOCV n = {len(order)}: {d['n_below']}/{len(order)} below 0.020", fontsize=8)
    b.scatter(d["hfc"], d["hfe"], s=16, color=PURPLE, alpha=0.8, zorder=3)
    lo = min(d["hfc"].min(), d["hfe"].min()) - 0.005; hi = max(d["hfc"].max(), d["hfe"].max()) + 0.005
    b.plot([lo, hi], [lo, hi], ":", color="grey", lw=0.9, label="Identity"); b.set_xlim(lo, hi); b.set_ylim(lo, hi)
    _box(b, f"same-fold ICC(C,1) = {d['icc']:.3f}\n95% CI [{d['icc_ci'][0]:.3f}–{d['icc_ci'][1]:.3f}]")
    b.legend(fontsize=6.5, loc="lower right")
    b.set_xlabel("HFC session prediction (nWBV)", fontsize=8); b.set_ylabel("HFE session prediction (nWBV)", fontsize=8)
    b.set_title(f"(B) Inter-Session Reproducibility\nPaired HFC vs HFE predictions (n = {len(order)})", fontsize=8)
    fig.tight_layout()
    return _save(fig, out, "Figure_14_intersession_reliability_icc")


# ── Figs. 15-18 ────────────────────────────────────────────────────────────────

def fig15(numbers, k, out):
    g = k["cdr"]["mae_groups"]
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    cols = [BLUE, GREEN, ORANGE, RED]
    ax.bar(range(len(g)), [x[2] for x in g], yerr=[x[3] for x in g], color=cols, width=0.6, capsize=5, error_kw=dict(lw=1))
    for i, x in enumerate(g):
        ax.text(i, x[2] + x[3] + 0.004, f"{x[2]:.3f}", ha="center", fontsize=10, weight="bold")
    _ref(ax); ax.legend(loc="upper left")
    ax.set_xticks(range(len(g)), [f"{x[0].replace('No CDR label', 'No CDR\nlabel')}\n(n = {x[1]})" for x in g])
    ax.set_ylabel("MAE (nWBV)")
    ratio = g[3][2] / g[0][2]
    ax.set_ylim(0, 0.23)
    ax.text(0.97, 0.97, f"CDR 1.0 ≈ {ratio:.0f}× the\nno-CDR-label MAE (n = 2)", transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color=RED, bbox=dict(boxstyle="round", fc="#fdecea", ec=RED))
    ax.set_title(f"CDR-Stratified MAE — Physics-Simulated 64 mT\nOASIS-1 Test Set (n = {sum(x[1] for x in g)}), No Domain Adaptation", fontsize=9.5)
    return _save(fig, out, "Figure_15_cdr_stratified_mae_simulated")


def fig16(numbers, k, out):
    s = k["unadapted"]["synthseg"]; pl = k["pseudolabel"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 3.1))
    a.scatter(s["x"], s["y"], s=16, color=BLUE, zorder=3)
    lo, hi = min(s["x"].min(), s["y"].min()) - 0.005, max(s["x"].max(), s["y"].max()) + 0.005
    a.plot([lo, hi], [lo, hi], "--", color="grey", lw=0.8); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    _box(a, f"r = {s['r']:.3f}\nMAE = {s['mae']:.4f}")
    a.set_xlabel("FastSurfer nWBV (3 T)", fontsize=8); a.set_ylabel("SynthSeg+ nWBV (64 mT)", fontsize=8)
    a.set_title(f"(A) Label Agreement\nSynthSeg+ vs FastSurfer (n = {len(s['x'])})", fontsize=8)
    rows = [("FastSurfer labels", pl["FastSurfer_GT"], GREEN), ("SynthSeg+ labels", pl["SynthSeg_pseudo"], ORANGE)]
    for i, (lab, r, col) in enumerate(rows):
        b.bar(i, r["mae"], color=col, width=0.55, yerr=[[r["mae"] - r["mae_ci"][0]], [r["mae_ci"][1] - r["mae"]]], capsize=5)
        b.text(i, r["mae_ci"][1] + 0.001, f"{r['mae']:.4f}", ha="center", fontsize=8, weight="bold")
    _ref(b); b.legend(fontsize=7, loc="upper left")
    b.set_xticks(range(2), [x[0] for x in rows], fontsize=8); b.set_ylabel("Test MAE (nWBV)", fontsize=8); b.set_ylim(0, 0.036)
    b.set_title("(B) Adapter Test MAE by Label Source\nSame session, fixed 15/8 split, vs FastSurfer", fontsize=8)
    fig.tight_layout()
    return _save(fig, out, "Figure_16_pseudolabel_ablation")


def fig17(numbers, k, out):
    bars = k["comparators"]; ms = k["ms"]; o = k["oasis"]; d = k["ds"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.7, 1]})
    cols = [NAVY if kind == "adopted" else LIGHT for _, _, _, kind in bars]
    a.bar(range(len(bars)), [x[1] for x in bars], color=cols, edgecolor="black", lw=0.4)
    for i, (_, m_, r_, _) in enumerate(bars):
        a.text(i, m_ * 1.02, f"{m_:.4f}\nr={r_:.3f}", ha="center", fontsize=5.2)
    a.set_xticks(range(len(bars)), [x[0] for x in bars], fontsize=6)
    a.set_ylabel(f"OASIS-1 test MAE (nWBV), n = {len(o['y'])}", fontsize=7)
    a.set_title("(A) High-field accuracy by architecture and pretraining objective", fontsize=7.5, loc="left", fontweight="normal")
    xs = [0, 1]
    b.plot(xs, [ms["dino"]["oasis"], ms["dino"]["real"]], "o-", color=RED, label="DINO pretraining")
    b.plot(xs, [o["mae"], d["mae"]], "o-", color=NAVY, label="proxy regression (adopted)")
    for x, v in zip(xs, [ms["dino"]["oasis"], ms["dino"]["real"]]):
        b.text(x + 0.04, v, f"{v:.4f}", fontsize=6.5, color=RED)
    for x, v in zip(xs, [o["mae"], d["mae"]]):
        b.text(x + 0.04, v, f"{v:.4f}", fontsize=6.5, color=NAVY)
    b.set_xticks(xs, [f"OASIS-1\nn = {len(o['y'])}", f"real 64 mT LOOCV\nn = {len(d['y'])}"], fontsize=6.5)
    b.set_xlim(-0.3, 1.4); b.set_ylabel("test MAE (nWBV)", fontsize=7); b.legend(fontsize=6)
    b.set_title("(B) The high-field advantage does not transfer", fontsize=7.5, loc="left", fontweight="normal")
    fig.tight_layout()
    return _save(fig, out, "Figure_17_architecture_pretraining_comparators")


def fig18(numbers, k, out):
    ad = k["ms"]["adapters"]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    cols = [NAVY if x["key"] == "ln_head" else "#7fa7cc" for x in ad]
    for i, x in enumerate(ad):
        ax.bar(i, x["mae"], color=cols[i], width=0.6, yerr=[[x["mae"] - x["ci"][0]], [x["ci"][1] - x["mae"]]], capsize=5)
        ax.text(i, x["ci"][1] * 1.02, f"{x['mae']:.4f}", ha="center", fontsize=8, weight="bold")
    ax.axhline(_v(numbers, "ds006557.constant_loo_mean.mae"), color=GREY, ls="--", lw=1.1,
               label=f"Constant-mean baseline ({_v(numbers, 'ds006557.constant_loo_mean.mae'):.4f})")
    ax.legend(fontsize=7)
    ax.set_xticks(range(len(ad)), [f"{x['label']}\n{x['params']}" for x in ad], fontsize=7.5)
    ax.set_xlabel("Adapter strategy (trainable parameters)", fontsize=8); ax.set_ylabel("Cross-session LOOCV MAE (nWBV)", fontsize=8)
    ax.set_ylim(0, 0.021)
    ax.set_title(f"Adapter strategy — all 95% CIs overlap (n = {numbers['ds006557.n']['value']})", fontsize=9)
    return _save(fig, out, "Figure_18_adapter_strategy_ablation")


# ── Figs. 20-23 ────────────────────────────────────────────────────────────────

def fig20(numbers, k, out):
    mc = k["mc"]
    o = np.argsort(mc["gt"])
    x = np.arange(len(o))
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.fill_between(x, mc["lo"][o], mc["hi"][o], color=LIGHT, alpha=0.8, label=f"MC Dropout 95% CI (mean width {mc['width'].mean():.3f})")
    ax.plot(x, mc["mean"][o], "o-", color=BLUE, ms=3.5, lw=1, label="ViT3D prediction (no adapt.)")
    ax.plot(x, mc["gt"][o], "s-", color=RED, ms=3.5, lw=1, label="True nWBV (FastSurfer, 3 T)")
    bad = np.abs(mc["mean"][o] - mc["gt"][o]) > THRESHOLD
    ax.scatter(x[bad], mc["gt"][o][bad], marker="x", s=60, color=RED, lw=1.6, label=f"|error| > 0.020 ({bad.sum()}/{len(o)})", zorder=4)
    ax.set_xlabel("Subject (sorted by true nWBV)"); ax.set_ylabel("nWBV")
    ax.legend(fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax.set_title(f"MC Dropout Uncertainty — Real 64 mT (n = {len(o)}, N = 100 passes)\n"
                 f"Empirical coverage {_v(numbers, 'kept.mc.coverage95') * 100:.1f}% at nominal 95%; intervals not suitable for clinical use", fontsize=8.5)
    return _save(fig, out, "Figure_20_mcdropout_intervals_real64mt")


def fig21(numbers, k, out):
    mc = k["mc"]; d = k["ds"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1))
    a.plot([0.45, 1.0], [0.45, 1.0], ":", color="black", lw=1, label="Perfect calibration")
    a.plot(mc["conf_levels"], mc["conf_cov"], "o-", color=GREEN, label="Leave-one-out conformal")
    lv = mc["levels"][mc["levels"] >= 0.5]
    a.plot(lv, mc["mc_cov"][mc["levels"] >= 0.5], "s-", color=RED, label="MC Dropout")
    a.annotate(f"{mc['conf_cov'][-1] * 100:.1f}% at 95%\n(fixed by construction)", xy=(0.95, mc["conf_cov"][-1]), xytext=(-40, -70),
               textcoords="offset points", fontsize=7, color=GREEN, arrowprops=dict(arrowstyle="->", color=GREEN))
    a.annotate(f"{mc['mc_cov'][-1] * 100:.1f}% at 95%", xy=(0.95, mc["mc_cov"][-1]), xytext=(-70, 35), textcoords="offset points",
               fontsize=7, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
    a.set_xlim(0.45, 1.0); a.set_ylim(-0.03, 1.03); a.legend(fontsize=6.5, loc="upper left")
    a.set_xlabel("Nominal confidence level", fontsize=8); a.set_ylabel("Empirical coverage", fontsize=8)
    a.set_title("(A) Reliability curve", fontsize=8.5)
    c = mc["c90"]
    o = np.argsort(d["y"]); x = np.arange(len(o))
    b.errorbar(x, d["hfe"][o], yerr=c["q"][o], fmt="none", ecolor=LIGHT, elinewidth=5, alpha=0.9)
    b.scatter(x, d["hfe"][o], marker="_", s=60, color=NAVY, label="Prediction", zorder=3)
    cov = c["covered"][o]
    b.scatter(x[cov], d["y"][o][cov], s=14, color=GREEN, label="True (covered)", zorder=4)
    b.scatter(x[~cov], d["y"][o][~cov], s=40, marker="X", color=RED, label="True (not covered)", zorder=4)
    b.set_xlabel("Subject (sorted by true nWBV)", fontsize=8); b.set_ylabel("nWBV", fontsize=8); b.legend(fontsize=6.3, loc="upper left")
    b.set_title(f"(B) 90% leave-one-out conformal intervals — {cov.sum()}/{len(cov)} covered", fontsize=8.5)
    fig.tight_layout()
    return _save(fig, out, "Figure_21_conformal_calibration")


def fig22(numbers, k, out):
    mc = k["mc"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.1))
    lv = np.concatenate([[0], mc["levels"]]); cv = np.concatenate([[0], mc["mc_cov"]])
    a.plot([0, 100], [0, 100], "--", color="grey", lw=1, label="Perfect calibration")
    a.fill_between(lv * 100, cv * 100, lv * 100, color=RED, alpha=0.15)
    a.plot(lv * 100, cv * 100, "o-", color=RED, ms=3.5, label="MC Dropout (ViT3D)")
    a.annotate(f"95% nominal\n→ {mc['mc_cov'][-1] * 100:.1f}% empirical", xy=(95, mc["mc_cov"][-1] * 100), xytext=(-75, 25),
               textcoords="offset points", fontsize=7, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
    a.set_xlim(0, 100); a.set_ylim(0, 100); a.legend(fontsize=6.8, loc="upper left")
    a.set_xlabel("Nominal Confidence Level (%)", fontsize=8); a.set_ylabel("Empirical Coverage (%)", fontsize=8)
    a.set_title(f"(A) Reliability Curve\n(MC Dropout, n = {len(mc['gt'])})", fontsize=8.5)
    sel = [0.5, 0.8, 0.9, 0.95]
    errs = [abs(mc["mc_cov"][np.isclose(mc["levels"], s)][0] - s) * 100 for s in sel]
    b.bar(range(4), errs, color=[BLUE, ORANGE, RED, RED], width=0.6)
    for i, e in enumerate(errs):
        b.text(i, e + 2, f"{e:.1f}%", ha="center", fontsize=8, weight="bold")
    b.axhline(5, color=GREEN, ls="--", lw=1, label="Acceptable (±5%)"); b.legend(fontsize=6.8)
    b.set_xticks(range(4), [f"{int(s * 100)}%" for s in sel]); b.set_ylim(0, 120)
    b.set_xlabel("Nominal Confidence Level", fontsize=8); b.set_ylabel("|Empirical − Nominal| (%)", fontsize=8)
    b.set_title("(B) Calibration Error\nby Confidence Level", fontsize=8.5)
    fig.suptitle(f"MC Dropout Calibration Analysis — Real 64 mT (n = {len(mc['gt'])})", fontsize=9.5, weight="bold")
    fig.tight_layout()
    return _save(fig, out, "Figure_22_mcdropout_calibration_curve")


def fig23(numbers, k, out):
    o = k["oasis"]; d = k["ds"]; mc = k["mc"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    a, b, c, e = axes.ravel()
    res = o["pred"] - o["y"]
    edge = o["tertile_edges"][0]
    a.scatter(o["y"], res, s=12, c=[RED if x <= edge else BLUE for x in o["y"]], zorder=3)
    a.axvline(edge, color=ORANGE, ls=":", lw=1, label=f"lowest-tertile edge (nWBV = {edge:.2f})")
    a.axhline(0, color="black", lw=0.7); a.axhspan(-THRESHOLD, THRESHOLD, color="#eeeeee")
    a.legend(fontsize=6.3); a.set_xlabel("True nWBV", fontsize=7.5); a.set_ylabel("Residual (pred − true)", fontsize=7.5)
    a.set_title("(A) Residuals — OASIS Test", fontsize=8)
    labs = [f"Low\n(atrophy)\n≤ {edge:.2f}", f"Mid\n{edge:.2f}–{o['tertile_edges'][1]:.2f}", f"High\n(normal)\n> {o['tertile_edges'][1]:.2f}"]
    b.bar(range(3), o["tertile_mae"], color=[RED, ORANGE, GREEN], width=0.6)
    for i, (m_, n_) in enumerate(zip(o["tertile_mae"], o["tertile_n"])):
        b.text(i, m_ + 0.003, f"{m_:.3f}\n(n={n_})", ha="center", fontsize=6.8)
    _ref(b); b.legend(fontsize=6.3)
    b.set_xticks(range(3), labs, fontsize=6.5); b.set_ylabel("MAE (nWBV)", fontsize=7.5); b.set_ylim(0, 0.17)
    b.set_title("(B) MAE by nWBV Tertile", fontsize=8)
    fail = d["err"] > THRESHOLD
    c.scatter(d["age"][~fail], d["err"][~fail], s=14, color=BLUE, label="≤ 0.020")
    c.scatter(d["age"][fail], d["err"][fail], s=26, marker="D", color=RED, label="> 0.020")
    for i in np.where(fail)[0]:
        c.annotate(d["subjects"][i].replace("sub-HYPE", "HY"), (d["age"][i], d["err"][i]), xytext=(3, 2), textcoords="offset points", fontsize=6, color=RED)
    c.axhline(THRESHOLD, color=RED, ls="--", lw=0.9)
    c.text(0.03, 0.95, f"ρ = {d['age_err'].statistic:+.3f}, p = {d['age_err'].pvalue:.3f}", transform=c.transAxes, va="top", fontsize=7,
           bbox=dict(boxstyle="round", fc="white", ec="#999999"))
    c.legend(fontsize=6.3, loc="lower left"); c.set_xlabel("Subject Age (years)", fontsize=7.5); c.set_ylabel("|Error| (nWBV)", fontsize=7.5)
    c.set_title("(C) Error vs Age — LOOCV", fontsize=8)
    e.scatter(mc["width"], mc["err"], s=14, color=PURPLE)
    sl, ic = np.polyfit(mc["width"], mc["err"], 1)
    xs = np.array([mc["width"].min(), mc["width"].max()])
    e.plot(xs, sl * xs + ic, "--", color="grey", lw=0.9)
    e.text(0.03, 0.95, f"r = {mc['r_saved']:+.3f}, p = {mc['p_saved']:.3f}\n(n.s. at n = {len(mc['gt'])})", transform=e.transAxes, va="top", fontsize=7,
           bbox=dict(boxstyle="round", fc="white", ec="#999999"))
    e.set_xlabel("MC Dropout CI Width (nWBV)", fontsize=7.5); e.set_ylabel("|Error| (nWBV)", fontsize=7.5)
    e.set_title("(D) CI Width vs Error\n(Calibration Check)", fontsize=8)
    fig.tight_layout()
    return _save(fig, out, "Figure_23_failure_analysis")


def build_kept(numbers, k, out):
    with plt.rc_context(STYLE):
        return [f(numbers, k, out) for f in (fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10, fig11, fig12,
                                             fig13, fig14, fig15, fig16, fig17, fig18, fig20, fig21, fig22, fig23)]
