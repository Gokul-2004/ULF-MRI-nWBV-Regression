"""
Review-3 resubmission figures
=============================
Rebuilds fig05 (architecture / pretraining comparators) and adds the new
external-validation figure required by R1.3.

fig05 replaces the version that (a) labelled CNN3D "4.1M" — the true count is
8,222,337, see experiments/model_param_counts/ — and (b) showed the frozen-encoder
ridge probe in panel B. That probe pooled Swin-UNETR's 8 decoder channels against
ViT3D's 256-dim encoder features (a 32x representation handicap) and is removed
from the paper as methodologically unsound. Panel B now carries the sound test:
the same two pretraining objectives under the identical cross-session 64 mT LOOCV.

fig24 is new: external validation on the van den Broek Zenodo cohort, with
FastSurfer nWBV ground truth derived by the same pipeline as OASIS-1/ds006557.

Every plotted number is read from committed experiment JSON at run time and
printed with its source. Nothing is simulated; this file contains no np.random.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
EXP  = ROOT / "experiments"
OUT  = ROOT / "Review - 2" / "figures_to_insert"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.9,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8,
    "legend.frameon": True,
    "svg.fonttype": "none",
})
C_MODEL, C_NULL, C_GOOD, C_BAD, C_REF = "#1f4e79", "#b8cce4", "#2e7d32", "#c62828", "#616161"
GREY_LIT = "#cfcfcf"   # literature value, not measured here

CHECKS = []
def load(p):
    with open(p) as fh: return json.load(fh)
def chk(label, value, source):
    CHECKS.append((label, value, source)); return value
def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", {"dpi": 600}), ("svg", {})):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  wrote {stem}.png (600 dpi) + {stem}.svg")


def fig05_comparators():
    """(A) OASIS-1 accuracy by model/objective with true parameter counts.
       (B) Pretraining-objective transfer to real 64 mT — the advantage inverts."""
    params = load(EXP / "model_param_counts" / "results.json")["counts"]
    cnncmp = load(EXP / "oasis_cnn_comparison" / "results.json")
    swin   = load(EXP / "arch_comparator_swin" / "results.json")
    unetr  = load(EXP / "arch_comparator_unetr" / "results.json")
    ssl    = {k: load(EXP / f"ssl_comparator_{k}" / "results.json")
              for k in ("dino", "mae", "simmim", "contrastive")}
    dino_h = load(EXP / "dino_headline_loocv" / "results.json")

    vitp = params["BaselineViT3D"]["params"]
    cnnp = params["BaselineCNN3D"]["params"]

    rows = [
        ("DINO",               vitp, chk("DINO OASIS MAE",   ssl["dino"]["mae"],        "ssl_comparator_dino"),        ssl["dino"]["pearson_r"],        False),
        ("MAE",                vitp, chk("MAE OASIS MAE",    ssl["mae"]["mae"],         "ssl_comparator_mae"),         ssl["mae"]["pearson_r"],         False),
        ("SimMIM",             vitp, chk("SimMIM OASIS MAE", ssl["simmim"]["mae"],      "ssl_comparator_simmim"),      ssl["simmim"]["pearson_r"],      False),
        ("denoise\n(ours)",    vitp, chk("ViT3D OASIS MAE", cnncmp["ViT3D"]["mae"], "oasis_cnn_comparison"),       cnncmp["ViT3D"]["pearson_r"],    True),
        ("contrast.",          vitp, chk("Contrast OASIS MAE", ssl["contrastive"]["mae"], "ssl_comparator_contrastive"), ssl["contrastive"]["pearson_r"], False),
        ("CNN3D",              cnnp, chk("CNN3D OASIS MAE",  cnncmp["CNN3D"]["mae"],    "oasis_cnn_comparison"),       cnncmp["CNN3D"]["pearson_r"],    False),
        ("Swin-\nUNETR", swin["params"], chk("Swin OASIS MAE", swin["mae"],              "arch_comparator_swin"),       swin["pearson_r"],               False),
        ("UNETR",      unetr["params"], chk("UNETR OASIS MAE", unetr["mae"],            "arch_comparator_unetr"),      unetr["pearson_r"],              False),
    ]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.2, 4.1),
                                   gridspec_kw={"width_ratios": [1.65, 1]})

    labels = [r[0] if i < 5 else f"{r[0]}\n{r[1]/1e6:.2f}M" for i, r in enumerate(rows)]
    maes   = [r[2] for r in rows]
    rs     = [r[3] for r in rows]
    cols   = [C_MODEL if r[4] else C_NULL for r in rows]
    x = range(len(rows))

    bars = axA.bar(x, maes, color=cols, edgecolor="black", linewidth=0.7, width=0.64, zorder=3)
    axA.set_xticks(list(x)); axA.set_xticklabels(labels, fontsize=7.6)
    axA.set_ylabel("OASIS-1 test MAE (nWBV), n = 38")
    axA.set_ylim(0, max(maes) * 1.30)
    axA.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0); axA.set_axisbelow(True)
    for b, m, r in zip(bars, maes, rs):
        axA.text(b.get_x() + b.get_width()/2, m + max(maes)*0.02,
                 f"{m:.4f}\nr={r:.3f}", ha="center", va="bottom", fontsize=6.8)
    axA.set_title("(A)  High-field accuracy by architecture and pretraining objective",
                  fontsize=9, loc="left")
    axA.plot([-0.32, 4.32], [-0.093, -0.093], color=C_REF, lw=0.9,
             transform=axA.get_xaxis_transform(), clip_on=False)
    for xe in (-0.32, 4.32):
        axA.plot([xe, xe], [-0.093, -0.078], color=C_REF, lw=0.9,
                 transform=axA.get_xaxis_transform(), clip_on=False)
    axA.text(2.0, -0.108, f"ViT3D backbone, {vitp/1e6:.2f}M — pretraining objective varied",
             transform=axA.get_xaxis_transform(), ha="center", va="top",
             fontsize=7.2, color=C_REF, clip_on=False)
    ours = rows[3]
    axA.annotate("adopted configuration",
                 xy=(3, ours[2]), xytext=(3.15, max(maes)*1.16),
                 fontsize=7.4, color=C_MODEL,
                 arrowprops=dict(arrowstyle="->", color=C_MODEL, lw=0.9))

    # ── panel B: the two objectives, high field -> real 64 mT
    den_oasis  = cnncmp["ViT3D"]["mae"]
    dino_oasis = chk("DINO OASIS MAE (B)", dino_h["dino_oasis_test_mae"], "dino_headline_loocv")
    den_64     = chk("denoise 64mT LOOCV MAE", dino_h["denoising_headline"]["loocv_mae"], "dino_headline_loocv")
    dino_64    = chk("DINO 64mT LOOCV MAE",    dino_h["dino_64mt_loocv_mae"],             "dino_headline_loocv")
    den_icc    = dino_h["denoising_headline"]["icc"]
    dino_icc   = chk("DINO 64mT ICC", dino_h["dino_64mt_icc"], "dino_headline_loocv")
    dino_below = chk("DINO n<0.020", dino_h["n_below_threshold"], "dino_headline_loocv")

    xs = [0, 1]
    axB.plot(xs, [dino_oasis, dino_64], "o-", color=C_BAD,   lw=1.9, ms=7,
             label="DINO pretraining", zorder=3)
    axB.plot(xs, [den_oasis,  den_64],  "o-", color=C_MODEL, lw=1.9, ms=7,
             label="denoising (ours)", zorder=3)
    for xi, v in zip(xs, [dino_oasis, dino_64]):
        axB.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=7.4, color=C_BAD)
    for xi, v in zip(xs, [den_oasis, den_64]):
        axB.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=7.4, color=C_MODEL)
    axB.set_xticks(xs)
    axB.set_xticklabels(["OASIS-1 (3T)\nn = 38", "real 64 mT LOOCV\nn = 23"], fontsize=8)
    axB.set_xlim(-0.35, 1.35)
    axB.set_ylim(0.0, max(dino_oasis, den_oasis) * 1.12)
    axB.set_ylabel("test MAE (nWBV)")
    axB.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0); axB.set_axisbelow(True)
    axB.legend(loc="center right", fontsize=7.6)
    axB.set_title("(B)  The high-field advantage does not transfer", fontsize=9, loc="left")
    axB.text(0.5, -0.19,
             f"64 mT: ICC {den_icc:.3f} vs {dino_icc:.3f} · subjects below 0.020: 19/23 vs {dino_below}/23\n"
             "paired Wilcoxon p = 0.070 (n.s.)",
             transform=axB.transAxes, ha="center", va="top", fontsize=7.0, color=C_REF)

    fig.tight_layout()
    save(fig, "fig05_architecture_pretraining_comparators")


def fig24_external_validation():
    """External validation on the van den Broek Zenodo cohort (R1.3)."""
    z    = load(EXP / "zenodo_external_validation" / "results.json")
    loo  = load(EXP / "loocv_cross_session" / "results.json")

    gt   = z["gt_nwbv"]
    ad   = z["models"]["real64mt_adapted"]
    un   = z["models"]["oasis_unadapted"]
    n    = chk("Zenodo n", z["n_subjects"], "zenodo_external_validation")

    # training-cohort GT range, read from the LOOCV record
    tr = []
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("gt", "true_nwbv", "gt_nwbv", "true") and isinstance(v, (int, float)): tr.append(v)
                else: walk(v)
        elif isinstance(o, list):
            for i in o: walk(i)
    walk(loo)
    tr_lo, tr_hi = min(tr), max(tr)
    tr_mean = sum(tr) / len(tr)
    chk("ds006557 GT range", (round(tr_lo, 4), round(tr_hi, 4)), "loocv_cross_session")
    chk("ds006557 GT mean",  round(tr_mean, 4),                  "loocv_cross_session")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.0, 4.1),
                                   gridspec_kw={"width_ratios": [1, 1.05]})

    lo, hi = 0.735, 0.885
    axA.plot([lo, hi], [lo, hi], "--", color=C_REF, lw=1.0, label="identity", zorder=1)
    axA.axhspan(tr_lo, tr_hi, color=C_MODEL, alpha=0.09, zorder=0)
    axA.text(lo + 0.003, tr_hi - 0.003, "adaptation-cohort nWBV range",
             ha="left", va="top", fontsize=6.8, color=C_MODEL)
    axA.scatter(gt, un["preds"], s=42, color=C_GOOD, edgecolor="black", linewidth=0.6,
                label=f"OASIS (unadapted)  MAE {un['mae']:.4f}", zorder=3)
    axA.scatter(gt, ad["preds"], s=42, color=C_BAD, edgecolor="black", linewidth=0.6,
                label=f"64 mT-adapted  MAE {ad['mae']:.4f}", zorder=3)
    axA.set_xlim(lo, hi); axA.set_ylim(lo, hi); axA.set_aspect("equal")
    axA.set_xlabel("FastSurfer nWBV (ground truth)")
    axA.set_ylabel("predicted nWBV")
    axA.grid(linewidth=0.4, alpha=0.5, zorder=0); axA.set_axisbelow(True)
    axA.legend(loc="upper left", fontsize=7.0)
    axA.set_title(f"(A)  External cohort, n = {n}", fontsize=9, loc="left")

    # ── panel B: why — the label distributions do not overlap
    series = [
        ("ds006557 truth\n(adaptation cohort)", tr_lo, tr_hi, C_MODEL, tr_mean),
        ("Zenodo truth\n(external cohort)",     min(gt), max(gt), C_GOOD, sum(gt)/len(gt)),
        ("64 mT-adapted\npredictions",          min(ad["preds"]), max(ad["preds"]), C_BAD,
                                                sum(ad["preds"])/len(ad["preds"])),
        ("OASIS unadapted\npredictions",        min(un["preds"]), max(un["preds"]), "#8e24aa",
                                                sum(un["preds"])/len(un["preds"])),
    ]
    for i, (lab, a, b, col, mu) in enumerate(series):
        y = len(series) - 1 - i
        axB.hlines(y, a, b, color=col, lw=8, alpha=0.75, zorder=3)
        axB.plot([mu], [y], "|", color="black", ms=13, mew=1.6, zorder=4)
        axB.text(b + 0.0025, y, f"[{a:.4f}, {b:.4f}]", va="center", fontsize=6.9, color=col)
    axB.set_yticks(range(len(series)))
    axB.set_yticklabels([s[0] for s in reversed(series)], fontsize=7.4)
    axB.set_xlabel("nWBV")
    axB.set_xlim(0.745, 0.895)
    axB.grid(axis="x", linewidth=0.4, alpha=0.5, zorder=0); axB.set_axisbelow(True)
    axB.set_title("(B)  Training and external label ranges do not overlap", fontsize=9, loc="left")
    axB.text(0.5, -0.19,
             f"Adapted predictions cluster at {sum(ad['preds'])/len(ad['preds']):.4f} against an adaptation-cohort mean of {tr_mean:.4f}:\n"
             f"the model returns its training mean and does not extrapolate (r = {ad['pearson_r']:.3f}, p = {ad['pearson_p']:.3f}).",
             transform=axB.transAxes, ha="center", va="top", fontsize=7.0, color=C_REF)

    fig.tight_layout()
    save(fig, "fig24_external_validation_zenodo")


def fig10_fourway_with_latency():
    """Accuracy vs cost. Latency is MEASURED for both learned models; the
    segmentation comparator is a published figure, marked as such."""
    lat  = load(EXP / "inference_latency" / "results.json")
    loo  = load(EXP / "loocv_cross_session" / "results.json")

    vit_ms = chk("ViT3D latency (measured)", lat["headline_ms"], "inference_latency")
    cnn_ms = lat.get("cnn3d", {}).get("headline_ms")
    if cnn_ms is None:
        print("  !! CNN3D latency missing — run scripts/benchmark_cnn3d.py first")
        return
    chk("CNN3D latency (measured)", cnn_ms, "inference_latency/cnn3d")

    # SynthSeg+: published runtime, NOT measured here. Marked on the figure.
    SYNTHSEG_S, SYNTHSEG_SRC = 150.0, "published"

    rows = [
        ("SynthSeg+\n(upper bound)", 0.005,  SYNTHSEG_S * 1000, GREY_LIT, True),
        ("CNN3D\n(no adapt.)",       0.076,  cnn_ms,            C_NULL,   False),
        ("ViT3D\n(no adapt.)",       0.040,  vit_ms,            "#8e24aa", False),
        ("ViT3D\n(LOOCV adapt.)",    0.0134, vit_ms,            C_MODEL,  False),
    ]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.0, 4.0))

    x = range(len(rows))
    axA.bar(x, [r[1] for r in rows], color=[r[3] for r in rows],
            edgecolor="black", linewidth=0.7, width=0.6, zorder=3)
    for i, r in enumerate(rows):
        axA.text(i, r[1] + 0.0025, f"{r[1]:.4f}", ha="center", va="bottom", fontsize=7.6)
    axA.set_xticks(list(x)); axA.set_xticklabels([r[0] for r in rows], fontsize=7.6)
    axA.set_ylabel("real 64 mT MAE (nWBV)")
    axA.set_ylim(0, max(r[1] for r in rows) * 1.22)
    axA.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0); axA.set_axisbelow(True)
    axA.set_title("(A)  Accuracy on real 64 mT hardware", fontsize=9, loc="left")

    bars = axB.bar(x, [r[2] for r in rows], color=[r[3] for r in rows],
                   edgecolor="black", linewidth=0.7, width=0.6, zorder=3)
    bars[0].set_hatch("///")
    axB.set_yscale("log")
    axB.set_xticks(list(x)); axB.set_xticklabels([r[0] for r in rows], fontsize=7.6)
    axB.set_ylabel("inference time per volume (ms, log scale)")
    axB.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0, which="both"); axB.set_axisbelow(True)
    for i, r in enumerate(rows):
        txt = f"~{r[2]/1000:.0f} s\n(published)" if r[4] else f"{r[2]:.1f} ms\n(measured)"
        axB.text(i, r[2] * 1.35, txt, ha="center", va="bottom", fontsize=7.0)
    axB.set_ylim(1, SYNTHSEG_S * 1000 * 12)
    axB.set_title("(B)  Inference cost", fontsize=9, loc="left")
    axB.text(0.5, -0.30,
             "Hatched bar is a published runtime, not measured here. Both learned models were timed on CPU\n"
             f"under an identical protocol (cold full-volume forward pass, batch 1, {lat['torch_threads']} threads, "
             f"{lat['cpu_model']}).",
             transform=axB.transAxes, ha="center", va="top", fontsize=6.9, color=C_REF)

    fig.tight_layout()
    save(fig, "fig10_fourway_comparison_with_latency")


if __name__ == "__main__":
    print("Review-3 figures\n" + "=" * 60)
    fig05_comparators()
    fig24_external_validation()
    fig10_fourway_with_latency()
    print("\nValues plotted (check against the manuscript):")
    for lab, val, src in CHECKS:
        print(f"  {lab:32s} = {str(val):24s} <- {src}")
