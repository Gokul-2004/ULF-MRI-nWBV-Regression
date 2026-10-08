"""
Rebuild fig8a_ci_oasis with REAL MC Dropout (integrity fix)
===========================================================
The previous fig8a_ci_oasis() in generate_all_figures.py drew its confidence
intervals from np.random.normal(...) — synthetic numbers, NOT actual MC Dropout —
and hardcoded a "4.3%" coverage that was computed on a DIFFERENT dataset (n=23
real 64mT), while the figure is labelled OASIS n=38. Both are integrity defects.

This script computes genuine MC Dropout on the OASIS-1 test set:
  - loads the OASIS-finetuned ViT3D (checkpoints/oasis_finetuned.pt),
  - enables its dropout layers at inference,
  - runs N=100 stochastic forward passes per subject,
  - forms per-subject 95% CIs from the empirical pass distribution,
  - computes the REAL empirical coverage (fraction of true nWBV inside the CI).

EVERY number on the figure is computed here from real forward passes. The
coverage in the title is the computed value, not a literal.

    python scripts/rebuild_fig8a_real_mcdropout.py

Output -> figures_rebuilt/fig8a_ci_oasis.{png,svg}  (600 dpi + vector)
"""

import sys, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from scipy.ndimage import zoom

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

import warnings; warnings.filterwarnings("ignore")

from models.baselines import BaselineViT3D
from utils.field_conversion import FieldConverter
import finetune_oasis as fo

DEVICE = torch.device("cpu")
TARGET = (64, 64, 64)
CKPT   = project_root / "checkpoints" / "oasis_finetuned.pt"
OUT    = project_root / "figures_rebuilt"
DPI    = 600
N_MC   = 100          # stochastic forward passes
THRESH = 0.020
SEED   = 42

BLUE, RED = "#1f77b4", "#d62728"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": DPI, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "savefig.transparent": False,
})


def load_test_volumes():
    """Reconstruct the seed-42 OASIS test split and load its physics-sim volumes."""
    scans = fo.find_oasis_scans()
    import pandas as pd
    df = pd.read_excel(fo.CSV_PATH)
    df["subj_id"] = df["ID"].str.extract(r"(OAS1_\d{4}_MR\d)")
    df = df.set_index("subj_id")
    records = fo.build_records(scans, df)
    n = len(records); n_tr = int(0.8 * n); n_va = int(0.1 * n)
    np.random.seed(42)
    idx = np.random.permutation(n)
    test = [records[i] for i in idx[n_tr + n_va:]]

    conv = FieldConverter({})
    vols, trues = [], []
    for rec in test:
        t1w = nib.load(str(rec["t1w"])).get_fdata(dtype=np.float32)
        if t1w.ndim == 4:
            t1w = t1w[..., 0]
        vmin, vmax = t1w.min(), t1w.max()
        if vmax > vmin:
            t1w = (t1w - vmin) / (vmax - vmin)
        if t1w.shape != TARGET:
            zf = [t / s for t, s in zip(TARGET, t1w.shape)]
            t1w = zoom(t1w, zf, order=1)
        lf = conv.convert(t1w.astype(np.float32), method="hyperfine")
        vols.append(lf.astype(np.float32))
        trues.append(float(rec["nwbv"]))
    return np.stack(vols), np.array(trues)


def enable_mc_dropout(model):
    """Put model in eval mode but re-enable Dropout layers (MC Dropout)."""
    model.eval()
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            n += 1
    return n


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    model = BaselineViT3D(img_size=TARGET, patch_size=16, num_classes=4,
                          embed_dim=256, num_layers=4, num_heads=8)
    model.head = nn.Linear(model.head.in_features, 1)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=False)["model_state_dict"])
    n_drop = enable_mc_dropout(model)
    print(f"MC Dropout enabled on {n_drop} dropout layers, N={N_MC} passes")

    vols, trues = load_test_volumes()
    print(f"OASIS test volumes: {vols.shape}, n={len(trues)}")

    X = torch.tensor(vols).unsqueeze(1).float()

    torch.manual_seed(SEED)
    # N_MC stochastic passes -> (N_MC, n_subjects)
    all_preds = np.zeros((N_MC, len(trues)), dtype=np.float32)
    with torch.no_grad():
        for k in range(N_MC):
            out = model(X).cpu().numpy().flatten()
            all_preds[k] = out
            if (k + 1) % 25 == 0:
                print(f"  pass {k+1}/{N_MC}")

    mean_pred = all_preds.mean(axis=0)
    lo = np.percentile(all_preds, 2.5, axis=0)   # 95% CI from empirical distribution
    hi = np.percentile(all_preds, 97.5, axis=0)
    ci_width = (hi - lo)

    # REAL empirical coverage: fraction of true values inside their own CI
    covered = (trues >= lo) & (trues <= hi)
    coverage_pct = 100.0 * covered.mean()
    mean_width = float(ci_width.mean())

    print(f"\nREAL MC Dropout results (n={len(trues)}):")
    print(f"  empirical coverage = {coverage_pct:.1f}% (nominal 95%)")
    print(f"  mean CI width = {mean_width:.4f}")
    print(f"  n covered = {int(covered.sum())}/{len(trues)}")

    # ── plot, sorted by true nWBV ─────────────────────────────────────────────
    order = np.argsort(trues)
    ts, ms = trues[order], mean_pred[order]
    los, his = lo[order], hi[order]
    x = np.arange(len(ts))

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.fill_between(x, los, his, alpha=0.25, color=BLUE,
                    label=f"MC Dropout 95% CI (mean width ≈ {mean_width:.3f})")
    ax.plot(x, ms, "o-", color=BLUE, ms=4, lw=1, label="ViT3D prediction (MC mean)")
    ax.plot(x, ts, "s-", color=RED, ms=4, lw=1, label="True nWBV")

    err = np.abs(ms - ts)
    fail = np.where(err > THRESH)[0]
    ax.scatter(fail, ts[fail], marker="x", color=RED, s=70, zorder=5,
               label=f"|error| > {THRESH}")

    ax.set_xlabel("Subject (sorted by true nWBV)")
    ax.set_ylabel("nWBV")
    ax.set_title(f"MC Dropout Uncertainty — OASIS-1 Test Set (n={len(trues)})\n"
                 f"Empirical coverage: {coverage_pct:.1f}% (nominal 95%); "
                 f"CIs {'under' if coverage_pct < 90 else 'well'}-calibrated",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left", ncol=2)
    ax.set_xlim(-0.5, len(ts) - 0.5)
    fig.tight_layout()

    fig.savefig(OUT / "fig8a_ci_oasis.png", format="png", dpi=DPI)
    fig.savefig(OUT / "fig8a_ci_oasis.svg", format="svg")
    plt.close(fig)

    # save the real numbers for the record / caption
    rec = {
        "experiment": "fig8a_real_mc_dropout_oasis",
        "n_subjects": int(len(trues)),
        "n_mc_passes": N_MC,
        "empirical_coverage_pct": round(coverage_pct, 2),
        "mean_ci_width": round(mean_width, 4),
        "n_covered": int(covered.sum()),
        "note": "Real MC Dropout: dropout enabled at inference, N=100 passes, "
                "95% CI from empirical percentiles. Replaces the prior figure whose "
                "CIs were np.random.normal and whose 4.3% was from the n=23 set.",
    }
    rec_dir = project_root / "experiments" / "oasis_mc_dropout"
    rec_dir.mkdir(parents=True, exist_ok=True)
    (rec_dir / "results.json").write_text(json.dumps(rec, indent=2))

    print(f"\nWrote {OUT/'fig8a_ci_oasis.png'} ({DPI} dpi) + .svg")
    print(f"Saved real numbers -> {rec_dir/'results.json'}")
    print(f"\n*** The figure's coverage ({coverage_pct:.1f}%) is COMPUTED, not hardcoded. ***")
    print(f"*** If it differs from the manuscript's 4.3%, the manuscript needs updating. ***")


if __name__ == "__main__":
    main()
