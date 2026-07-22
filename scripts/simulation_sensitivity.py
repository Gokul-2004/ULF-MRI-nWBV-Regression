"""
Simulation Parameter Sensitivity Analysis (Reviewer R1 #10)
============================================================
Tests whether the headline nWBV result is robust to the exact physics
constants used in the 64 mT simulator. The *trained* model
(oasis_finetuned.pt) is held fixed; only the simulation parameters are
perturbed, the OASIS-1 test set is re-simulated, and the change in test
MAE is reported.

Parameters swept (each ±20% about its default):
  - SNR_64MT_EFFECTIVE   (Rician noise level; default ≈ 4.0)
  - B0_INHOMO_STRENGTH   (B0 bias-field amplitude; default 0.10)
  - T1/T2 relaxation     (all 64 mT tissue values scaled ±20%)

Test set is the identical seed-42 OASIS split used in finetune_oasis.py,
so the baseline row reproduces the manuscript OASIS test MAE.

Output: experiments/simulation_sensitivity/results.json
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
from scipy.ndimage import zoom

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

import utils.field_conversion as fc
from utils.field_conversion import FieldConverter
import finetune_oasis as fo
from models.baselines import BaselineViT3D

TARGET_SHAPE = (64, 64, 64)
DEVICE       = torch.device("cpu")
CKPT         = project_root / "checkpoints" / "oasis_finetuned.pt"
OUT_DIR      = project_root / "experiments" / "simulation_sensitivity"

# Defaults captured once (module constants)
DEF_SNR = fc.SNR_64MT_EFFECTIVE
DEF_B0  = fc.B0_INHOMO_STRENGTH
DEF_T1  = dict(WM=fc.T1_WM_64MT, GM=fc.T1_GM_64MT, CSF=fc.T1_CSF_64MT)
DEF_T2  = dict(WM=fc.T2_WM_64MT, GM=fc.T2_GM_64MT, CSF=fc.T2_CSF_64MT)


def load_t1w(rec) -> np.ndarray:
    """Replicate OASISDataset.__getitem__ preprocessing exactly."""
    img = nib.load(str(rec["t1w"]))
    t1w = img.get_fdata(dtype=np.float32)
    if t1w.ndim == 4:
        t1w = t1w[..., 0]
    vmin, vmax = t1w.min(), t1w.max()
    if vmax > vmin:
        t1w = (t1w - vmin) / (vmax - vmin)
    if t1w.shape != TARGET_SHAPE:
        zf = [t / s for t, s in zip(TARGET_SHAPE, t1w.shape)]
        t1w = zoom(t1w, zf, order=1)
    return t1w.astype(np.float32)


def reset_defaults():
    fc.SNR_64MT_EFFECTIVE = DEF_SNR
    fc.B0_INHOMO_STRENGTH = DEF_B0
    fc.T1_WM_64MT, fc.T1_GM_64MT, fc.T1_CSF_64MT = DEF_T1["WM"], DEF_T1["GM"], DEF_T1["CSF"]
    fc.T2_WM_64MT, fc.T2_GM_64MT, fc.T2_CSF_64MT = DEF_T2["WM"], DEF_T2["GM"], DEF_T2["CSF"]


def eval_mae(model, test_recs) -> float:
    """Re-simulate every test T1w under current fc constants; return MAE."""
    converter = FieldConverter({})   # reads fc constants at convert() time
    abs_errs = []
    for rec in test_recs:
        t1w = load_t1w(rec)
        lf  = converter.convert(t1w, method="hyperfine")
        x   = torch.from_numpy(lf).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            pred = float(model(x).item())
        abs_errs.append(abs(pred - rec["nwbv"]))
    return float(np.mean(abs_errs))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 62)
    print("Simulation Parameter Sensitivity (OASIS test, frozen model)")
    print("=" * 62, flush=True)

    # Reconstruct the exact seed-42 OASIS test split
    scans   = fo.find_oasis_scans()
    import pandas as pd
    df = pd.read_excel(fo.CSV_PATH)
    df["subj_id"] = df["ID"].str.extract(r"(OAS1_\d{4}_MR\d)")
    df = df.set_index("subj_id")
    records = fo.build_records(scans, df)

    n = len(records)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    np.random.seed(42)
    idx = np.random.permutation(n)
    test_recs = [records[i] for i in idx[n_train + n_val:]]
    print(f"OASIS test subjects: {len(test_recs)}", flush=True)

    # Load frozen trained model
    model = BaselineViT3D(img_size=TARGET_SHAPE, patch_size=16, num_classes=4,
                          embed_dim=256, num_layers=4, num_heads=8)
    model.head = torch.nn.Linear(model.head.in_features, 1)
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Baseline (default constants)
    reset_defaults()
    base_mae = eval_mae(model, test_recs)
    print(f"\n  BASELINE MAE = {base_mae:.4f}\n", flush=True)

    runs = []

    def record(name, param, value, default):
        mae = eval_mae(model, test_recs)
        d   = mae - base_mae
        pct = 100.0 * (value - default) / default if default else 0.0
        runs.append({"perturbation": name, "param": param,
                     "value": round(float(value), 4),
                     "pct_change": round(pct, 1),
                     "mae": round(mae, 4),
                     "delta_mae": round(d, 4)})
        print(f"  {name:<22} MAE={mae:.4f}  ΔMAE={d:+.4f}", flush=True)
        reset_defaults()

    # SNR ±20%  (higher SNR = less noise)
    reset_defaults(); fc.SNR_64MT_EFFECTIVE = DEF_SNR * 1.20
    record("SNR +20% (less noise)", "SNR_64MT_EFFECTIVE", fc.SNR_64MT_EFFECTIVE, DEF_SNR)
    reset_defaults(); fc.SNR_64MT_EFFECTIVE = DEF_SNR * 0.80
    record("SNR -20% (more noise)", "SNR_64MT_EFFECTIVE", fc.SNR_64MT_EFFECTIVE, DEF_SNR)

    # B0 inhomogeneity ±20%
    reset_defaults(); fc.B0_INHOMO_STRENGTH = DEF_B0 * 1.20
    record("B0 +20%", "B0_INHOMO_STRENGTH", fc.B0_INHOMO_STRENGTH, DEF_B0)
    reset_defaults(); fc.B0_INHOMO_STRENGTH = DEF_B0 * 0.80
    record("B0 -20%", "B0_INHOMO_STRENGTH", fc.B0_INHOMO_STRENGTH, DEF_B0)

    # Relaxation ±20% (scale all 64mT T1 and T2 values together)
    reset_defaults()
    for a in ("T1_WM_64MT","T1_GM_64MT","T1_CSF_64MT","T2_WM_64MT","T2_GM_64MT","T2_CSF_64MT"):
        setattr(fc, a, getattr(fc, a) * 1.20)
    record("Relaxation +20%", "T1/T2_64MT_all", 1.20, 1.0)
    reset_defaults()
    for a in ("T1_WM_64MT","T1_GM_64MT","T1_CSF_64MT","T2_WM_64MT","T2_GM_64MT","T2_CSF_64MT"):
        setattr(fc, a, getattr(fc, a) * 0.80)
    record("Relaxation -20%", "T1/T2_64MT_all", 0.80, 1.0)

    reset_defaults()

    max_delta = max(abs(r["delta_mae"]) for r in runs)
    print("\n" + "=" * 62)
    print(f"  Baseline MAE = {base_mae:.4f}")
    print(f"  Max |ΔMAE| across all ±20% perturbations = {max_delta:.4f}")
    print("=" * 62, flush=True)

    out = {
        "experiment":   "simulation_sensitivity",
        "test_n":       len(test_recs),
        "baseline_mae": round(base_mae, 4),
        "max_abs_delta_mae": round(max_delta, 4),
        "runs":         runs,
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {OUT_DIR / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
