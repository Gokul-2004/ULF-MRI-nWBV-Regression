"""
Simulated Dementia Evaluation
==============================
Addresses reviewer concern: "no dementia patients on real 64mT hardware."

Protocol:
  - Take the held-out OASIS-1 test set (n=38, CDR 0 / 0.5 / 1.0)
  - Apply physics-constrained 64mT simulator to their T1w scans
  - Evaluate OASIS fine-tuned ViT3D on simulated 64mT inputs
  - Report MAE stratified by CDR group

Framing in paper:
  "In simulated 64mT conditions, CDR-stratified MAE shows X-fold degradation
  in atrophic subjects (CDR=1.0: MAE=0.XXX vs CDR=0: MAE=0.XXX), consistent
  with the atrophy-failure pattern documented on real hardware."

Output: experiments/simulated_dementia/results.json
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
from scipy.ndimage import zoom
from scipy import stats

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.baselines import BaselineViT3D
from utils.field_conversion import FieldConverter

# ── Config ─────────────────────────────────────────────────────────────────────
OASIS_PROC   = project_root / "data" / "oasis_processed"
OASIS_XLSX   = project_root / "data" / "oasis_raw" / "oasis_cross-sectional.xlsx"
CNN_RESULTS  = project_root / "experiments" / "oasis_cnn_comparison" / "results.json"
VIT_CKPT     = project_root / "checkpoints" / "oasis_finetuned.pt"
OUT_DIR      = project_root / "experiments" / "simulated_dementia"
TARGET_SHAPE = (64, 64, 64)
DEVICE       = torch.device("cpu")


# ── Data helpers ───────────────────────────────────────────────────────────────

def find_t1w(subject_id: str) -> Path | None:
    """Find OASIS T1w .img file for a subject across all discs."""
    for disc in sorted(OASIS_PROC.iterdir()):
        subj_dir = disc / subject_id
        if not subj_dir.exists():
            continue
        # Try RAW/mpr-1_anon.img first (most subjects)
        p = subj_dir / "RAW" / f"{subject_id}_mpr-1_anon.img"
        if p.exists():
            return p
        # Fallback: any mpr-1_anon.img in subject dir
        candidates = list(subj_dir.glob("**/*mpr-1_anon.img"))
        if candidates:
            return candidates[0]
    return None


def load_analyse(path: Path) -> np.ndarray | None:
    """Load Analyse .img/.hdr pair via nibabel."""
    try:
        img = nib.load(str(path))
        vol = img.get_fdata(dtype=np.float32)
        if vol.ndim == 4:
            vol = vol[..., 0]
        return vol
    except Exception as e:
        print(f"  Load error {path.name}: {e}")
        return None


def resize64(vol: np.ndarray) -> np.ndarray:
    if vol.shape == TARGET_SHAPE:
        return vol
    factors = [t / s for t, s in zip(TARGET_SHAPE, vol.shape)]
    return zoom(vol, factors, order=1).astype(np.float32)


def norm_percentile(vol: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
    if hi > lo:
        vol = np.clip(vol, lo, hi)
        vol = (vol - lo) / (hi - lo)
    return vol.astype(np.float32)


def load_test_subjects() -> list[dict]:
    """Load test subject IDs + GT nWBV from CNN comparison (same split)."""
    with open(CNN_RESULTS) as f:
        d = json.load(f)
    return [
        {"subject": s["subject"], "true_nwbv": s["gt"]}
        for s in d["CNN3D"]["per_subject"]
    ]


def load_cdr_map() -> dict[str, float | None]:
    """Load CDR ratings from OASIS xlsx keyed by subject ID."""
    import pandas as pd
    df = pd.read_excel(OASIS_XLSX)
    cdr_map = {}
    for _, row in df.iterrows():
        sid = str(row.get("ID", "")).strip()
        cdr = row.get("CDR")
        if sid:
            cdr_map[sid] = None if (cdr != cdr) else float(cdr)
    return cdr_map


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model() -> BaselineViT3D:
    model = BaselineViT3D(
        img_size=TARGET_SHAPE, patch_size=16, num_classes=4,
        embed_dim=256, num_layers=4, num_heads=8,
    )
    model.head = torch.nn.Linear(model.head.in_features, 1)
    ckpt = torch.load(VIT_CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict(model: BaselineViT3D, vol: np.ndarray) -> float:
    x = torch.tensor(vol).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    return float(model(x).item())


# ── Statistics ─────────────────────────────────────────────────────────────────

def cdr_stats(records: list[dict]) -> dict:
    out = {}
    for cdr_val in [0.0, 0.5, 1.0, None]:
        group = [r for r in records if r["cdr"] == cdr_val]
        if not group:
            continue
        errs = [r["abs_error"] for r in group]
        key = "CDR_None" if cdr_val is None else f"CDR_{cdr_val}"
        out[key] = {
            "n": len(group),
            "mae": round(float(np.mean(errs)), 4),
            "mae_std": round(float(np.std(errs)), 4),
            "subjects": [r["subject"] for r in group],
        }
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("Simulated Dementia Evaluation  (OASIS test → sim 64mT → ViT3D)")
    print("=" * 62)

    test_subjects = load_test_subjects()
    cdr_map       = load_cdr_map()
    converter     = FieldConverter(config={"method": "hyperfine"})
    model         = build_model()

    print(f"Test subjects: {len(test_subjects)}")

    records    = []
    n_missing  = 0
    n_sim_fail = 0

    for entry in test_subjects:
        subj     = entry["subject"]
        true_val = entry["true_nwbv"]
        cdr      = cdr_map.get(subj)

        t1w_path = find_t1w(subj)
        if t1w_path is None:
            print(f"  SKIP {subj}: T1w not found")
            n_missing += 1
            continue

        raw = load_analyse(t1w_path)
        if raw is None:
            n_sim_fail += 1
            continue

        # Physics-constrained 64mT simulation
        try:
            sim = converter.convert(raw)
        except Exception as e:
            print(f"  SIM FAIL {subj}: {e}")
            n_sim_fail += 1
            continue

        vol  = norm_percentile(resize64(sim))
        pred = predict(model, vol)
        err  = abs(pred - true_val)

        records.append({
            "subject":    subj,
            "cdr":        cdr,
            "true_nwbv":  round(true_val, 4),
            "pred_nwbv":  round(pred, 4),
            "abs_error":  round(err, 4),
        })

        cdr_str = f"CDR={cdr}" if cdr is not None else "CDR=?"
        print(f"  {subj}  {cdr_str:8s}  "
              f"GT={true_val:.4f}  pred={pred:.4f}  err={err:.4f}")

    if not records:
        print("No records produced — check data paths.")
        return

    # ── Aggregate ──────────────────────────────────────────────────────────────
    all_errs = [r["abs_error"] for r in records]
    mae_all  = float(np.mean(all_errs))
    by_cdr   = cdr_stats(records)

    r_val, p_r = stats.pearsonr(
        [r["pred_nwbv"] for r in records],
        [r["true_nwbv"] for r in records],
    )

    print(f"\n{'='*62}")
    print(f"SIMULATED DEMENTIA RESULTS  (n={len(records)})")
    print(f"{'='*62}")
    print(f"Overall MAE = {mae_all:.4f}  (Pearson r={r_val:.4f}, p={p_r:.4f})")
    print(f"\nCDR-stratified MAE:")
    for k, v in by_cdr.items():
        print(f"  {k:12s}  n={v['n']:2d}  MAE={v['mae']:.4f} ± {v['mae_std']:.4f}")

    if "CDR_0.0" in by_cdr and "CDR_1.0" in by_cdr:
        ratio = by_cdr["CDR_1.0"]["mae"] / max(by_cdr["CDR_0.0"]["mae"], 1e-6)
        print(f"\nAtrophy penalty: CDR=1.0 MAE is {ratio:.1f}× CDR=0.0 MAE")

    if n_missing:
        print(f"\nSkipped (T1w not found): {n_missing}")
    if n_sim_fail:
        print(f"Skipped (simulation error): {n_sim_fail}")

    # ── Save ───────────────────────────────────────────────────────────────────
    results = {
        "experiment":    "simulated_dementia_eval",
        "description":   "OASIS test set (n=38) → physics 64mT sim → ViT3D (OASIS fine-tuned, no domain adapt)",
        "n_evaluated":   len(records),
        "n_missing":     n_missing,
        "n_sim_fail":    n_sim_fail,
        "mae_overall":   round(mae_all, 4),
        "pearson_r":     round(float(r_val), 4),
        "pearson_p":     round(float(p_r), 4),
        "cdr_stratified": by_cdr,
        "per_subject":   records,
    }
    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
