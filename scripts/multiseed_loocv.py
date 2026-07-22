"""
Multi-Seed LOOCV Robustness (Reviewer R1 #9, R3 #5)
====================================================
Re-runs the *published* cross-session LN+head LOOCV protocol across
multiple random seeds and reports MAE / ICC mean ± SD across seeds.

This imports the original loocv_cross_session module and reuses its
exact train_adapter / predict / data-loading functions, so seed 42
reproduces the headline manuscript result bit-for-bit. Only the
random seed changes between runs — the model, data, augmentation,
and adapter (LayerNorm + head, 769 params) are identical.

Output: experiments/multiseed_loocv/results.json
"""

import sys
import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy import stats

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

# Reuse the published protocol verbatim
import loocv_cross_session as base

SEEDS   = [42, 1, 7, 123, 2024]
OUT_DIR = project_root / "experiments" / "multiseed_loocv"


def run_one_seed(seed: int, subjects, gt, base_model) -> dict:
    """One full 23-fold LN+head LOOCV pass at a given seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    records = []
    n = len(subjects)
    for i, held_out in enumerate(subjects):
        train_subjs = [s for s in subjects if s != held_out]
        model = base.train_adapter(base_model, train_subjs, gt, i)
        pred_hfe = base.predict(model, held_out, "HFE")
        pred_hfc = base.predict(model, held_out, "HFC")
        true_val = gt[held_out]
        if pred_hfe is None:
            continue
        records.append({
            "subject":   held_out,
            "true_nwbv": round(true_val, 4),
            "pred_hfe":  round(pred_hfe, 4),
            "pred_hfc":  round(pred_hfc, 4) if pred_hfc is not None else None,
        })
        print(f"    seed {seed}  [{i+1:2d}/{n}] {held_out}  "
              f"err={abs(pred_hfe - true_val):.4f}", flush=True)

    trues     = np.array([r["true_nwbv"] for r in records])
    preds_hfe = np.array([r["pred_hfe"]  for r in records])
    abs_errs  = np.abs(preds_hfe - trues)

    mae  = float(np.mean(abs_errs))
    bias = float(np.mean(preds_hfe - trues))
    n_below = int(np.sum(abs_errs < 0.020))

    icc_val = None
    if all(r["pred_hfc"] is not None for r in records) and len(records) == n:
        preds_hfc = np.array([r["pred_hfc"] for r in records])
        icc_val = float(base.compute_icc31(preds_hfc, preds_hfe))

    print(f"  → seed {seed}: MAE={mae:.4f}  bias={bias:+.4f}  "
          f"{n_below}/{n} below 0.020  ICC={icc_val}", flush=True)

    return {
        "seed":     seed,
        "mae":      round(mae, 4),
        "bias":     round(bias, 4),
        "n_below_threshold": n_below,
        "icc_31":   round(icc_val, 4) if icc_val is not None else None,
        "per_subject": records,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 62)
    print("Multi-Seed LOOCV Robustness (LN+head, 769 params)")
    print(f"Seeds: {SEEDS}")
    print("=" * 62, flush=True)

    subjects, gt = base.load_subjects()
    base_model   = base.build_base_model()

    per_seed = [run_one_seed(s, subjects, gt, base_model) for s in SEEDS]

    maes = np.array([r["mae"]  for r in per_seed])
    iccs = np.array([r["icc_31"] for r in per_seed
                     if r["icc_31"] is not None])

    summary = {
        "mae_mean":  round(float(maes.mean()), 4),
        "mae_std":   round(float(maes.std(ddof=1)), 4),
        "mae_min":   round(float(maes.min()), 4),
        "mae_max":   round(float(maes.max()), 4),
        "icc_mean":  round(float(iccs.mean()), 4) if len(iccs) else None,
        "icc_std":   round(float(iccs.std(ddof=1)), 4) if len(iccs) else None,
    }

    print("\n" + "=" * 62)
    print("MULTI-SEED SUMMARY")
    print(f"  MAE  = {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}  "
          f"(range {summary['mae_min']:.4f}–{summary['mae_max']:.4f})")
    if summary["icc_mean"] is not None:
        print(f"  ICC  = {summary['icc_mean']:.4f} ± {summary['icc_std']:.4f}")
    print("=" * 62, flush=True)

    out = {
        "experiment": "multiseed_loocv",
        "protocol":   "cross_session_loocv_ln_head",
        "seeds":      SEEDS,
        "summary":    summary,
        "per_seed":   per_seed,
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {OUT_DIR / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
