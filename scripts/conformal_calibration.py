"""
Split-Conformal Calibration for LOOCV nWBV (Reviewer R1.7)
==========================================================
The manuscript reports MC-Dropout 95% intervals with only 4.3% empirical
coverage — severely overconfident under domain shift. This script provides a
*calibrated* alternative with a finite-sample coverage guarantee, computed
from the committed cross-session LOOCV predictions (no model / no data /
no retraining). Pure standard library.

Method — leave-one-out split conformal on the LOOCV residuals:
  Each of the 23 LOOCV predictions is already produced by an adapter that
  never saw that subject (subject-independent), so the absolute residuals
  r_i = |pred_i - true_i| are genuine out-of-sample nonconformity scores.
  For a held-out subject i and target coverage (1 - alpha):
     q_i = the ceil((m+1)(1-alpha))-th smallest of the OTHER m = n-1 residuals
           (clipped to the max residual when that rank exceeds m),
     interval_i = [pred_i - q_i,  pred_i + q_i].
  Marginal coverage is then measured empirically and compared to MC-Dropout.

This turns the reported negative (miscalibration) into a delivered fix with
valid ~90/95% coverage, which is the remedy named in the R1.7 response.

Output: experiments/conformal_calibration/results.json
"""

import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LOOCV_JSON   = PROJECT_ROOT / "experiments" / "loocv_cross_session" / "results.json"
MCDROP_JSON  = PROJECT_ROOT / "experiments" / "real64mt_eval" / "mc_dropout_ci.json"
OUT_DIR      = PROJECT_ROOT / "experiments" / "conformal_calibration"
ALPHAS       = [0.10, 0.05]        # 90% and 95% target coverage


def conformal_halfwidth(cal_residuals, alpha):
    """Split-conformal quantile for a calibration residual set."""
    m = len(cal_residuals)
    s = sorted(cal_residuals)
    k = math.ceil((m + 1) * (1 - alpha))     # 1-indexed rank
    if k > m:                                 # rank exceeds sample -> use max (finite)
        return s[-1], True
    return s[k - 1], False


def evaluate(preds, trues, alpha):
    n = len(preds)
    residuals = [abs(preds[i] - trues[i]) for i in range(n)]
    covered, widths, capped = 0, [], 0
    per_subject = []
    for i in range(n):
        cal = [residuals[j] for j in range(n) if j != i]     # leave-one-out
        q, was_capped = conformal_halfwidth(cal, alpha)
        capped += int(was_capped)
        lo, hi = preds[i] - q, preds[i] + q
        is_cov = lo <= trues[i] <= hi
        covered += int(is_cov)
        widths.append(2 * q)
        per_subject.append({
            "pred": round(preds[i], 4), "true": round(trues[i], 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "width": round(2 * q, 4), "covered": is_cov,
        })
    return {
        "target_coverage":   round(1 - alpha, 2),
        "empirical_coverage": round(covered / n, 4),
        "n_covered":         covered,
        "n":                 n,
        "mean_interval_width": round(sum(widths) / n, 4),
        "n_capped_to_max_residual": capped,
        "per_subject":       per_subject,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    recs  = json.load(open(LOOCV_JSON))["per_subject"]
    preds = [r["pred_hfe"]  for r in recs]
    trues = [r["true_nwbv"] for r in recs]

    conformal = {f"level_{int((1-a)*100)}": evaluate(preds, trues, a) for a in ALPHAS}

    # MC-Dropout baseline (the miscalibrated method being replaced)
    mc = {}
    if MCDROP_JSON.exists():
        d = json.load(open(MCDROP_JSON))
        mc = {
            "method": "MC Dropout (95%)",
            "empirical_coverage": round(d.get("gt_coverage_pct", 0) / 100.0, 4),
            "mean_interval_width": round(d.get("mean_ci_width", 0), 4),
            "note": "Reported in manuscript as severely overconfident under domain shift.",
        }

    results = {
        "experiment": "split_conformal_calibration",
        "addresses":  ["R1.7"],
        "method":     "Leave-one-out split conformal on subject-independent LOOCV residuals",
        "n_subjects": len(recs),
        "conformal":  conformal,
        "mc_dropout_baseline": mc,
    }
    out_path = OUT_DIR / "results.json"
    json.dump(results, open(out_path, "w"), indent=2)

    print("=" * 64)
    print("SPLIT-CONFORMAL CALIBRATION (R1.7)")
    print("=" * 64)
    if mc:
        print(f"MC-Dropout 95%:   coverage={mc['empirical_coverage']*100:5.1f}%   "
              f"width={mc['mean_interval_width']:.4f}   (miscalibrated)")
    for lvl, r in conformal.items():
        print(f"Conformal {int(r['target_coverage']*100)}%:   "
              f"coverage={r['empirical_coverage']*100:5.1f}%   "
              f"width={r['mean_interval_width']:.4f}   "
              f"({r['n_covered']}/{r['n']} covered)")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
