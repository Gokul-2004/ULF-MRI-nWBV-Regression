"""
E7 — Conformal intervals: exact algorithm, construction check, per-cohort coverage
and discrimination (R3.1, R4.4, R2.9)
=================================================================================
Algorithm used in the submission (scripts/conformal_calibration.py), stated exactly:
  residuals e_j = |pred_HFE_j - y_j| from the cross-session LOOCV (j = 1..23);
  for subject i, calibration set C_i = {e_j : j != i} (m = 22);
  q_i = the k-th smallest element of C_i, k = ceil((m+1)(1-alpha));
  interval_i = pred_HFE_i +/- q_i.
This is a jackknife-style leave-one-out conformal procedure on cross-validated
residuals (not split conformal). Subject i's own residual never calibrates its
own interval, but every residual comes from a fold model that shares 21 of its
22 training subjects with subject i's fold model, so only jackknife+/CV+-type
guarantees (>= 1 - 2*alpha) apply, not the exact 1 - alpha guarantee.

With m = 22 and continuous residuals, the number of covered subjects is
DETERMINISTIC: subject i is covered iff e_i <= q_i, and q_i is the k-th order
statistic of the other 22. Exactly k of the 23 subjects satisfy this (no ties),
so coverage is k/23 for ANY residual vector: 21/23 at 90%, 22/23 at 95%.
Section 1 verifies this numerically.

Per cohort (R2.9) we report coverage, width, width/SD, width/range, the mean
fraction of the cohort's true values inside each subject's interval, and how
many intervals contain the cohort mean (i.e. whether intervals discriminate
subjects or merely cover the distribution).

Cohorts: ds006557 internal (headline adapter; Swin readout), external
van den Broek (headline all-23 adapter, ds006557 quantiles; from E3),
OASIS-1 split conformal (calibrate on the 37 validation subjects, test on 38).

Output: experiments/r3_e7_conformal/results.json
"""

import json
import math

import numpy as np
import torch

from common import EXP, ROOT, lcs, provenance, save_json

OUT = EXP / "r3_e7_conformal" / "results.json"
ALPHAS = (0.10, 0.05)


def kth(res, alpha):
    m = len(res)
    k = math.ceil((m + 1) * (1 - alpha))
    return sorted(res)[min(k, m) - 1], k, k > m


def loo_conformal(pred, y, alpha):
    e = np.abs(pred - y)
    q = np.array([kth(np.delete(e, i), alpha)[0] for i in range(len(e))])
    return q


def discrimination(pred, y, q):
    lo, hi = pred - q, pred + q
    inside = [(np.sum((y >= lo[i]) & (y <= hi[i])) / len(y)) for i in range(len(y))]
    cov = (y >= lo) & (y <= hi)
    width = 2 * q
    return {"n": len(y), "n_covered": int(cov.sum()), "coverage": round(float(cov.mean()), 3),
            "mean_width": round(float(width.mean()), 4),
            "width_over_truth_sd": round(float(width.mean() / y.std(ddof=1)), 2),
            "width_over_truth_range": round(float(width.mean() / (y.max() - y.min())), 2),
            "mean_fraction_of_cohort_inside_each_interval": round(float(np.mean(inside)), 3),
            "n_intervals_containing_cohort_mean": int(np.sum((lo <= y.mean()) & (hi >= y.mean())))}


def main():
    res = {"experiment": "r3_e7_conformal_per_cohort", "answers": ["R3.1", "R4.4", "R2.9"],
           "algorithm": __doc__.split("Algorithm used in the submission (scripts/conformal_calibration.py), stated exactly:")[1].split("Per cohort")[0].strip(),
           "label": "internal, retrospective leave-one-out (jackknife-style) conformal calibration on cross-validated residuals",
           "provenance": provenance("scripts/round3/e7_conformal_per_cohort.py")}

    # 1. construction check
    rng = np.random.default_rng(0)
    counts = {a: set() for a in ALPHAS}
    for t in range(5000):
        y = rng.normal(0, 1, 23)
        p = y + rng.standard_t(df=1 + t % 5, size=23) * rng.uniform(0.1, 5)
        for a in ALPHAS:
            q = loo_conformal(p, y, a)
            counts[a].add(int(np.sum(np.abs(p - y) <= q)))
    res["coverage_is_fixed_by_construction"] = {
        f"level_{int((1-a)*100)}": {"k": math.ceil(23 * (1 - a)), "covered_counts_observed_over_5000_random_residual_vectors": sorted(counts[a])}
        for a in ALPHAS}
    print("construction check:", res["coverage_is_fixed_by_construction"])

    cohorts = {}
    # 2. ds006557 internal — headline
    head = json.loads((EXP / "loocv_cross_session" / "results.json").read_text())["per_subject"]
    y = np.array([r["true_nwbv"] for r in head]); p = np.array([r["pred_hfe"] for r in head])
    cohorts["ds006557_headline_adapter"] = {f"level_{int((1-a)*100)}": discrimination(p, y, loo_conformal(p, y, a)) for a in ALPHAS}
    sw = json.loads((EXP / "r3_e2b_swin_verification" / "results.json").read_text())["per_subject"]
    ys = np.array([r["true_nwbv"] for r in sw]); ps = np.array([r["swin_ridge_hfe"] for r in sw])
    cohorts["ds006557_swin_ridge_readout"] = {f"level_{int((1-a)*100)}": discrimination(ps, ys, loo_conformal(ps, ys, a)) for a in ALPHAS}

    # 3. external (E3)
    e3p = EXP / "r3_e3_external" / "results.json"
    if e3p.exists():
        e3 = json.loads(e3p.read_text())
        yz = np.array([r["true_nwbv"] for r in e3["per_subject"]])
        for orient in ("native_as_submitted", "reoriented_RAS"):
            for model, resid_src in (("headline_adapter_seedmean", np.abs(p - y)), ("swin_ridge_readout", np.abs(ps - ys))):
                pz = np.array([r[f"{orient}:{model}"] for r in e3["per_subject"]])
                cohorts[f"external_{orient}_{model}"] = {
                    f"level_{int((1-a)*100)}": discrimination(pz, yz, np.full(len(yz), kth(list(resid_src), a)[0])) for a in ALPHAS}

    # 4. OASIS split conformal
    from e4_matched_physics_vs_blur import fixed_inputs, oasis_split
    from utils.field_conversion import FieldConverter
    _, va, te = oasis_split()
    conv = FieldConverter({})
    m = lcs.build_base_model().eval()
    xv, yv = fixed_inputs(va, "physics", conv, salt=5_000)
    xt, yt = fixed_inputs(te, "physics", conv, salt=6_000)
    with torch.no_grad():
        pv = m(xv).numpy().ravel(); pt = m(xt).numpy().ravel()
    cal = np.abs(pv - yv)
    block = {}
    for a in ALPHAS:
        q, k, capped = kth(list(cal), a)
        block[f"level_{int((1-a)*100)}"] = discrimination(pt, yt, np.full(len(yt), q)) | {"k": k, "m_cal": len(cal), "capped_at_max": capped}
    block["note"] = ("Split conformal: calibration = 37 validation subjects, evaluation = 38 test subjects, inputs physics-simulated "
                     "with fixed noise. The validation set also selected the Stage-2 checkpoint (early stopping on val r), "
                     "so calibration is mildly optimistic.")
    block["test_mae"] = round(float(np.mean(np.abs(pt - yt))), 4)
    block["test_bias"] = round(float(np.mean(pt - yt)), 4)
    block["val_mae"] = round(float(np.mean(cal)), 4)
    cohorts["oasis1_split_conformal_stage2"] = block
    res["cohorts"] = cohorts
    save_json(OUT, res)
    for k, v in cohorts.items():
        l95 = v["level_95"]
        print(f"{k:58s} cov95 {l95['n_covered']}/{l95['n']}  width {l95['mean_width']}  w/SD {l95['width_over_truth_sd']}  "
              f"frac-cohort-inside {l95['mean_fraction_of_cohort_inside_each_interval']}  contain-mean {l95['n_intervals_containing_cohort_mean']}")


if __name__ == "__main__":
    main()
