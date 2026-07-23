"""
Permutation Tests for Input-Dependent Learning (Reviewer R1.1, R3.4, R3.5)
==========================================================================
Directly tests the central reviewer concern: "is the adapted model doing
anything beyond range-compressed mean regression?"

Two non-parametric permutation tests on the *already-computed* cross-session
LOOCV predictions (no model, no data, no retraining — reads the committed
experiments/loocv_cross_session/results.json). Pure standard library so it
runs anywhere with python3.

  TEST 1 — Inter-session reproducibility (the key evidence FOR the model).
    H0: the per-subject agreement between the held-out subject's HFC and HFE
        predictions is no better than chance.
    Statistic: ICC(3,1) between paired (pred_hfc, pred_hfe).
    Null: shuffle the HFE subject labels, recompute ICC, 20000 times.
    A significant result => the adapter emits reproducible, SUBJECT-SPECIFIC
    outputs across two independent acquisitions, i.e. input-dependent (not a
    constant / pure-noise predictor).

  TEST 2 — Anatomical tracking (the honest counter-evidence).
    H0: predicted nWBV is unrelated to true nWBV.
    Statistic: Pearson r(pred_hfe, true).
    Null: shuffle the truth labels, recompute |r|, 20000 times.
    Reported transparently: the reproducible signal above does NOT track the
    true value — consistent with the feasibility (not deployable) framing.

Also emits the constant-mean baselines (global + leave-one-out) so the exact
number quoted in R1.1 is reproducible from committed data.

Output: experiments/permutation_test/results.json
"""

import json
import math
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LOOCV_JSON   = PROJECT_ROOT / "experiments" / "loocv_cross_session" / "results.json"
OUT_DIR      = PROJECT_ROOT / "experiments" / "permutation_test"
N_PERM       = 20_000
SEED         = 42


# ── small stats helpers (stdlib only) ───────────────────────────────────────────

def mean(x):
    return sum(x) / len(x)


def pearson(a, b):
    ma, mb = mean(a), mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da * db > 0 else 0.0


def icc31(a, b):
    """ICC(3,1): two-way mixed, single-measures, consistency. Same estimator
    as scripts/loocv_cross_session.py::compute_icc31."""
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


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    recs = json.load(open(LOOCV_JSON))["per_subject"]
    true = [r["true_nwbv"] for r in recs]
    hfe  = [r["pred_hfe"]  for r in recs]
    hfc  = [r["pred_hfc"]  for r in recs]
    n = len(recs)

    # ── constant-mean baselines (reconciles the R1.1 number) ────────────────────
    gmean = mean(true)
    mae_model  = mean([abs(hfe[i] - true[i]) for i in range(n)])
    mae_global = mean([abs(t - gmean) for t in true])
    loo_means  = [mean([true[j] for j in range(n) if j != i]) for i in range(n)]
    mae_loo    = mean([abs(true[i] - loo_means[i]) for i in range(n)])

    # ── TEST 1: inter-session ICC permutation ───────────────────────────────────
    obs_icc = icc31(hfc, hfe)
    ge = 0
    null_icc_mean = 0.0
    for _ in range(N_PERM):
        perm = hfe[:]
        rng.shuffle(perm)
        v = icc31(hfc, perm)
        null_icc_mean += v
        if v >= obs_icc:
            ge += 1
    p_icc = (ge + 1) / (N_PERM + 1)          # add-one (never reports p=0)
    null_icc_mean /= N_PERM

    # ── TEST 2: prediction-vs-truth correlation permutation (two-sided) ─────────
    obs_r = pearson(hfe, true)
    ge_r = 0
    for _ in range(N_PERM):
        perm = true[:]
        rng.shuffle(perm)
        if abs(pearson(hfe, perm)) >= abs(obs_r):
            ge_r += 1
    p_r = (ge_r + 1) / (N_PERM + 1)

    results = {
        "experiment":  "permutation_test_input_dependence",
        "addresses":   ["R1.1", "R3.4", "R3.5"],
        "n_subjects":  n,
        "n_permutations": N_PERM,
        "seed": SEED,
        "constant_mean_baseline": {
            "model_loocv_mae":     round(mae_model, 4),
            "global_mean_mae":     round(mae_global, 4),
            "leave_one_out_mean_mae": round(mae_loo, 4),
            "note": "Model MAE is comparable to the constant-mean baseline; "
                    "the evidence for input-dependence is the ICC permutation test below.",
        },
        "test1_intersession_icc": {
            "statistic": "ICC(3,1) between per-subject HFC and HFE predictions",
            "observed_icc":   round(obs_icc, 4),
            "null_mean_icc":  round(null_icc_mean, 4),
            "p_value":        round(p_icc, 5),
            "significant_0.05": p_icc < 0.05,
            "interpretation": "Per-subject predictions agree across two independent "
                              "sessions far beyond chance => input-dependent, not a "
                              "constant/mean predictor.",
        },
        "test2_anatomical_tracking": {
            "statistic": "Pearson r between predicted and true nWBV",
            "observed_r": round(obs_r, 4),
            "p_value":    round(p_r, 5),
            "significant_0.05": p_r < 0.05,
            "interpretation": "The reproducible signal does NOT track true nWBV; "
                              "reported transparently, consistent with the "
                              "feasibility-baseline framing.",
        },
    }

    out_path = OUT_DIR / "results.json"
    json.dump(results, open(out_path, "w"), indent=2)

    print("=" * 64)
    print("PERMUTATION TESTS — input-dependence (R1.1 / R3.4 / R3.5)")
    print("=" * 64)
    print(f"Constant-mean baseline: model={mae_model:.4f}  "
          f"global={mae_global:.4f}  LOO={mae_loo:.4f}")
    print(f"\nTEST 1 inter-session ICC(3,1)")
    print(f"  observed = {obs_icc:.4f}   null-mean = {null_icc_mean:+.4f}   "
          f"p = {p_icc:.5f}  {'** SIGNIFICANT' if p_icc < 0.05 else 'n.s.'}")
    print(f"\nTEST 2 prediction-vs-truth Pearson")
    print(f"  observed = {obs_r:+.4f}   p = {p_r:.5f}  "
          f"{'significant' if p_r < 0.05 else 'n.s. (no anatomical tracking)'}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
