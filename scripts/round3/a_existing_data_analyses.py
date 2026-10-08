"""
Round-3 analyses computed from committed per-subject results (no inference)
==========================================================================
  r2_2   Normalised error and calibration of the headline LOOCV (R2.2):
         MAE/SD, RMSE/SD, MSE- and MAE-skill vs the leave-one-out mean,
         calibration slope/intercept (true regressed on predicted) with
         bootstrap CIs, prediction/truth SD ratio, and the band of constant
         predictors that clear MAE < 0.020 in this cohort. Same for each
         multiseed run.
  r2_11  The three LN+head runs side by side, unpooled (R2.11), with
         per-subject agreement between them.
  r4_3   SynthSeg+ pseudo-label experiment (fixed 15/8 split, same-session):
         constant-mean and SynthSeg-direct rows on the same 8 test subjects.
  e5     Label-uncertainty propagation (R2.4): Monte-Carlo perturbation of the
         FastSurfer labels with sigma in {0.0025, 0.005, SD(SynthSeg-FastSurfer)};
         recompute MAE CI, count < 0.020, r, constant baseline and skill.

Output: experiments/r3_existing_data_analyses/results.json
"""

import csv
import json

import numpy as np
from scipy import stats

from common import EXP, SEED, provenance, save_json

OUT = EXP / "r3_existing_data_analyses" / "results.json"
THR = 0.020
NB = 10_000


def load(p):
    return json.loads((EXP / p).read_text())


def loo_mean(y):
    return np.array([np.delete(y, i).mean() for i in range(len(y))])


def norm_metrics(pred, y):
    err = pred - y
    lm = loo_mean(y)

    def f(idx):
        p, t = pred[idx], y[idx]
        lmm = np.array([np.delete(t, i).mean() for i in range(len(t))])
        e = p - t
        lr = stats.linregress(p, t) if np.std(p) > 0 else None
        return (1 - np.mean(e ** 2) / np.mean((lmm - t) ** 2),
                1 - np.mean(np.abs(e)) / np.mean(np.abs(lmm - t)),
                lr.slope if lr else np.nan, lr.intercept if lr else np.nan,
                np.mean(np.abs(e)) / np.std(t, ddof=1))
    rng = np.random.default_rng(SEED)
    B = np.array([f(rng.integers(0, len(y), len(y))) for _ in range(NB)])
    ci = lambda j: [round(float(np.nanpercentile(B[:, j], 2.5)), 4), round(float(np.nanpercentile(B[:, j], 97.5)), 4)]
    lr = stats.linregress(pred, y)
    sd = y.std(ddof=1)
    return {
        "n": len(y), "truth_sd": round(float(sd), 5), "truth_range": round(float(y.max() - y.min()), 4),
        "mae": round(float(np.mean(np.abs(err))), 5), "rmse": round(float(np.sqrt(np.mean(err ** 2))), 5),
        "mae_over_sd": round(float(np.mean(np.abs(err)) / sd), 3), "mae_over_sd_ci95": ci(4),
        "rmse_over_sd": round(float(np.sqrt(np.mean(err ** 2)) / sd), 3),
        "loo_mean_mae": round(float(np.mean(np.abs(lm - y))), 5),
        "mse_skill_vs_loo_mean": round(float(1 - np.mean(err ** 2) / np.mean((lm - y) ** 2)), 4), "mse_skill_ci95": ci(0),
        "mae_skill_vs_loo_mean": round(float(1 - np.mean(np.abs(err)) / np.mean(np.abs(lm - y))), 4), "mae_skill_ci95": ci(1),
        "calibration_slope_true_on_pred": round(float(lr.slope), 3), "calibration_slope_se": round(float(lr.stderr), 3),
        "calibration_slope_ci95": ci(2), "calibration_intercept": round(float(lr.intercept), 3),
        "pearson_r": round(float(stats.pearsonr(pred, y)[0]), 4), "pearson_p": round(float(stats.pearsonr(pred, y)[1]), 4),
        "pred_sd_over_truth_sd": round(float(pred.std(ddof=1) / sd), 3),
        "n_below_0.020": int((np.abs(err) < THR).sum()),
    }


def constant_band(y):
    grid = np.linspace(y.min() - 0.03, y.max() + 0.03, 20001)
    ok = np.array([np.mean(np.abs(c - y)) < THR for c in grid])
    lo, hi = grid[ok].min(), grid[ok].max()
    return {"constants_with_mae_below_0.020": [round(float(lo), 4), round(float(hi), 4)],
            "band_width": round(float(hi - lo), 4),
            "band_width_over_cohort_range": round(float((hi - lo) / (y.max() - y.min())), 3),
            "best_constant_mae": round(float(min(np.mean(np.abs(c - y)) for c in grid)), 5),
            "interpretation": "Any constant output inside this band clears the 0.020 criterion; the criterion therefore cannot discriminate a working model from a non-working one in this cohort."}


def main():
    res = {"experiment": "r3_existing_data_analyses", "provenance": provenance("scripts/round3/a_existing_data_analyses.py")}
    head = load("loocv_cross_session/results.json")["per_subject"]
    subjects = [r["subject"] for r in head]
    y = np.array([r["true_nwbv"] for r in head])
    ph = np.array([r["pred_hfe"] for r in head])

    # ── R2.2 ──
    ms = load("multiseed_loocv/results.json")["per_seed"]
    res["r2_2"] = {"headline": norm_metrics(ph, y),
                   "multiseed": {str(s["seed"]): norm_metrics(np.array([r["pred_hfe"] for r in s["per_subject"]]), y) for s in ms},
                   "constant_band": constant_band(y)}
    print("R2.2 headline", {k: res["r2_2"]["headline"][k] for k in ("mae_over_sd", "rmse_over_sd", "mse_skill_vs_loo_mean", "mse_skill_ci95", "calibration_slope_true_on_pred", "calibration_slope_ci95", "pred_sd_over_truth_sd")})
    print("    band", res["r2_2"]["constant_band"])

    # ── R2.11 ──
    abl = load("ablation_adapter/results.json")["results"]["ln_head"]["per_subject"]
    pa = np.array([r["pred"] for r in abl])
    assert [r["subject"] for r in abl] == subjects
    s42 = next(s for s in ms if s["seed"] == 42)["per_subject"]
    p42 = np.array([r["pred_hfe"] for r in s42])
    runs = {"headline_loocv_cross_session": ph, "ablation_adapter_ln_head": pa, "multiseed_seed42": p42}
    tbl = {}
    for k, p in runs.items():
        e = np.abs(p - y)
        rng = np.random.default_rng(SEED)
        b = [e[rng.integers(0, 23, 23)].mean() for _ in range(NB)]
        tbl[k] = {"mae": round(float(e.mean()), 5), "mae_ci95": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)],
                  "bias": round(float((p - y).mean()), 5), "n_below_0.020": int((e < THR).sum()),
                  "pearson_r": round(float(stats.pearsonr(p, y)[0]), 4)}
    agree = {}
    keys = list(runs)
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = runs[keys[i]], runs[keys[j]]
            agree[f"{keys[i]}__vs__{keys[j]}"] = {"per_subject_r": round(float(stats.pearsonr(a, b)[0]), 3),
                                                   "mean_abs_diff": round(float(np.mean(np.abs(a - b))), 4),
                                                   "max_abs_diff": round(float(np.max(np.abs(a - b))), 4)}
    res["r2_11"] = {"runs": tbl, "pairwise_agreement": agree,
                    "multiseed_mae_sd_across_5_seeds": load("multiseed_loocv/results.json")["summary"].get("mae_std"),
                    "provenance_of_difference": (
                        "Identical recipe (oasis_finetuned.pt, 80 epochs, lr 1e-4, wd 1e-4, batch 4, 5 augmentations, best-training-loss state). "
                        "Only the RNG state differs: headline seeds 42 at import then consumes torch RNG building the base model; "
                        "the ablation seeds once and runs 23 head-only folds before ln_head; multiseed re-seeds 42 after model construction, "
                        "so its numpy augmentation stream matches the headline but its torch randperm stream does not."),
                    "pooled": False}
    print("R2.11", tbl)

    # ── R4.3 ──
    pl = load("pseudolabel_ablation/results.json")
    fs = pl["adapter_results"]["FastSurfer_GT"]["per_subject"]
    test = [r["subject"] for r in fs]
    gt = {r["subject"]: r["true_nwbv"] for r in head}
    sy = {r["subject"]: r["synthseg"] for r in pl["label_agreement"]["per_subject"]}
    train = [s for s in subjects if s not in test]
    yt = np.array([gt[s] for s in test])

    def row(pred):
        e = np.abs(np.asarray(pred) - yt)
        rng = np.random.default_rng(SEED)
        b = [e[rng.integers(0, len(e), len(e))].mean() for _ in range(NB)]
        return {"mae": round(float(e.mean()), 4), "mae_ci95": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)],
                "bias": round(float(np.mean(np.asarray(pred) - yt)), 4)}
    res["r4_3"] = {
        "design": "same-session (HFC train -> HFC test), fixed split np.random.default_rng(42) permutation, 15 train / 8 test",
        "test_subjects": test, "train_subjects": train,
        "rows": {
            "adapter_fastsurfer_labels": row([r["pred"] for r in fs]),
            "adapter_synthseg_pseudolabels": row([r["pred"] for r in pl["adapter_results"]["SynthSeg_pseudo"]["per_subject"]]),
            "constant_train_mean_fastsurfer": row([np.mean([gt[s] for s in train])] * len(test)),
            "constant_train_mean_synthseg": row([np.mean([sy[s] for s in train])] * len(test)),
            "synthseg_direct_on_64mT": row([sy[s] for s in test]),
        },
        "no_test_information": "Adapter trained only on train-subject labels; per-volume 1-99 percentile normalisation (no cohort statistics); checkpoint = lowest training-loss epoch; fixed hyper-parameters (scripts/pseudolabel_ablation.py lines 50-56, 66-75, 151-161, 193-195, 261-277).",
        "label_definition_note": "SynthSeg nWBV = (GM+WM)/TIV from run_synthseg_ds006557.sh column matching; FastSurfer nWBV = BrainSegVol/MaskVol. Definitions differ."}
    print("R4.3", res["r4_3"]["rows"])

    # ── E5 ──
    diff = np.array([r["diff"] for r in pl["label_agreement"]["per_subject"]])
    sd_diff = float(diff.std(ddof=1))
    rng = np.random.default_rng(SEED)
    e5 = {"sigma_sources": {"0.0025": "illustrative small test-retest error", "0.005": "illustrative moderate error",
                            f"{sd_diff:.4f}": "SD of SynthSeg(64 mT T2w) minus FastSurfer(3T MPRAGE) nWBV, n=23 — an UPPER bound on FastSurfer label noise because it includes the modality gap and the definition difference"},
          "noise_floor_note": "With label noise sigma, even a perfect predictor has expected MAE sigma*sqrt(2/pi) against the noisy labels.",
          "by_sigma": {}}
    n_mc = 5000
    for sig in (0.0025, 0.005, sd_diff):
        maes, below, rs, consts, skills = [], [], [], [], []
        for _ in range(n_mc):
            yt_ = y + rng.normal(0, sig, len(y))
            e = np.abs(ph - yt_)
            lm = loo_mean(yt_)
            maes.append(e.mean()); below.append((e < THR).sum()); rs.append(stats.pearsonr(ph, yt_)[0])
            consts.append(np.mean(np.abs(lm - yt_)))
            skills.append(1 - np.mean((ph - yt_) ** 2) / np.mean((lm - yt_) ** 2))
        pc = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
        e5["by_sigma"][f"{sig:.4f}"] = {
            "perfect_predictor_noise_floor_mae": round(float(sig * np.sqrt(2 / np.pi)), 4),
            "model_mae_mean": round(float(np.mean(maes)), 4), "model_mae_95pct_interval": pc(maes),
            "n_below_0.020_median": int(np.median(below)), "n_below_0.020_95pct_interval": [int(np.percentile(below, 2.5)), int(np.percentile(below, 97.5))],
            "pearson_r_95pct_interval": pc(rs),
            "loo_mean_mae_mean": round(float(np.mean(consts)), 4),
            "mse_skill_95pct_interval": pc(skills),
            "fraction_runs_model_beats_loo_mean": round(float(np.mean(np.array(maes) < np.array(consts))), 3)}
    res["e5_label_uncertainty"] = e5
    print("E5", json.dumps(e5["by_sigma"], indent=0)[:1200])
    save_json(OUT, res)


if __name__ == "__main__":
    main()
