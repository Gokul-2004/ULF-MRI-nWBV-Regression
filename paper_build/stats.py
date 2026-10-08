"""
Statistics used by every table and figure in the paper.

Self-contained (numpy + scipy only) so the build runs without torch, GPUs or
MRI data. All resampling uses np.random.default_rng(SEED); nothing else in
paper_build may draw random numbers (enforced by paper_build/lint.py).
"""

import numpy as np
from scipy import stats

SEED = 42
N_BOOT = 10_000
THRESHOLD = 0.020


def loo_mean(y):
    y = np.asarray(y, float)
    return (y.sum() - y) / (len(y) - 1)


def _rng():
    return np.random.default_rng(SEED)


def boot_ci(fn, *arrays, n_boot=N_BOOT):
    """Percentile bootstrap CI, resampling subjects jointly across arrays."""
    rng = _rng()
    arrays = [np.asarray(a, float) for a in arrays]
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = fn(*[a[idx] for a in arrays])
        if np.isfinite(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def mae(p, y):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def mse_skill(p, y):
    """1 - MSE(model) / MSE(leave-one-out constant mean). > 0 means better than the constant."""
    y = np.asarray(y, float)
    return float(1 - np.mean((np.asarray(p) - y) ** 2) / np.mean((loo_mean(y) - y) ** 2))


def calib_slope(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if np.std(p) == 0:
        return float("nan")
    return float(stats.linregress(p, y).slope)


def readout_row(p, y, ref_err=None):
    """Every metric the paper reports for one readout on one cohort."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    ae = np.abs(p - y)
    const_err = np.abs(loo_mean(y) - y)
    row = {
        "n": int(len(y)),
        "mae": mae(p, y),
        "mae_ci95": boot_ci(mae, p, y),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "bias": float(np.mean(p - y)),
        "mae_over_sd": float(ae.mean() / y.std(ddof=1)),
        "mse_skill": mse_skill(p, y),
        "mse_skill_ci95": boot_ci(mse_skill, p, y),
        "pred_sd_over_truth_sd": float(p.std(ddof=1) / y.std(ddof=1)),
        "n_below_threshold": int((ae < THRESHOLD).sum()),
    }
    if np.std(p) > 0:
        r, pr = stats.pearsonr(p, y)
        row.update({"pearson_r": float(r), "pearson_p": float(pr),
                    "calibration_slope": calib_slope(p, y),
                    "calibration_slope_ci95": boot_ci(calib_slope, p, y)})
    else:
        row.update({"pearson_r": float("nan"), "pearson_p": float("nan"),
                    "calibration_slope": float("nan"), "calibration_slope_ci95": [float("nan")] * 2})
    if not np.allclose(ae, const_err):
        row["vs_constant_wilcoxon_p"] = float(stats.wilcoxon(ae, const_err).pvalue)
        row["vs_constant_delta_mae_ci95"] = boot_ci(lambda a, b: float(np.mean(a - b)), ae, const_err)
    if ref_err is not None and not np.allclose(ae, ref_err):
        row["vs_ref_wilcoxon_p"] = float(stats.wilcoxon(ae, ref_err).pvalue)
        row["vs_ref_delta_mae"] = float(np.mean(ae - ref_err))
        row["vs_ref_delta_mae_ci95"] = boot_ci(lambda a, b: float(np.mean(a - b)), ae, ref_err)
    return row


def _anova2(a, b):
    data = np.column_stack([np.asarray(a, float), np.asarray(b, float)])
    n, k = data.shape
    gm = data.mean()
    ss_r = k * np.sum((data.mean(axis=1) - gm) ** 2)
    ss_c = n * np.sum((data.mean(axis=0) - gm) ** 2)
    ss_e = np.sum((data - gm) ** 2) - ss_r - ss_c
    return n, k, ss_r / (n - 1), ss_c / (k - 1), ss_e / ((n - 1) * (k - 1))


def icc_c1(a, b):
    """ICC(C,1) consistency, two-way mixed, single measures (McGraw & Wong) = the paper's ICC(3,1)."""
    n, k, ms_r, ms_c, ms_e = _anova2(a, b)
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e))


def icc_a1(a, b):
    """ICC(A,1) absolute agreement (McGraw & Wong)."""
    n, k, ms_r, ms_c, ms_e = _anova2(a, b)
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n))


def jackknife_conformal(pred, y, alpha):
    """Leave-one-out conformal on cross-validated residuals: subject i is calibrated on the others."""
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    e = np.abs(pred - y)
    m = len(e) - 1
    k = int(np.ceil((m + 1) * (1 - alpha)))
    q = np.array([np.sort(np.delete(e, i))[min(k, m) - 1] for i in range(len(e))])
    covered = e <= q
    return {"k": k, "m": m, "q": q, "covered": covered, "coverage": float(covered.mean()),
            "mean_width": float(np.mean(2 * q))}
