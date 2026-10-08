"""
Compute every number, table and figure dataset in the paper from sources.py.

Output of compute_all(): (numbers, tables)
  numbers  flat dict  name -> {"value": ..., "source": "<file> / <how>"}
  tables   dict       name -> {"title", "columns", "rows", "note"}
Figures read only from these two objects.
"""

import numpy as np
from scipy import stats

from . import sources as S
from .stats import (THRESHOLD, icc_a1, icc_c1, jackknife_conformal, loo_mean, readout_row)

# Display order and labels for the real-64 mT readout comparison (Table: readouts).
READOUTS = [
    ("constant_loo_mean", "Constant (leave-one-out mean)", "baseline"),
    ("age_only_ridge", "Age only (ridge)", "baseline"),
    ("vit_unadapted", "ViT3D, unadapted", "vit"),
    ("vit_ln_head_headline", "ViT3D + LayerNorm/head adapter (headline)", "vit"),
    ("adapter_head_only", "ViT3D + head-only adapter", "vit"),
    ("adapter_lora", "ViT3D + LoRA (r = 4)", "vit"),
    ("adapter_full_ft", "ViT3D, full fine-tune", "vit"),
    ("dino_ln_head", "ViT3D (DINO pretraining) + LN/head adapter", "vit"),
    ("vit_frozen_offset", "ViT3D frozen + offset", "vit"),
    ("vit_frozen_affine", "ViT3D frozen + affine", "vit"),
    ("ridge_vit_cls_postnorm", "ViT3D frozen [CLS] + ridge", "vit"),
    ("ridge_vit_meantokens", "ViT3D frozen mean tokens + ridge", "vit"),
    ("ridge_image_stats", "Image statistics (9) + ridge", "baseline"),
    ("cnn_unadapted_hfe", "CNN3D, unadapted", "cnn"),
    ("cnn_frozen_affine", "CNN3D frozen + affine", "cnn"),
    ("ridge_cnn_globalpool256", "CNN3D frozen global-pool + ridge", "cnn"),
    ("ridge_swin_gap", "Swin-UNETR frozen GAP + ridge", "swin"),
    ("swin_plus_age_ridge", "Swin-UNETR frozen GAP + age + ridge", "swin"),
]


def _n(numbers, name, value, source):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    numbers[name] = {"value": value, "source": source}


def readouts(numbers, tables):
    subjects, y, age, preds = S.ds006557()
    preds = dict(preds, constant_loo_mean=loo_mean(y))
    ref = np.abs(preds["vit_ln_head_headline"] - y)
    rows, data = [], {}
    for key, label, group in READOUTS:
        r = readout_row(preds[key], y, ref_err=ref)
        data[key] = r | {"label": label, "group": group, "pred": preds[key]}
        rows.append([label, f"{r['mae']:.4f} [{r['mae_ci95'][0]:.4f}, {r['mae_ci95'][1]:.4f}]",
                     f"{r['mae_over_sd']:.2f}", f"{r['mse_skill']:+.2f} [{r['mse_skill_ci95'][0]:+.2f}, {r['mse_skill_ci95'][1]:+.2f}]",
                     "—" if np.isnan(r["pearson_r"]) else f"{r['pearson_r']:+.2f} ({r['pearson_p']:.3g})",
                     "—" if np.isnan(r["calibration_slope"]) else f"{r['calibration_slope']:.2f}",
                     f"{r['n_below_threshold']}/{r['n']}",
                     f"{r['vs_constant_wilcoxon_p']:.3g}" if "vs_constant_wilcoxon_p" in r else "—"])
        for m in ("mae", "mse_skill", "pearson_r", "pearson_p", "calibration_slope", "n_below_threshold", "mae_over_sd",
                  "pred_sd_over_truth_sd", "bias"):
            _n(numbers, f"ds006557.{key}.{m}", r[m], "readouts table")
        for m in ("mae_ci95", "mse_skill_ci95", "calibration_slope_ci95"):
            _n(numbers, f"ds006557.{key}.{m}", r[m], "readouts table")
        for m in ("vs_constant_wilcoxon_p", "vs_ref_wilcoxon_p", "vs_ref_delta_mae", "vs_ref_delta_mae_ci95"):
            if m in r:
                _n(numbers, f"ds006557.{key}.{m}", r[m], "readouts table")
    tables["readouts_real64mt"] = {
        "title": "Real 64 mT, cross-session LOOCV (fit on HFC of 22, predict HFE of held-out subject), n = 23",
        "columns": ["Readout", "MAE [95% CI]", "MAE/SD", "MSE skill vs constant [95% CI]", "r (p)",
                    "Calibration slope", "n < 0.020", "Wilcoxon p vs constant"],
        "rows": rows,
        "note": ("MSE skill = 1 − MSE/MSE(leave-one-out mean); > 0 beats the constant. Calibration slope = truth regressed on "
                 "prediction (1 = ideal). Ridge readouts: standardisation and α chosen by inner leave-one-out on the 22 training "
                 "subjects. Bootstrap CIs: 10,000 subject resamples, seed 42.")}
    _n(numbers, "ds006557.n", len(y), "loocv_cross_session")
    _n(numbers, "ds006557.truth_sd", float(np.std(y, ddof=1)), "loocv_cross_session")
    _n(numbers, "ds006557.truth_min", float(y.min()), "loocv_cross_session")
    _n(numbers, "ds006557.truth_max", float(y.max()), "loocv_cross_session")
    _n(numbers, "ds006557.truth_range", float(y.max() - y.min()), "loocv_cross_session")
    _n(numbers, "ds006557.truth_mean", float(y.mean()), "loocv_cross_session")
    # band of constants that clear the 0.020 criterion
    grid = np.linspace(y.min() - 0.03, y.max() + 0.03, 20001)
    ok = grid[[np.mean(np.abs(c - y)) < THRESHOLD for c in grid]]
    _n(numbers, "ds006557.constant_band_clearing_threshold", [float(ok.min()), float(ok.max())], "grid search over constants")
    _n(numbers, "ds006557.constant_band_fraction_of_range", float((ok.max() - ok.min()) / (y.max() - y.min())), "grid search")
    rho, p = stats.spearmanr(age, y)
    _n(numbers, "ds006557.age_truth_spearman", [float(rho), float(p)], "e2b ages")
    rho, p = stats.spearmanr(age, preds["vit_ln_head_headline"])
    _n(numbers, "ds006557.age_headline_pred_spearman", [float(rho), float(p)], "e2b ages")
    return {"subjects": subjects, "y": y, "age": age, "rows": data}


def multiseed_rows(numbers, tables):
    ms = S.multiseed()
    _, y, _, _ = S.ds006557()
    rows = []
    maes = []
    for s in ms["per_seed"]:
        p = np.array([r["pred_hfe"] for r in s["per_subject"]])
        c = np.array([r["pred_hfc"] for r in s["per_subject"]])
        r = readout_row(p, y)
        maes.append(r["mae"])
        rows.append([s["seed"], f"{r['mae']:.4f}", f"{r['mse_skill']:+.2f}", f"{r['pearson_r']:+.2f}",
                     f"{icc_c1(c, p):.3f}", f"{r['n_below_threshold']}/23"])
        _n(numbers, f"multiseed.seed{s['seed']}.mae", r["mae"], "multiseed_loocv")
        _n(numbers, f"multiseed.seed{s['seed']}.icc_same_fold", icc_c1(c, p), "multiseed_loocv")
    _n(numbers, "multiseed.mae_mean", float(np.mean(maes)), "multiseed_loocv")
    _n(numbers, "multiseed.mae_sd", float(np.std(maes, ddof=1)), "multiseed_loocv")
    tables["multiseed"] = {"title": "ViT3D LN+head adapter, five seeds, cross-session LOOCV (n = 23)",
                           "columns": ["Seed", "MAE", "MSE skill", "r", "ICC(C,1) same fold", "n < 0.020"], "rows": rows,
                           "note": "Same-fold ICC shares the fold-level offset and is not evidence of input dependence (see reproducibility table)."}


def reproducibility(numbers, tables):
    subjects, ses, seeds, z_hfc, z_hfe = S.sessions()
    _, y, _, _ = S.ds006557()
    rows = []
    a, b = ses["headline_same_fold"]
    const = loo_mean(y)
    items = [("Headline, same fold model for both sessions", icc_c1(a, b), icc_a1(a, b)),
             ("Leave-one-out constant mean, same fold", icc_c1(const, const + 0 * const) if np.std(const) > 0 else 1.0, 1.0)]
    fa, fb = ses["frozen_single_model"]
    items.append(("Frozen Stage-2 model (one model, no folds)", icc_c1(fa, fb), icc_a1(fa, fb)))
    cross = []
    for sa in seeds:
        for sb in seeds:
            if sa != sb:
                cross.append(icc_c1(seeds[sa][0], seeds[sb][1]))
    items.append(("Cross-seed (HFC from seed a, HFE from seed b; mean of 20 pairs)", float(np.mean(cross)), None))
    for label, c1, a1 in items:
        rows.append([label, f"{c1:.3f}", "—" if a1 is None else f"{a1:.3f}"])
    _n(numbers, "repro.icc_headline_same_fold", icc_c1(a, b), "loocv_cross_session HFC/HFE")
    _n(numbers, "repro.icc_a1_headline_same_fold", icc_a1(a, b), "loocv_cross_session HFC/HFE")
    _n(numbers, "repro.icc_frozen_single_model", icc_c1(fa, fb), "r3_frozen_predictions")
    _n(numbers, "repro.icc_cross_seed_mean", float(np.mean(cross)), "multiseed_loocv, 20 ordered pairs")
    _n(numbers, "repro.icc_cross_seed_range", [float(min(cross)), float(max(cross))], "multiseed_loocv")
    e1 = S.e1()
    _n(numbers, "repro.icc_frozen_ci95", e1["a_frozen_model"]["icc_c1_ci95"], "r3_e1 a_frozen_model")
    _n(numbers, "repro.icc_frozen_perm_p", e1["a_frozen_model"]["icc_c1_perm_p"], "r3_e1 a_frozen_model")
    _n(numbers, "repro.icc_headline_ci95", e1["reference_headline_same_fold_model"]["icc_c1_ci95"], "r3_e1")
    r, p = stats.pearsonr(b, z_hfe)
    _n(numbers, "repro.headline_pred_vs_z_offset", [float(r), float(p)], "r3_e1 d_image_qc per_scan (HFE)")
    r2, p2 = stats.pearsonr(z_hfc, z_hfe)
    _n(numbers, "repro.z_offset_between_session_r", [float(r2), float(p2)], "r3_e1 d_image_qc per_scan")
    tables["reproducibility"] = {"title": "Session reproducibility of predictions (n = 23)",
                                 "columns": ["Design", "ICC(C,1)", "ICC(A,1)"], "rows": rows,
                                 "note": ("When one fold model predicts both sessions of its held-out subject, a fold-level offset is shared by "
                                          "both predictions; a constant predictor then scores ICC = 1. Only the frozen-model and "
                                          "cross-seed rows are free of that shared component.")}
    cov = e1["d_image_qc"]["covariate_table"]
    nrows = []
    for k, lab in (("com_offset_z_mm", "Head position, superior–inferior (mm)"), ("com_offset_y_mm", "Head position, anterior–posterior (mm)"),
                   ("com_offset_x_mm", "Head position, left–right (mm)"), ("head_mask_volume_ml", "Head-mask volume (mL)"),
                   ("fg_mean", "Foreground mean intensity"), ("bg_sd", "Background noise SD")):
        c = cov[k]
        nrows.append([lab, f"{c['between_session_r']:+.2f}", f"{c['r_with_headline_pred_hfe']:+.2f} ({c['p_with_headline_pred_hfe']:.2g})",
                      f"{c['r_with_frozen_pred_hfe']:+.2f} ({c['p_with_frozen_pred_hfe']:.2g})", f"{c['r_with_truth']:+.2f}"])
    pi = e1["d_image_qc"]["partial_icc"]["headline"]
    for k, lab in (("head_position_only", "head position"), ("head_mask_volume_only", "head-mask volume"),
                   ("intensity_and_snr_only", "intensity and SNR"), ("all_qc_covariates", "all QC covariates")):
        _n(numbers, f"repro.partial_icc_headline.{k}", [pi[k]["icc_after"], pi[k]["p_drop_larger_than_random"]], "r3_e1 partial_icc")
    tables["nuisance"] = {"title": "Stable nuisance signals in the 64 mT scans (n = 23)",
                          "columns": ["Covariate", "Between-session r", "r with headline prediction (p)",
                                      "r with frozen-model prediction (p)", "r with true nWBV"],
                          "rows": nrows,
                          "note": ("Single site and scanner, so site effects cannot be tested. Partial ICC of the headline after removing "
                                   "head-mask volume: {:.3f} (drop larger than for random covariates, p = {:.3f}).").format(
                              pi["head_mask_volume_only"]["icc_after"], pi["head_mask_volume_only"]["p_drop_larger_than_random"])}
    return {"headline": (a, b), "frozen": (fa, fb), "cross": cross, "z_hfe": z_hfe, "y": y}


def swin_nuisance(numbers, tables):
    d = S.e2b()
    pc = d["nuisance_partial_correlations"]
    rows = []
    for k, lab in (("age", "Age"), ("sex", "Sex"), ("fastsurfer_maskvol", "Head size (FastSurfer MaskVol)"),
                   ("head_mask_volume_hfe", "Head-mask volume, 64 mT"), ("z_offset_hfe", "Head position (z)"),
                   ("all_jointly", "All of the above jointly")):
        rows.append([lab, f"{pc[k]['partial_r_pred_truth']:.2f}", f"{pc[k]['p']:.2g}"])
        _n(numbers, f"swin.partial_r.{k}", [pc[k]["partial_r_pred_truth"], pc[k]["p"]], "r3_e2b nuisance_partial_correlations")
    perm = d["permutation_null"]
    _n(numbers, "swin.permutation_p_mae", perm["p_mae"], "r3_e2b permutation_null (1000 label permutations)")
    _n(numbers, "swin.permutation_p_r", perm["p_r"], "r3_e2b permutation_null")
    _n(numbers, "swin.session_directions", {k: v["mae"] for k, v in d["session_directions"].items()}, "r3_e2b")
    tables["swin_partial"] = {"title": "Swin-UNETR ridge readout: correlation with true nWBV after removing nuisance variables (n = 23)",
                              "columns": ["Controlled for", "Partial r (prediction, truth)", "p"], "rows": rows,
                              "note": "Unadjusted r = {:.2f}. Permutation test (1,000 label permutations): p = {} for MAE and for r.".format(
                                  numbers["ds006557.ridge_swin_gap.pearson_r"]["value"], perm["p_mae"])}


def oasis(numbers, tables):
    e8 = S.oasis_fixed()
    single = S.oasis_single_draw()
    pc = S.params()
    y = np.array(e8["true_nwbv"])
    rows, fig = [], {}
    labels = {"vit3d_headline": ("ViT3D (proxy-regression pretraining, headline)", pc["BaselineViT3D"]["params"]),
              "cnn3d": ("CNN3D", pc["BaselineCNN3D"]["params"]),
              "swin_unetr": ("Swin-UNETR", pc["SwinUNETR"]["params"]),
              "vit3d_dino": ("ViT3D (DINO pretraining)", pc["BaselineViT3D"]["params"])}
    for k, (lab, npar) in labels.items():
        m = e8["models"][k]
        rows.append([lab, f"{npar / 1e6:.2f} M", f"{m['mae_mean']:.4f} ± {m['mae_sd']:.4f}", f"{m['r_mean']:.3f} ± {m['r_sd']:.3f}",
                     f"{np.mean([r['bias'] for r in m['runs']]):+.4f}", "fixed, 10 draws"])
        fig[k] = {"label": lab, "pred": np.array(m["runs"][0]["pred"]), "mae_mean": m["mae_mean"], "mae_sd": m["mae_sd"]}
        _n(numbers, f"oasis.{k}.mae_mean", m["mae_mean"], "r3_e8 (10 fixed noise draws)")
        _n(numbers, f"oasis.{k}.mae_sd", m["mae_sd"], "r3_e8")
        _n(numbers, f"oasis.{k}.r_mean", m["r_mean"], "r3_e8")
        _n(numbers, f"oasis.{k}.r_sd", m["r_sd"], "r3_e8")
        _n(numbers, f"oasis.{k}.r_range", m["r_range"], "r3_e8")
    for k, lab, npar in (("unetr", "UNETR", pc["UNETR"]["params"]), ("ssl_mae", "ViT3D (MAE pretraining)", pc["BaselineViT3D"]["params"]),
                         ("ssl_simmim", "ViT3D (SimMIM pretraining)", pc["BaselineViT3D"]["params"]),
                         ("ssl_contrastive", "ViT3D (SimCLR pretraining)", pc["BaselineViT3D"]["params"])):
        t, p = single[k]["true"], single[k]["pred"]
        assert np.allclose(t, y), f"{k}: OASIS test subjects differ"
        r = float(stats.pearsonr(p, t)[0])
        rows.append([lab, f"{npar / 1e6:.2f} M", f"{np.mean(np.abs(p - t)):.4f}", f"{r:.3f}", f"{np.mean(p - t):+.4f}",
                     "single draw (weights not saved)"])
        _n(numbers, f"oasis.{k}.mae_single", float(np.mean(np.abs(p - t))), f"{k} per_test")
        _n(numbers, f"oasis.{k}.r_single", r, f"{k} per_test")
    tables["oasis"] = {"title": "OASIS-1 test set (n = 38), physics-simulated 64 mT inputs",
                       "columns": ["Model", "Parameters", "MAE", "r", "Bias", "Test inputs"], "rows": rows,
                       "note": ("Test inputs are simulated from the 1.5 T scans. Rows marked 'fixed' are mean ± SD over 10 seeded noise "
                                "realisations of the same 38 inputs; the other rows are one unseeded realisation, as originally reported, "
                                "because their fine-tuned weights were not saved.")}
    return {"y": y, "models": fig}


def physics_vs_blur(numbers, tables):
    summ, arms = S.e4()
    rows = []
    for seed in (42, 1, 7):
        ps = summ["per_seed"][str(seed)]
        rr = ps["loocv_real_64mt"]
        rows.append([f"seed {seed}", f"{rr['mean_abs_err_physics']:.4f}", f"{rr['mean_abs_err_blur']:.4f}",
                     f"{rr['delta_mae_physics_minus_blur']:+.4f} [{rr['delta_ci95'][0]:+.4f}, {rr['delta_ci95'][1]:+.4f}]",
                     f"{rr['wilcoxon_p']:.2g}",
                     f"{ps['oasis_test_on_physics_inputs']['delta_mae_physics_minus_blur']:+.4f}",
                     f"{ps['oasis_test_on_blur_inputs']['delta_mae_physics_minus_blur']:+.4f}"])
    po = summ["pooled_over_seeds"]
    rr = po["loocv_real_64mt"]
    rows.append(["pooled", f"{rr['mean_abs_err_physics']:.4f}", f"{rr['mean_abs_err_blur']:.4f}",
                 f"{rr['delta_mae_physics_minus_blur']:+.4f} [{rr['delta_ci95'][0]:+.4f}, {rr['delta_ci95'][1]:+.4f}]",
                 f"{rr['wilcoxon_p']:.2g}", f"{po['oasis_test_on_physics_inputs']['delta_mae_physics_minus_blur']:+.4f}",
                 f"{po['oasis_test_on_blur_inputs']['delta_mae_physics_minus_blur']:+.4f}"])
    _n(numbers, "e4.pooled.real64mt.delta_mae", rr["delta_mae_physics_minus_blur"], "r3_e4 summary pooled")
    _n(numbers, "e4.pooled.real64mt.delta_ci95", rr["delta_ci95"], "r3_e4 summary pooled")
    _n(numbers, "e4.pooled.real64mt.wilcoxon_p", rr["wilcoxon_p"], "r3_e4 summary pooled")
    _n(numbers, "e4.pooled.real64mt.mae_physics", rr["mean_abs_err_physics"], "r3_e4 summary pooled")
    _n(numbers, "e4.pooled.real64mt.mae_blur", rr["mean_abs_err_blur"], "r3_e4 summary pooled")
    _n(numbers, "e4.pooled.oasis_physics_inputs.delta", po["oasis_test_on_physics_inputs"]["delta_mae_physics_minus_blur"], "r3_e4")
    _n(numbers, "e4.pooled.oasis_blur_inputs.delta", po["oasis_test_on_blur_inputs"]["delta_mae_physics_minus_blur"], "r3_e4")
    rec = {seed: (arms[("physics", seed)]["oasis_test_on_physics_inputs"]["mae"], arms[("physics", seed)]["loocv_real_64mt"]["mae"])
           for seed in (42, 1, 7)}
    _n(numbers, "e4.recipe_rerun.oasis_mae_range", [min(v[0] for v in rec.values()), max(v[0] for v in rec.values())], "r3_e4 physics arms")
    _n(numbers, "e4.recipe_rerun.real64mt_mae_range", [min(v[1] for v in rec.values()), max(v[1] for v in rec.values())], "r3_e4 physics arms")
    _n(numbers, "e4.recipe_rerun.real64mt_mae_mean", float(np.mean([v[1] for v in rec.values()])), "r3_e4 physics arms")
    tables["physics_vs_blur"] = {
        "title": "Matched physics-simulation vs Gaussian-blur pretraining (identical pipeline, splits, budgets and seeds)",
        "columns": ["", "Real 64 mT MAE, physics", "Real 64 mT MAE, blur", "Δ (physics − blur) [95% CI]", "Wilcoxon p",
                    "OASIS Δ on physics inputs", "OASIS Δ on blur inputs"],
        "rows": rows,
        "note": "Real 64 mT: cross-session LOOCV, n = 23, paired per subject. OASIS: 38 test subjects with one fixed noise realisation for both arms."}
    return {"summary": summ, "arms": arms}


def conformal(numbers, tables, ro):
    y = ro["y"]
    rows, fig = [], []
    ext_sub, yz, ext, _, e3 = S.external()
    for key, lab in (("vit_ln_head_headline", "ViT3D LN+head adapter"), ("ridge_swin_gap", "Swin-UNETR + ridge")):
        p = ro["rows"][key]["pred"]
        for alpha in (0.10, 0.05):
            c = jackknife_conformal(p, y, alpha)
            frac = float(np.mean([np.mean(np.abs(y - p[i]) <= c["q"][i]) for i in range(len(y))]))
            rows.append([lab, "ds006557 (internal)", f"{int(round((1 - alpha) * 100))}%", f"{c['coverage'] * 100:.1f}% ({int(c['covered'].sum())}/23)",
                         f"{c['mean_width']:.4f}", f"{c['mean_width'] / (y.max() - y.min()):.2f}", f"{frac * 100:.0f}%"])
            _n(numbers, f"conformal.ds006557.{key}.{int(round((1 - alpha) * 100))}", {"coverage": c["coverage"], "width": c["mean_width"],
               "width_over_range": c["mean_width"] / (y.max() - y.min()), "cohort_fraction_inside": frac, "k": c["k"]}, "jackknife conformal")
            fig.append((lab, "internal", int(round((1 - alpha) * 100)), c["mean_width"] / (y.max() - y.min()), frac, c["coverage"]))
            # external: calibrate on all 23 internal residuals
            ekey = {"vit_ln_head_headline": "vit_ln_head_seedmean", "ridge_swin_gap": "ridge_swin_gap"}[key]
            e = np.sort(np.abs(p - y))
            k = int(np.ceil((len(e) + 1) * (1 - alpha)))
            q = e[min(k, len(e)) - 1]
            pz = ext[ekey]
            cov = float(np.mean(np.abs(pz - yz) <= q))
            fz = float(np.mean([np.mean(np.abs(yz - pz[i]) <= q) for i in range(len(yz))]))
            rows.append([lab, "external (independent)", f"{int(round((1 - alpha) * 100))}%", f"{cov * 100:.1f}% ({int(round(cov * len(yz)))}/{len(yz)})",
                         f"{2 * q:.4f}", f"{2 * q / (yz.max() - yz.min()):.2f}", f"{fz * 100:.0f}%"])
            _n(numbers, f"conformal.external.{ekey}.{int(round((1 - alpha) * 100))}", {"coverage": cov, "width": 2 * q}, "jackknife conformal, calibrated on ds006557")
    e7 = S.e7()
    for lv in ("level_90", "level_95"):
        b = e7["cohorts"]["oasis1_split_conformal_stage2"][lv]
        rows.append(["ViT3D (Stage 2)", "OASIS-1 split conformal (37 cal / 38 test)", lv[-2:] + "%",
                     f"{b['coverage'] * 100:.1f}% ({b['n_covered']}/38)", f"{b['mean_width']:.4f}", f"{b['width_over_truth_range']:.2f}",
                     f"{b['mean_fraction_of_cohort_inside_each_interval'] * 100:.0f}%"])
    _n(numbers, "conformal.k_fixed", {"90": e7["coverage_is_fixed_by_construction"]["level_90"]["k"],
                                      "95": e7["coverage_is_fixed_by_construction"]["level_95"]["k"]}, "r3_e7 construction check")
    tables["conformal"] = {"title": "Conformal prediction intervals by cohort",
                           "columns": ["Readout", "Cohort", "Nominal", "Coverage", "Mean width", "Width / cohort range",
                                       "Share of cohort inside each interval"], "rows": rows,
                           "note": ("Internal rows: leave-one-out (jackknife-style) conformal on cross-validated residuals; subject i is "
                                    "calibrated on the other 22, q_i = k-th smallest residual, k = ⌈(m+1)(1−α)⌉, m = 22. With this "
                                    "construction the number of covered subjects is fixed by rank (21/23 at 90%, 22/23 at 95%) for any "
                                    "residuals, so internal coverage carries no information about calibration quality; width and the share "
                                    "of the cohort inside each interval do. External rows are calibrated on all 23 internal residuals.")}
    return fig


def external(numbers, tables):
    subjects, y, preds, ras, e3 = S.external()
    rows = []
    for i, s in enumerate(subjects):
        rows.append([s, f"{y[i]:.4f}", f"{preds['vit_ln_head_seedmean'][i]:.4f}", f"{preds['vit_unadapted'][i]:.4f}",
                     f"{preds['ridge_swin_gap'][i]:.4f}"])
    summ = []
    for k, lab in (("vit_ln_head_seedmean", "ViT3D LN+head adapter (5-seed mean)"), ("vit_unadapted", "ViT3D unadapted"),
                   ("ridge_swin_gap", "Swin-UNETR + ridge"), ("vit_ln_head_old15_as_submitted", "Old 15-subject adapter (as submitted)")):
        p = preds[k]
        m = float(np.mean(np.abs(p - y)))
        r, pr = stats.pearsonr(p, y)
        summ.append([lab, f"{m:.4f}", f"{np.mean(p - y):+.4f}", f"{np.std(p, ddof=1):.4f}", f"{r:+.2f} ({pr:.2g})",
                     f"{np.mean(np.abs(ras[k] - y)):.4f}"])
        _n(numbers, f"external.{k}.mae", m, "r3_e3 per_subject")
        _n(numbers, f"external.{k}.r", [float(r), float(pr)], "r3_e3 per_subject")
        _n(numbers, f"external.{k}.pred_sd", float(np.std(p, ddof=1)), "r3_e3 per_subject")
    seeds = [float(np.mean(np.abs(preds[f"vit_ln_head_seed{s}"] - y))) for s in (42, 1, 7, 123, 2024)]
    _n(numbers, "external.vit_ln_head.per_seed_mae_range", [min(seeds), max(seeds)], "r3_e3 per_subject")
    lm = loo_mean(y)
    _n(numbers, "external.loo_mean_mae", float(np.mean(np.abs(lm - y))), "r3_e3 per_subject")
    _n(numbers, "external.ds006557_mean_as_constant_mae", float(np.mean(np.abs(e3["constant_baselines"]["ds006557_training_mean"]["value"] - y))), "r3_e3")
    ds = e3["distribution_shift"]
    _n(numbers, "external.truth_range", [ds["nwbv_external"]["min"], ds["nwbv_external"]["max"]], "r3_e3 distribution_shift")
    _n(numbers, "external.truth_mean", ds["nwbv_external"]["mean"], "r3_e3 distribution_shift")
    _n(numbers, "external.truth_sd", float(np.std(y, ddof=1)), "r3_e3 per_subject")
    _n(numbers, "external.n_within_ds006557_range", ds["n_external_within_ds006557_range"], "r3_e3 distribution_shift")
    sd = e3["simulated_domain_check"]["external_sim_from_3T_T2w"]
    _n(numbers, "external.sim_from_3T.mae", sd["mae"], "r3_e3 simulated_domain_check")
    _n(numbers, "external.sim_from_3T.pred_sd", sd["pred_sd"], "r3_e3 simulated_domain_check")
    _n(numbers, "external.n", len(y), "r3_e3")
    tables["external_per_subject"] = {"title": "External cohort (van den Broek et al.), per subject",
                                      "columns": ["Subject", "True nWBV", "ViT3D adapter", "ViT3D unadapted", "Swin + ridge"],
                                      "rows": rows, "note": "sub-0064 excluded: no 64 mT T2w scan in the release."}
    tables["external_summary"] = {"title": "External cohort (n = 10): every readout fails",
                                  "columns": ["Readout", "MAE", "Bias", "Prediction SD", "r (p)", "MAE after reorienting to RAS"],
                                  "rows": summ,
                                  "note": "Leave-one-out mean of the external labels: MAE {:.4f}. True-label SD {:.4f}.".format(
                                      numbers["external.loo_mean_mae"]["value"], numbers["external.truth_sd"]["value"])}
    return {"y": y, "preds": preds, "ds_range": (numbers["ds006557.truth_min"]["value"], numbers["ds006557.truth_max"]["value"])}


def misc(numbers, tables):
    an = S.e5_and_r2()
    r4 = an["r4_3"]["rows"]
    rows = [[k.replace("_", " "), f"{v['mae']:.4f} [{v['mae_ci95'][0]:.4f}, {v['mae_ci95'][1]:.4f}]", f"{v['bias']:+.4f}"] for k, v in r4.items()]
    for k, v in r4.items():
        _n(numbers, f"pseudolabel.{k}.mae", v["mae"], "r3_existing_data_analyses r4_3")
    tables["pseudolabel"] = {"title": "SynthSeg+ pseudo-label test: internal same-session feasibility test (HFC → HFC, fixed 15/8 split)",
                             "columns": ["Row", "MAE on the 8 test subjects [95% CI]", "Bias"], "rows": rows,
                             "note": an["r4_3"]["label_definition_note"]}
    e5 = an["e5_label_uncertainty"]["by_sigma"]
    rows = []
    for s, v in e5.items():
        rows.append([s, f"{v['perfect_predictor_noise_floor_mae']:.3f}", f"{v['model_mae_mean']:.4f}",
                     f"{v['mse_skill_95pct_interval'][0]:+.3f} to {v['mse_skill_95pct_interval'][1]:+.3f}",
                     f"{v['fraction_runs_model_beats_loo_mean'] * 100:.1f}%"])
    tables["label_noise"] = {"title": "Propagation of reference-label noise (Monte Carlo, ViT3D headline)",
                             "columns": ["Label noise σ", "Noise floor MAE", "Model MAE", "MSE skill (95% interval)", "Runs beating constant"],
                             "rows": rows, "note": "; ".join(f"σ = {k}: {v}" for k, v in an["e5_label_uncertainty"]["sigma_sources"].items())}
    r211 = an["r2_11"]["runs"]
    tables["run_reconciliation"] = {"title": "Three runs of the identical LN+head recipe (not pooled)",
                                    "columns": ["Run", "MAE", "Bias", "n < 0.020", "r"],
                                    "rows": [[k, f"{v['mae']:.4f}", f"{v['bias']:+.4f}", v["n_below_0.020"], f"{v['pearson_r']:+.3f}"] for k, v in r211.items()],
                                    "note": an["r2_11"]["provenance_of_difference"]}
    tb = S.e6()
    tables["time_budget"] = {"title": "Time budget per scan and per site (Intel i7-8750H CPU, 6 threads)",
                             "columns": ["Step", "Time", "Needed for"],
                             "rows": [[r["step"], r["time"], r["needed_for"]] for r in tb["summary_table"]],
                             "note": "Acquisition time from the scanner's BIDS sidecar; all other steps measured or taken from run logs."}
    _n(numbers, "time.forward_ms", tb["forward_pass_cold"]["median_ms"], "r3_e6 / inference_latency")
    _n(numbers, "time.end_to_end_ms", tb["end_to_end_scan_to_prediction"]["median_ms"], "r3_e6")
    _n(numbers, "time.preprocess_ms", tb["load_preprocess_per_scan"]["median_ms"], "r3_e6")
    _n(numbers, "time.acquisition_s", tb["acquisition_axial_T2w"]["median_s"], "r3_e6")
    _n(numbers, "time.adapter_training_s", tb["adapter_training_one_fold"]["seconds"], "r3_e6")
    _n(numbers, "time.fastsurfer_min", tb["fastsurfer_reference_labels"]["median_min"], "r3_e6")
    _n(numbers, "time.synthseg_min", tb["synthseg_plus_pseudolabels"]["median_min"], "r3_e6")
    lat = S.latency()
    _n(numbers, "time.cnn3d_forward_ms", lat.get("cnn3d", {}).get("cold_full_volume_forward", {}).get("median_ms"), "inference_latency")
    tables["labels"] = {"title": "Reference labels by cohort", "columns": ["Cohort", "Target", "Source", "Acquired on"],
                        "rows": [[r["cohort"], r["target"], r["source"], r["acquired_on"]] for r in S.label_table()],
                        "note": "ds006557 and external label files were re-checked row by row against their stated formula during the build."}
    mc = S.mc_dropout()
    _n(numbers, "mcdropout.oasis.coverage_pct", mc["oasis"]["empirical_coverage_pct"], "oasis_mc_dropout")
    _n(numbers, "mcdropout.real64mt.coverage_pct", mc["real64mt"]["gt_coverage_pct"], "real64mt_eval mc_dropout_ci")
    sens = S.sensitivity()
    _n(numbers, "sensitivity.max_abs_delta_mae", sens["max_abs_delta_mae"], "simulation_sensitivity")
    return {"sensitivity": sens}


def compute_all():
    numbers, tables = {}, {}
    ro = readouts(numbers, tables)
    multiseed_rows(numbers, tables)
    rep = reproducibility(numbers, tables)
    swin_nuisance(numbers, tables)
    oa = oasis(numbers, tables)
    pb = physics_vs_blur(numbers, tables)
    cf = conformal(numbers, tables, ro)
    ex = external(numbers, tables)
    mi = misc(numbers, tables)
    figdata = {"readouts": ro, "repro": rep, "oasis": oa, "physics_blur": pb, "conformal": cf, "external": ex, "misc": mi}
    return numbers, tables, figdata
