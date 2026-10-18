"""
Numbers, tables and figure data for the figures and tables carried over from the
round-2 manuscript: Figs. 2-18 and 20-23, Tables II, IV, VI, VIII and IX.

Everything is read from a committed result file through sources.load() or
recomputed here from that file's per-subject rows; the experiment's own saved
value is kept next to each recomputation so checks.py can compare them.

Not rebuilt here, with the reason:
  Fig. 1   axial MRI slices drawn from the raw scans (data/ is not in git)
  Fig. 19  MC Dropout on OASIS-1: the run saved only its summary
           (experiments/oasis_mc_dropout/results.json), not per-subject intervals
  Fig. 24  built in figures.py (fig24_external)
"""

import numpy as np
from scipy import stats

from . import sources as S
from .stats import (THRESHOLD, boot_ci, boot_dist, icc_c1, jackknife_conformal, loo_mean, mae,
                    permutation_replay)


def _n(numbers, name, value, source):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    numbers[name] = {"value": value, "source": source}


def _fisher_ci(r, n):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]


def _rows(rel, key="per_subject"):
    return S.load(rel)[key]


# ── OASIS-1 headline evaluation (one simulated draw, as reported) ──────────────

def oasis_headline(numbers):
    ft = _rows("oasis_finetune/finetune_results.json", "per_subject_test")
    y = np.array([r["true_nwbv"] for r in ft])
    p = np.array([r["pred_nwbv"] for r in ft])
    cdr = [r["cdr"] for r in ft]
    cnn = S.load("oasis_cnn_comparison/results.json")["CNN3D"]
    assert np.allclose([r["gt"] for r in cnn["per_subject"]], y), "OASIS CNN3D and ViT3D rows not aligned"
    pc = np.array([r["pred"] for r in cnn["per_subject"]])
    r, _ = stats.pearsonr(p, y)
    rc, _ = stats.pearsonr(pc, y)
    # The per-subject rows are stored at 4 decimals, so r, its Fisher CI and the RMSE are shown as the
    # experiment saved them (computed on unrounded predictions); the recomputed values are cross-checked.
    fr = S.load("oasis_finetune/finetune_results.json")
    ob = S.load("oasis_bootstrap/results.json")
    out = {"y": y, "pred": p, "cdr": cdr, "cnn_pred": pc, "r_recomputed": float(r), "cnn_r_recomputed": float(rc)}
    for k, v in (("r", fr["test_btf_pearson_r"]), ("r_ci", [fr["fisher_ci_95_lo"], fr["fisher_ci_95_hi"]]), ("mae", mae(p, y)),
                 ("mae_ci", ob["ViT3D"]["mae_ci"]), ("mae_ci_recomputed", boot_ci(mae, p, y)),
                 ("rmse", ob["ViT3D"]["rmse"]), ("bias", float(np.mean(p - y))),
                 ("cnn_r", cnn["pearson_r"]), ("cnn_r_ci", [cnn["fisher_ci_lo"], cnn["fisher_ci_hi"]]), ("cnn_mae", mae(pc, y)),
                 ("cnn_mae_ci", ob["CNN3D"]["mae_ci"]), ("cnn_rmse", cnn["rmse"])):
        _n(numbers, f"kept.oasis.{k}", v, "oasis_finetune per_subject_test / oasis_cnn_comparison")
        out[k] = v
    # Table IX T1: paired Wilcoxon on absolute errors, normal approximation with continuity correction
    # (the method that produced the reported p = 0.0037).
    t1 = stats.wilcoxon(np.abs(p - y), np.abs(pc - y), method="approx", correction=True).pvalue
    _n(numbers, "kept.stats.T1_vit_vs_cnn_oasis_p", float(t1), "oasis per-subject; Wilcoxon approx + continuity correction")
    out["t1_p"] = float(t1)
    # failure analysis (Fig. 23 A/B, Section V-O)
    order = np.argsort(y)
    terc = np.empty(len(y), int)
    terc[order[:13]], terc[order[13:25]], terc[order[25:]] = 0, 1, 2
    err = np.abs(p - y)
    out["tertile"] = terc
    out["tertile_mae"] = [float(err[terc == k].mean()) for k in range(3)]
    out["tertile_n"] = [int((terc == k).sum()) for k in range(3)]
    out["tertile_edges"] = [float(y[order[12]]), float(y[order[25]])]
    for k in range(3):
        _n(numbers, f"kept.failure.tertile{k}_mae", out["tertile_mae"][k], "oasis_finetune per_subject_test")
        _n(numbers, f"kept.failure.tertile{k}_n_err_gt_0.020", int(((err > THRESHOLD) & (terc == k)).sum()), "oasis_finetune per_subject_test")
    _n(numbers, "kept.failure.n_err_gt_0.050", int((err > 0.05).sum()), "oasis_finetune per_subject_test")
    _n(numbers, "kept.failure.n_err_gt_0.050_lowest_tertile", int(((err > 0.05) & (terc == 0)).sum()), "oasis_finetune per_subject_test")
    return out


def cdr(numbers, tables):
    sd = S.load("simulated_dementia/results.json")
    rows = sd["per_subject"]
    truth = {}
    for r in rows:
        if r["cdr"] is not None:
            truth.setdefault(float(r["cdr"]), []).append(r["true_nwbv"])
    g0, g5, g1 = (np.array(truth[k]) for k in (0.0, 0.5, 1.0))
    kw = stats.kruskal(g0, g5, g1)
    mw = stats.mannwhitneyu(g5, g1, alternative="greater")
    sp = np.sqrt(((len(g0) - 1) * g0.var(ddof=1) + (len(g1) - 1) * g1.var(ddof=1)) / (len(g0) + len(g1) - 2))
    for k, v in (("kw_H", kw.statistic), ("kw_p", kw.pvalue), ("mw_05_gt_10_p", mw.pvalue), ("cohen_d_0_vs_10", (g0.mean() - g1.mean()) / sp)):
        _n(numbers, f"kept.cdr_truth.{k}", float(v), "simulated_dementia per_subject (true nWBV by CDR)")
    # Table IV: CDR-stratified MAE on simulated inputs
    groups = [("No CDR label", "CDR_None"), ("CDR = 0.0", "CDR_0.0"), ("CDR = 0.5", "CDR_0.5"), ("CDR = 1.0", "CDR_1.0")]
    err = {}
    for r in rows:
        key = "CDR_None" if r["cdr"] is None else f"CDR_{float(r['cdr'])}"
        err.setdefault(key, []).append(abs(r["pred_nwbv"] - r["true_nwbv"]))
    tab, fig = [], []
    for label, key in groups:
        e = np.array(err[key])
        assert abs(e.mean() - sd["cdr_stratified"][key]["mae"]) < 6e-4, key
        g = sd["cdr_stratified"][key]  # saved by the experiment; std is the population SD (ddof = 0)
        assert abs(e.std() - g["mae_std"]) < 6e-4, key
        tab.append([label, len(e), f"{g['mae']:.3f} ± {g['mae_std']:.3f}", ("Below 0.020" if g["mae"] < THRESHOLD else f"{g['mae'] / THRESHOLD:.1f} × 0.020")])
        fig.append((label, len(e), float(g["mae"]), float(g["mae_std"])))
        _n(numbers, f"kept.cdr_mae.{key}", float(e.mean()), "simulated_dementia per_subject")
    yt = np.array([r["true_nwbv"] for r in rows]); pt = np.array([r["pred_nwbv"] for r in rows])
    r, pr = stats.pearsonr(pt, yt)
    tab.append(["Overall", len(rows), f"MAE = {mae(pt, yt):.3f}", f"Pearson r = {r:.3f} (p = {pr:.3f})"])
    tables["kept_table4_cdr"] = {"title": "Table IV. CDR-stratified MAE on physics-simulated 64 mT inputs (OASIS-1 test set, n = 38; no adaptation)",
                                 "columns": ["CDR group", "n", "MAE ± SD", "Interpretation"], "rows": tab,
                                 "note": "0.020 nWBV is a descriptive reference only. CDR = 1.0 has n = 2."}
    return {"truth_by_cdr": {0.0: g0, 0.5: g5, 1.0: g1}, "mw_p": float(mw.pvalue), "mae_groups": fig}


# ── real 64 mT (ds006557) ─────────────────────────────────────────────────────

def ds_headline(numbers, tables):
    lc = S.load("loocv_cross_session/results.json")
    rows = lc["per_subject"]
    subj = [r["subject"] for r in rows]
    y = np.array([r["true_nwbv"] for r in rows])
    hfe = np.array([r["pred_hfe"] for r in rows])
    hfc = np.array([r["pred_hfc"] for r in rows])
    age_map = {r["subject"]: r["age"] for r in S.load("r3_e2b_swin_verification/results.json")["per_subject"]}
    age = np.array([age_map[s] for s in subj])
    err = np.abs(hfe - y)
    out = {"subjects": subj, "y": y, "hfe": hfe, "hfc": hfc, "age": age, "err": err,
           "mae": mae(hfe, y), "mae_ci": lc["mae_bootstrap_ci_95"], "bias": float(np.mean(hfe - y)),
           "icc": icc_c1(hfc, hfe), "icc_ci": lc["icc_31_bootstrap_ci_95"],
           "boot": boot_dist(mae, hfe, y), "n_below": int((err < THRESHOLD).sum())}
    out["age_truth"] = stats.spearmanr(age, y)
    out["age_pred"] = stats.spearmanr(age, hfe)
    out["age_err"] = stats.spearmanr(age, err)
    for k, v in (("age_truth_rho", out["age_truth"].statistic), ("age_pred_rho", out["age_pred"].statistic),
                 ("age_pred_p", out["age_pred"].pvalue), ("age_err_rho", out["age_err"].statistic), ("age_err_p", out["age_err"].pvalue),
                 ("icc_same_fold", out["icc"]), ("bias", out["bias"])):
        _n(numbers, f"kept.ds.{k}", float(v), "loocv_cross_session per_subject (+ ages from r3_e2b)")
    d = hfe - y
    out["ba"] = {"mean": (hfe + y) / 2, "diff": d, "bias": float(d.mean()), "loa": [float(d.mean() - 1.96 * d.std(ddof=1)), float(d.mean() + 1.96 * d.std(ddof=1))]}
    out["perm"] = permutation_replay(hfc, hfe, y)
    pf = S.load("permutation_test/results.json")
    _n(numbers, "kept.perm.p_icc", out["perm"]["p_icc"], "replay of scripts/permutation_test_loocv.py")
    _n(numbers, "kept.perm.p_icc_file", pf["test1_intersession_icc"]["p_value"], "permutation_test")
    _n(numbers, "kept.perm.p_r", out["perm"]["p_r"], "replay of scripts/permutation_test_loocv.py")
    _n(numbers, "kept.perm.p_r_file", pf["test2_anatomical_tracking"]["p_value"], "permutation_test")
    worst = np.argsort(-err)[:5]
    tables["kept_table8_worst5"] = {"title": "Table VIII. Five largest LOOCV errors (real 64 mT, n = 23)",
                                    "columns": ["Subject", "True nWBV", "Predicted nWBV", "|Error|", "Above 0.020?"],
                                    "rows": [[subj[i], f"{y[i]:.3f}", f"{hfe[i]:.3f}", f"{err[i]:.3f}",
                                              f"{'Yes' if err[i] > THRESHOLD else 'No'} ({err[i] / THRESHOLD:.2f}×)"] for i in worst],
                                    "note": "0.020 nWBV is a descriptive reference only. N× = ratio of the error to 0.020."}
    out["worst"] = worst
    return out


def unadapted_and_comparators(numbers):
    vc = S.load("ablation_vit_vs_cnn/results.json")
    agree = S.load("pseudolabel_ablation/results.json")["label_agreement"]
    ss = np.array([r["synthseg"] for r in agree["per_subject"]]); fs = np.array([r["fastsurfer"] for r in agree["per_subject"]])
    out = {"synthseg": {"mae": mae(ss, fs), "r": stats.pearsonr(ss, fs)[0], "x": fs, "y": ss}}
    for name in ("ViT3D", "CNN3D"):
        rows = vc[name]["per_subject"]
        p = np.array([r["pred"] for r in rows]); y = np.array([r["gt"] for r in rows])
        out[name] = {"mae": mae(p, y), "r": stats.pearsonr(p, y)[0], "bias": float(np.mean(p - y))}
    for k in ("synthseg", "ViT3D", "CNN3D"):
        _n(numbers, f"kept.unadapted.{k}.mae", out[k]["mae"], "ablation_vit_vs_cnn / pseudolabel_ablation label_agreement")
        _n(numbers, f"kept.unadapted.{k}.r", float(out[k]["r"]), "ablation_vit_vs_cnn / pseudolabel_ablation label_agreement")
    lat = S.latency()
    tb = S.e6()
    out["latency_ms"] = {"vit": lat["headline_ms"], "cnn": lat["cnn3d"]["headline_ms"],
                         "synthseg": tb["synthseg_plus_pseudolabels"]["median_min"] * 60 * 1000}
    _n(numbers, "kept.latency.synthseg_min", tb["synthseg_plus_pseudolabels"]["median_min"], "r3_e6")
    return out


def physics_blur_unpaired(numbers):
    b = S.load("ablation_gaussblur/results.json")
    out = {}
    for k in ("physics_sim", "gaussian_blur"):
        p, y = np.array(b[k]["preds"]), np.array(b[k]["trues"])
        out[k] = {"pred": p, "true": y, "r": stats.pearsonr(p, y)[0], "r_ci": [b[k]["r_ci_lo"], b[k]["r_ci_hi"]], "mae": mae(p, y)}
    e1 = np.abs(out["physics_sim"]["pred"] - out["physics_sim"]["true"]); e2 = np.abs(out["gaussian_blur"]["pred"] - out["gaussian_blur"]["true"])
    out["mw_p"] = float(stats.mannwhitneyu(e1, e2).pvalue)
    _n(numbers, "kept.blur_unpaired.mw_p", out["mw_p"], "ablation_gaussblur")
    return out


def multiseed_and_ablation(numbers):
    ms = S.multiseed()
    seeds = [(s["seed"], s["mae"], icc_c1([r["pred_hfc"] for r in s["per_subject"]], [r["pred_hfe"] for r in s["per_subject"]])) for s in ms["per_seed"]]
    ab = S.load("ablation_adapter/results.json")["results"]
    lora = S.load("ablation_lora/results.json")["lora"]
    adapters = []
    for key, label, n_par in (("head_only", "Head only", "257"), ("ln_head", "LayerNorm\n+ head", "769"),
                              ("lora", "LoRA\n(r = 4)", f"{lora['trainable_params'] / 1000:.1f} K"), ("full_ft", "Full\nfine-tune", "4.23 M")):
        rows = lora["per_subject"] if key == "lora" else ab[key]["per_subject"]
        p = np.array([r["pred"] for r in rows]); y = np.array([r["true"] for r in rows])
        adapters.append({"key": key, "label": label, "params": n_par, "mae": mae(p, y), "ci": boot_ci(mae, p, y)})
        _n(numbers, f"kept.adapter.{key}.mae", mae(p, y), "ablation_adapter / ablation_lora")
    dino = S.load("dino_headline_loocv/results.json")
    return {"seeds": seeds, "adapters": adapters,
            "dino": {"oasis": dino["dino_oasis_test_mae"], "real": dino["dino_64mt_loocv_mae"]}}


def mc_and_conformal(numbers, ds):
    mc = S.mc_dropout()["real64mt"]["per_subject"]
    m = np.array([r["mean"] for r in mc]); sd = np.array([r["std"] for r in mc]); g = np.array([r["gt"] for r in mc])
    lo = np.array([r["ci_lo"] for r in mc]); hi = np.array([r["ci_hi"] for r in mc])
    levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    mc_cov = np.array([np.mean(np.abs(g - m) <= stats.norm.ppf(0.5 + lv / 2) * sd) for lv in levels])
    assert abs(np.mean((g >= lo) & (g <= hi)) - mc_cov[-1]) < 1e-9, "Gaussian and percentile 95% MC intervals disagree"
    w = hi - lo
    r, pr = stats.pearsonr(w, np.abs(m - g))
    _n(numbers, "kept.mc.coverage95", float(mc_cov[-1]), "real64mt_eval mc_dropout_ci")
    _n(numbers, "kept.mc.width_vs_error_r", float(r), "real64mt_eval mc_dropout_ci")
    _n(numbers, "kept.mc.width_vs_error_p", float(pr), "real64mt_eval mc_dropout_ci")
    conf_levels = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    conf_cov = np.array([jackknife_conformal(ds["hfe"], ds["y"], 1 - lv)["coverage"] for lv in conf_levels])
    c90 = jackknife_conformal(ds["hfe"], ds["y"], 0.10)
    fa = S.load("paper_statistics/failure_analysis.json")["real_64mt"]  # computed on unrounded intervals; shown as reported
    assert abs(fa["ci_vs_error_pearson_r"] - r) < 6e-4 and abs(fa["ci_vs_error_p"] - pr) < 6e-3
    return {"mean": m, "lo": lo, "hi": hi, "gt": g, "width": w, "err": np.abs(m - g), "levels": levels, "mc_cov": mc_cov,
            "r": float(r), "p": float(pr), "r_saved": fa["ci_vs_error_pearson_r"], "p_saved": fa["ci_vs_error_p"], "conf_levels": conf_levels, "conf_cov": conf_cov, "c90": c90}


def oasis_comparators(numbers, tables, oh):
    single = S.oasis_single_draw()
    par = S.params()
    rows = [("DINO", "ssl_dino", "vit"), ("MAE", "ssl_mae", "vit"), ("SimMIM", "ssl_simmim", "vit"),
            ("proxy\nregression\n(adopted)", None, "adopted"), ("contrast.", "ssl_contrastive", "vit"),
            ("CNN3D\n8.22M", "cnn", "other"), ("Swin-\nUNETR\n62.19M", "swin_unetr", "other"), ("UNETR\n92.69M", "unetr", "other")]
    bars = []
    for label, key, kind in rows:
        if key is None:
            m_, r_ = oh["mae"], oh["r"]
        elif key == "cnn":
            m_, r_ = oh["cnn_mae"], oh["cnn_r"]
        else:
            saved = S.load({"ssl_dino": "ssl_comparator_dino", "ssl_mae": "ssl_comparator_mae", "ssl_simmim": "ssl_comparator_simmim",
                            "ssl_contrastive": "ssl_comparator_contrastive", "swin_unetr": "arch_comparator_swin",
                            "unetr": "arch_comparator_unetr"}[key] + "/results.json")
            m_ = saved.get("mae", saved.get("test_mae"))
            r_ = saved.get("pearson_r", saved.get("test_pearson_r"))
            assert abs(m_ - mae(single[key]["pred"], single[key]["true"])) < 6e-4, key
        bars.append((label, float(m_), float(r_), kind))
    tab = [[lab.replace("\n", " "), f"{m_:.4f}", f"{r_:.4f}"] for lab, m_, r_, k in bars[:5]]
    tables["kept_table6_pretraining"] = {"title": "Table VI. Pretraining comparators on OASIS-1 (n = 38)", "columns": ["Objective", "OASIS-1 MAE", "Pearson r"],
                                         "rows": tab, "note": "Single simulated noise draw, as reported. The adopted row used the generic degradation and 30 pretraining epochs; the comparators used physics-simulated IXI inputs and 25 epochs."}
    _n(numbers, "kept.params.swin", par["SwinUNETR"]["params"], "model_param_counts")
    return bars


def stats_table(numbers, tables, oh, ds, mcd):
    sv = S.load("sim_validation/results.json")
    ncc_p = np.array([r["physics"]["ncc"] for r in sv]); ncc_a = np.array([r["arnold"]["ncc"] for r in sv])
    snr_p = stats.wilcoxon([r["physics"]["snr_err"] for r in sv], [r["arnold"]["snr_err"] for r in sv]).pvalue
    t2 = stats.wilcoxon(ncc_p, ncc_a).pvalue
    _n(numbers, "kept.fidelity.ncc_physics_mean", float(ncc_p.mean()), "sim_validation")
    _n(numbers, "kept.fidelity.ncc_blur_mean", float(ncc_a.mean()), "sim_validation")
    _n(numbers, "kept.fidelity.ncc_p", float(t2), "sim_validation")
    _, y, _, preds = S.ds006557()
    dino_e, head_e = np.abs(preds["dino_ln_head"] - y), np.abs(preds["vit_ln_head_headline"] - y)
    t4 = stats.wilcoxon(head_e, dino_e, method="approx", correction=True).pvalue
    t4_ci = boot_ci(lambda a, b: float(np.mean(a - b)), head_e, dino_e)
    _n(numbers, "kept.stats.T4_p", float(t4), "ds006557 per-subject (dino_headline_loocv vs loocv_cross_session)")
    _n(numbers, "kept.stats.T4_delta_ci", t4_ci, "ds006557 per-subject")
    v = lambda k: numbers[k]["value"]  # noqa: E731
    tab = [
        ["T1", "ViT3D vs CNN3D, OASIS-1", "paired Wilcoxon (normal approx.)", 38,
         f"MAE {oh['mae']:.4f} vs {oh['cnn_mae']:.4f}, p = {oh['t1_p']:.4f} (CNN3D better)"],
        ["T2", "Physics vs blur, image fidelity", "paired Wilcoxon", 23,
         f"NCC {ncc_p.mean():.3f} vs {ncc_a.mean():.3f}, p = {t2:.2f} (blur better); SNR error p = {snr_p:.0e} (blur closer)"],
        ["T3", "Physics vs blur, downstream nWBV, real 64 mT (matched, pooled over 3 seeds)", "paired Wilcoxon", 23,
         f"ΔMAE {v('e4.pooled.real64mt.delta_mae'):+.4f}, p = {v('e4.pooled.real64mt.wilcoxon_p'):.2f} (n.s.)"],
        ["T4", "Proxy-regression vs DINO pretraining, real 64 mT LOOCV", "paired Wilcoxon (normal approx.)", 23,
         f"MAE {head_e.mean():.4f} vs {dino_e.mean():.4f}, p = {t4:.3f} (n.s.); bootstrap ΔMAE {np.mean(head_e - dino_e):+.4f} [{t4_ci[0]:+.3f}, {t4_ci[1]:+.3f}]"],
        ["P1", "Inter-session ICC, single frozen model, vs permutation null", "permutation, 20,000", 23,
         f"ICC(C,1) {v('repro.icc_frozen_single_model'):.3f}, p = {v('repro.icc_frozen_perm_p'):.3f} (same-fold test, ICC {ds['perm']['obs_icc']:.4f}, withdrawn)"],
        ["P2", "Prediction vs true nWBV", "permutation, 20,000", 23, f"r = {ds['perm']['obs_r']:.3f}, p = {ds['perm']['p_r']:.3f} (n.s.)"],
        ["S1", "Swin-UNETR frozen + ridge vs adapted ViT3D, real 64 mT", "paired Wilcoxon", 23,
         f"MAE {v('ds006557.ridge_swin_gap.mae'):.4f} vs {v('ds006557.vit_ln_head_headline.mae'):.4f}, p = {v('ds006557.ridge_swin_gap.vs_ref_wilcoxon_p'):.3f}"],
        ["S2", "Swin-UNETR readout, labels permuted through the full nested procedure", "permutation, 1,000", 23,
         f"p = {S.e2b()['permutation_null']['p_mae']:.3f}"],
        ["S3", "Adapted ViT3D vs leave-one-out constant mean", "paired Wilcoxon", 23,
         f"MAE {v('ds006557.vit_ln_head_headline.mae'):.4f} vs {v('ds006557.constant_loo_mean.mae'):.4f}, p = {v('ds006557.vit_ln_head_headline.vs_constant_wilcoxon_p'):.2f} (n.s.)"],
    ]
    tables["kept_table9_stats"] = {"title": "Table IX. Summary of statistical comparisons", "columns": ["#", "Comparison", "Test", "n", "Result"],
                                   "rows": tab, "note": "T1 and T4 use the normal approximation with continuity correction (the method of the original analysis); the exact test gives smaller p."}


def summary_table(numbers, tables, oh, un, ds):
    v = lambda k: numbers[k]["value"]  # noqa: E731
    cdr_rows = [(k, numbers[f"kept.cdr_mae.{k}"]["value"]) for k in ("CDR_0.0", "CDR_0.5", "CDR_1.0", "CDR_None")]
    tab = [
        ["(A) OASIS-1 test set (n = 38)", "ViT3D (ours)", f"{oh['r']:.3f} [{oh['r_ci'][0]:.3f}–{oh['r_ci'][1]:.3f}]",
         f"{oh['mae']:.3f} [{oh['mae_ci'][0]:.3f}–{oh['mae_ci'][1]:.3f}]", f"{oh['rmse']:.3f}"],
        ["", "CNN3D", f"{oh['cnn_r']:.3f} [{oh['cnn_r_ci'][0]:.3f}–{oh['cnn_r_ci'][1]:.3f}]",
         f"{oh['cnn_mae']:.3f} [{oh['cnn_mae_ci'][0]:.3f}–{oh['cnn_mae_ci'][1]:.3f}]", f"{oh['cnn_rmse']:.3f}"],
        ["(B) Real 64 mT, unadapted (n = 23)", "SynthSeg+", f"{un['synthseg']['r']:.3f}", f"{un['synthseg']['mae']:.3f}", "—"],
        ["", "CNN3D (no adapt.)", f"{un['CNN3D']['r']:.3f}", f"{un['CNN3D']['mae']:.3f}", "—"],
        ["", "ViT3D (no adapt.)", f"{un['ViT3D']['r']:.3f}", f"{un['ViT3D']['mae']:.3f}", "—"],
        ["(C) Real 64 mT, LOOCV-adapted (n = 23, cross-session)", "ViT3D (adapt.)", f"{v('ds006557.vit_ln_head_headline.pearson_r'):.2f}",
         f"{ds['mae']:.4f} [{ds['mae_ci'][0]:.3f}–{ds['mae_ci'][1]:.3f}]", f"{float(np.sqrt(np.mean((ds['hfe'] - ds['y']) ** 2))):.3f}"],
    ]
    for key, label in (("constant_loo_mean", "Constant (leave-one-out mean)"), ("age_only_ridge", "Age only (ridge)"),
                       ("ridge_cnn_globalpool256", "CNN3D frozen features + ridge"), ("ridge_swin_gap", "Swin-UNETR frozen features + ridge")):
        ci = v(f"ds006557.{key}.mae_ci95")
        r_ = v(f"ds006557.{key}.pearson_r")
        tab.append(["", label, "—" if r_ != r_ else f"{r_:.2f}", f"{v(f'ds006557.{key}.mae'):.4f} [{ci[0]:.4f}–{ci[1]:.4f}]", "—"])
    tab.append(["(D) Inter-session reliability", "ICC(C,1) HFC vs HFE, same fold model", f"{ds['icc']:.3f} [{ds['icc_ci'][0]:.3f}–{ds['icc_ci'][1]:.3f}]", "—", "—"])
    ci = v("repro.icc_frozen_ci95")
    tab.append(["", "ICC(C,1) HFC vs HFE, single frozen model", f"{v('repro.icc_frozen_single_model'):.3f} [{ci[0]:.3f}–{ci[1]:.3f}]", "—", "—"])
    for k, m_ in cdr_rows:
        tab.append(["(E) Pathology stress test, simulated 64 mT (n = 38, no adapt.)" if k == "CDR_0.0" else "",
                    {"CDR_None": "No CDR label"}.get(k, k.replace("_", " = ")), "—", f"{m_:.3f}", "—"])
    tables["kept_table2_summary"] = {"title": "Table II. Summary of results by evaluation context", "columns": ["Context", "Method", "Pearson r [95% CI]", "MAE [95% CI]", "RMSE"],
                                     "rows": tab, "note": "OASIS-1 rows: one simulated noise draw, as reported (fixed-draw means are in tables/oasis.md). 0.020 nWBV is a descriptive reference only."}


def compute_kept(numbers, tables):
    oh = oasis_headline(numbers)
    cd = cdr(numbers, tables)
    ds = ds_headline(numbers, tables)
    un = unadapted_and_comparators(numbers)
    pb = physics_blur_unpaired(numbers)
    ma = multiseed_and_ablation(numbers)
    mcd = mc_and_conformal(numbers, ds)
    comp = oasis_comparators(numbers, tables, oh)
    stats_table(numbers, tables, oh, ds, mcd)
    summary_table(numbers, tables, oh, un, ds)
    pl = S.load("pseudolabel_ablation/results.json")["adapter_results"]
    sens = S.sensitivity()
    return {"oasis": oh, "cdr": cd, "ds": ds, "unadapted": un, "blur": pb, "ms": ma, "mc": mcd, "comparators": comp,
            "pseudolabel": pl, "sensitivity": sens}
