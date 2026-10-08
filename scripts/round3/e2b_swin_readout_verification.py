"""
E2b — Stress tests for the frozen Swin-UNETR feature readout found in E2
=======================================================================
E2 found that a nested ridge on the 8-d GAP features of the OASIS-trained
Swin-UNETR (checkpoints/swin_oasis.pt; trained on OASIS-1 only), fitted on HFC
of 22 subjects and applied to the held-out HFE scan, reaches MAE 0.0075,
r 0.83. Before any claim, this script tests whether that survives:

  1. Label-permutation null of the ENTIRE nested cross-session procedure
     (labels shuffled across subjects, features fixed) -> p for MAE and r.
  2. Fixed-alpha sensitivity (no inner selection).
  3. Session direction: HFC->HFE (headline), HFE->HFC, HFC->HFC (LOO), HFE->HFE (LOO).
  4. Nuisance controls: does the readout still predict truth after partialling
     out head size (FastSurfer MaskVol, SynthSeg TIV, 64 mT head-mask volume),
     head position in the FOV (z offset, the E1 nuisance), age and sex?
     Also: readouts on the nuisance variables alone.
  5. Cross-session agreement of the readout (ICC of HFC vs HFE predictions
     made by the same fold readout, and by readouts fitted on different
     sessions).
  6. Frozen Swin scalar output + affine (no feature readout).

Output: experiments/r3_e2b_swin_verification/results.json
"""

import csv
import json

import numpy as np
from scipy import stats

from common import (DS_DIR, EXP, SEED, icc_c1, maskvol, provenance,
                    residualise, save_json, subjects_and_gt, synthseg_tiv)
from e2_nonneural_baselines import ALPHAS, nested_alpha, ridge_fit_predict

OUT = EXP / "r3_e2b_swin_verification" / "results.json"
N_PERM = 1000


def xsession(Xtr_all, Xte_all, y, alpha=None):
    n = len(y)
    p = np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i
        a = nested_alpha(Xtr_all[tr], y[tr]) if alpha is None else alpha
        p[i] = ridge_fit_predict(Xtr_all[tr], y[tr], Xte_all[i:i + 1], a)[0]
    return p


def summ(p, y):
    r, pr = stats.pearsonr(p, y)
    loo = np.array([np.delete(y, i).mean() for i in range(len(y))])
    return {"mae": round(float(np.mean(np.abs(p - y))), 5), "pearson_r": round(float(r), 4),
            "pearson_p": round(float(pr), 6),
            "mse_skill_vs_loo_mean": round(float(1 - np.mean((p - y) ** 2) / np.mean((loo - y) ** 2)), 4),
            "calibration_slope": round(float(stats.linregress(p, y).slope), 3)}


def partial_r(x, y, Z):
    return stats.pearsonr(residualise(x, Z), residualise(y, Z))


def main():
    subjects, gt = subjects_and_gt()
    y = np.array([gt[s] for s in subjects])
    F = dict(np.load(EXP / "r3_e2_nonneural_baselines" / "features.npz"))
    Xc, Xe = F["HFC_swin"], F["HFE_swin"]
    res = {"experiment": "r3_e2b_swin_readout_verification",
           "features": "8-d GAP of SwinUNETR(feature_size=48) trained on OASIS-1 only (checkpoints/swin_oasis.pt, scripts/arch_comparator_swin.py); frozen",
           "provenance": provenance("scripts/round3/e2b_swin_readout_verification.py")}

    base = xsession(Xc, Xe, y)
    res["headline_design_hfc_to_hfe"] = summ(base, y)
    print("HFC->HFE", res["headline_design_hfc_to_hfe"])

    # 1. permutation null of the whole procedure
    rng = np.random.default_rng(SEED)
    null_mae, null_r = [], []
    for k in range(N_PERM):
        yp = y[rng.permutation(len(y))]
        pp = xsession(Xc, Xe, yp)
        null_mae.append(np.mean(np.abs(pp - yp)))
        null_r.append(stats.pearsonr(pp, yp)[0])
        if (k + 1) % 200 == 0:
            print(f"  perm {k+1}/{N_PERM}")
    obs_mae, obs_r = np.mean(np.abs(base - y)), stats.pearsonr(base, y)[0]
    res["permutation_null"] = {
        "n_perm": N_PERM,
        "null_mae_mean": round(float(np.mean(null_mae)), 5),
        "null_mae_p2.5": round(float(np.percentile(null_mae, 2.5)), 5),
        "p_mae": round(float((np.sum(np.array(null_mae) <= obs_mae) + 1) / (N_PERM + 1)), 4),
        "null_r_p97.5": round(float(np.percentile(null_r, 97.5)), 4),
        "p_r": round(float((np.sum(np.array(null_r) >= obs_r) + 1) / (N_PERM + 1)), 4)}
    print("perm", res["permutation_null"])

    # 2. fixed alphas
    res["fixed_alpha"] = {f"{a:g}": summ(xsession(Xc, Xe, y, a), y) for a in (0.01, 0.1, 1, 10, 100)}

    # 3. directions
    def loo_same(X):
        return xsession(X, X, y)
    res["session_directions"] = {"HFC_to_HFE": summ(base, y),
                                 "HFE_to_HFC": summ(xsession(Xe, Xc, y), y),
                                 "HFC_to_HFC_loo": summ(loo_same(Xc), y),
                                 "HFE_to_HFE_loo": summ(loo_same(Xe), y)}

    # 4. nuisance controls
    e1 = json.loads((EXP / "r3_e1_reproducibility" / "results.json").read_text())["d_image_qc"]["per_scan"]
    mv, tiv = maskvol(), synthseg_tiv()
    part = {}
    with open(DS_DIR / "participants.tsv") as f:
        pt = {("sub-" + r["participant_id"] if not r["participant_id"].startswith("sub-") else r["participant_id"]): r
              for r in csv.DictReader(f, delimiter="\t")}
    age = np.array([float(pt[s]["age"]) for s in subjects])
    sex = np.array([1.0 if str(pt[s].get("sex", "")).upper().startswith("M") else 0.0 for s in subjects])
    nuis = {
        "fastsurfer_maskvol": np.array([mv[s] for s in subjects]),
        "synthseg_tiv": np.array([tiv[s] for s in subjects]),
        "head_mask_volume_hfe": np.array([e1[s]["HFE"]["head_mask_volume_ml"] for s in subjects]),
        "z_offset_hfe": np.array([e1[s]["HFE"]["com_offset_z_mm"] for s in subjects]),
        "age": age, "sex": sex}
    for k, z in nuis.items():
        r, p = partial_r(base, y, z)
        part[k] = {"partial_r_pred_truth": round(float(r), 4), "p": round(float(p), 5),
                   "r_pred_vs_nuisance": round(float(stats.pearsonr(base, z)[0]), 4),
                   "r_truth_vs_nuisance": round(float(stats.pearsonr(y, z)[0]), 4)}
    Zall = np.column_stack(list(nuis.values()))
    r, p = partial_r(base, y, Zall)
    part["all_jointly"] = {"partial_r_pred_truth": round(float(r), 4), "p": round(float(p), 5)}
    Zns = np.column_stack([nuis[k] for k in nuis if k not in ("age",)])
    r, p = partial_r(base, y, Zns)
    part["all_except_age"] = {"partial_r_pred_truth": round(float(r), 4), "p": round(float(p), 5)}
    res["nuisance_partial_correlations"] = part
    # readouts on nuisance variables alone (same cross-session design where session-specific)
    p_age = xsession(age[:, None], age[:, None], y)
    res["nuisance_only_readouts"] = {
        "age_only": summ(p_age, y),
        "head_mask_volume_hfc_to_hfe": summ(xsession(
            np.array([[e1[s]["HFC"]["head_mask_volume_ml"]] for s in subjects]),
            nuis["head_mask_volume_hfe"][:, None], y), y),
        "maskvol_only": summ(xsession(nuis["fastsurfer_maskvol"][:, None], nuis["fastsurfer_maskvol"][:, None], y), y),
        "swin_plus_age_vs_age": None}
    both_c = np.column_stack([Xc, age]); both_e = np.column_stack([Xe, age])
    p_swin_age = xsession(both_c, both_e, y)
    res["nuisance_only_readouts"]["swin_plus_age_vs_age"] = {
        "swin_plus_age": summ(p_swin_age, y),
        "age_only": res["nuisance_only_readouts"]["age_only"]}

    # 5. agreement
    n = len(y)
    pc_same = np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i
        a = nested_alpha(Xc[tr], y[tr])
        pc_same[i] = ridge_fit_predict(Xc[tr], y[tr], Xc[i:i + 1], a)[0]
    res["agreement"] = {"icc_c1_same_fold_readout_hfc_vs_hfe": round(icc_c1(pc_same, base), 4),
                        "feature_between_session_r_per_dim": [round(float(stats.pearsonr(Xc[:, j], Xe[:, j])[0]), 3) for j in range(Xc.shape[1])],
                        "feature_vs_truth_r_hfe_per_dim": [round(float(stats.pearsonr(Xe[:, j], y)[0]), 3) for j in range(Xc.shape[1])]}

    # 6. scalar output
    try:
        import torch
        from e2_nonneural_baselines import swin_model
        from common import lcs
        m = swin_model()
        with torch.no_grad():
            sc = np.array([float(m.head(torch.tensor(Xc[i])[None].float())) for i in range(n)])
            se = np.array([float(m.head(torch.tensor(Xe[i])[None].float())) for i in range(n)])
        aff = np.zeros(n)
        for i in range(n):
            tr = np.arange(n) != i
            b, a0 = np.polyfit(sc[tr], y[tr], 1)
            aff[i] = a0 + b * se[i]
        res["swin_scalar_output"] = {"unadapted_hfe": summ(se, y) | {"bias": round(float(np.mean(se - y)), 4)},
                                     "affine_hfc_to_hfe": summ(aff, y),
                                     "icc_c1_unadapted_hfc_vs_hfe": round(icc_c1(sc, se), 4)}
    except Exception as e:
        res["swin_scalar_output"] = {"error": str(e)}

    res["per_subject"] = [{"subject": s, "true_nwbv": float(y[i]), "swin_ridge_hfe": round(float(base[i]), 5),
                           "swin_ridge_hfc_same_fold": round(float(pc_same[i]), 5),
                           "age": float(age[i]), "age_only_ridge": round(float(p_age[i]), 5),
                           "swin_plus_age_ridge": round(float(p_swin_age[i]), 5)} for i, s in enumerate(subjects)]
    save_json(OUT, res)
    for k in ("session_directions", "nuisance_partial_correlations", "swin_scalar_output", "fixed_alpha"):
        print(k, json.dumps(res[k], indent=0)[:1500])


if __name__ == "__main__":
    main()
