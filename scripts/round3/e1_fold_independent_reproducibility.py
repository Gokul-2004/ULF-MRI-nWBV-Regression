"""
E1 — Fold-independent reproducibility and nuisance controls (R2.3, R4.1)
=======================================================================
The headline ICC (0.615) pairs HFC and HFE predictions made by the SAME
fold-specific adapter, so a fold-level output offset is shared by both
sessions and inflates agreement (a leave-one-out constant-mean predictor
scores ICC = 1). This script reports agreement measures that remove that
shared-model component, and tests stable nuisance explanations.

  (a) Frozen, unadapted Stage-2 model (checkpoints/oasis_finetuned.pt): ONE
      model for all subjects and both sessions, so no fold structure.
      ICC(C,1), ICC(A,1), Pearson r, permutation p (re-pairing), bootstrap CI.
      Per-subject predictions are saved for reuse by E2.
  (b) Cross-seed ICC from experiments/multiseed_loocv: HFC from seed a vs
      HFE from seed b (a != b) — different fold models for the two sessions.
  (c) Head-size control: ICC after residualising each session's predictions
      on FastSurfer MaskVol or SynthSeg+ TIV.
  (d) Image-QC nuisance covariates per scan (head position in the field of
      view, head-mask volume, intensity statistics, background-noise SNR).
      ICC after residualising on QC covariates, compared with a null in which
      the covariate rows are randomly permuted across subjects (to separate a
      real nuisance effect from the loss of degrees of freedom).

Output: experiments/r3_e1_reproducibility/results.json
        experiments/r3_frozen_predictions/predictions.json
"""

import itertools
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage, stats

from common import (EXP, ROOT, SEED, boot_ci, icc_a1, icc_c1, lcs, maskvol,
                    perm_p_pairing, provenance, residualise, save_json,
                    scan_path, subjects_and_gt, synthseg_tiv)

OUT = EXP / "r3_e1_reproducibility" / "results.json"
PRED_OUT = EXP / "r3_frozen_predictions" / "predictions.json"
N_PERM = 20_000


def agreement(a, b, label):
    a, b = np.asarray(a, float), np.asarray(b, float)
    r, p_r = stats.pearsonr(a, b)
    return {
        "label": label,
        "n": len(a),
        "icc_c1": round(icc_c1(a, b), 4),
        "icc_c1_ci95": [round(x, 4) for x in boot_ci(icc_c1, a, b)],
        "icc_a1": round(icc_a1(a, b), 4),
        "icc_c1_perm_p": round(perm_p_pairing(icc_c1, a, b, N_PERM), 5),
        "pearson_r": round(float(r), 4),
        "pearson_p": round(float(p_r), 5),
        "mean_session_a": round(float(a.mean()), 4),
        "mean_session_b": round(float(b.mean()), 4),
        "sd_session_a": round(float(a.std(ddof=1)), 4),
        "sd_session_b": round(float(b.std(ddof=1)), 4),
    }


# ── (a) frozen model ──────────────────────────────────────────────────────────

def frozen_predictions(subjects, gt):
    model = lcs.build_base_model()
    model.eval()
    rows = []
    with torch.no_grad():
        for s in subjects:
            rows.append({"subject": s, "true_nwbv": gt[s],
                         "pred_hfc": round(lcs.predict(model, s, "HFC"), 5),
                         "pred_hfe": round(lcs.predict(model, s, "HFE"), 5)})
            print(f"  {s}  HFC={rows[-1]['pred_hfc']:.4f}  HFE={rows[-1]['pred_hfe']:.4f}")
    return rows


# ── (d) image QC ──────────────────────────────────────────────────────────────

def qc_metrics(path: Path) -> dict:
    img = nib.load(str(path))
    raw = img.get_fdata(dtype=np.float32)
    if raw.ndim == 4:
        raw = raw[..., 0]
    zooms = np.array(img.header.get_zooms()[:3], float)
    # head mask: Otsu-like threshold on the raw image, largest component, filled
    thr = _otsu(raw)
    mask = raw > thr
    lab, nlab = ndimage.label(mask)
    if nlab > 1:
        sizes = ndimage.sum(mask, lab, range(1, nlab + 1))
        mask = lab == (1 + int(np.argmax(sizes)))
    mask = ndimage.binary_fill_holes(mask)
    com_vox = np.array(ndimage.center_of_mass(mask))
    centre_vox = (np.array(raw.shape) - 1) / 2.0
    offset_mm = (com_vox - centre_vox) * zooms
    # background: voxels far outside the dilated head mask
    bg = ~ndimage.binary_dilation(mask, iterations=4)
    bg_vals = raw[bg]
    fg_vals = raw[mask]
    noise_sd = float(np.std(bg_vals)) if bg_vals.size > 100 else float("nan")
    return {
        "com_offset_x_mm": float(offset_mm[0]),
        "com_offset_y_mm": float(offset_mm[1]),
        "com_offset_z_mm": float(offset_mm[2]),
        "head_mask_volume_ml": float(mask.sum() * np.prod(zooms) / 1000.0),
        "fg_mean": float(fg_vals.mean()),
        "fg_sd": float(fg_vals.std()),
        "fg_p99": float(np.percentile(fg_vals, 99)),
        "bg_mean": float(bg_vals.mean()),
        "bg_sd": noise_sd,
        "snr_fg_mean_over_bg_sd": float(fg_vals.mean() / noise_sd) if noise_sd > 0 else float("nan"),
        "bg_fraction": float(bg.mean()),
    }


def _otsu(vol, bins=256):
    v = vol[np.isfinite(vol)].ravel()
    hist, edges = np.histogram(v, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centres) - np.cumsum(hist * centres)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centres[int(np.argmax(between))])


QC_KEYS = ["com_offset_x_mm", "com_offset_y_mm", "com_offset_z_mm",
           "head_mask_volume_ml", "fg_mean", "fg_sd", "bg_sd", "snr_fg_mean_over_bg_sd"]


def partial_icc(a, b, Xa, Xb):
    return icc_c1(residualise(a, Xa), residualise(b, Xb))


def partial_icc_with_null(a, b, Xa, Xb, n_null=5_000, seed=SEED):
    """Observed ICC after residualising each session on its own covariates,
    plus the null distribution when covariate rows are permuted (same
    permutation for both sessions, preserving the HFC/HFE covariate link)."""
    rng = np.random.default_rng(seed)
    obs = partial_icc(a, b, Xa, Xb)
    null = []
    for _ in range(n_null):
        idx = rng.permutation(len(a))
        null.append(partial_icc(a, b, Xa[idx], Xb[idx]))
    null = np.array(null)
    return {
        "icc_after": round(obs, 4),
        "null_mean_icc_after_random_covariates": round(float(null.mean()), 4),
        "null_ci95": [round(float(np.percentile(null, 2.5)), 4), round(float(np.percentile(null, 97.5)), 4)],
        "p_drop_larger_than_random": round(float((np.sum(null <= obs) + 1) / (n_null + 1)), 4),
    }


def main():
    subjects, gt = subjects_and_gt()
    print(f"E1: {len(subjects)} subjects")
    res = {"experiment": "r3_e1_fold_independent_reproducibility",
           "answers": ["R2.3", "R4.1"],
           "icc_definitions": {
               "icc_c1": "ICC(3,1) consistency, two-way mixed, single measures (McGraw & Wong ICC(C,1)); the form computed by the paper's loocv_cross_session.compute_icc31",
               "icc_a1": "ICC(A,1) absolute agreement"},
           "provenance": provenance("scripts/round3/e1_fold_independent_reproducibility.py")}

    # (a)
    print("\n(a) frozen unadapted model on both sessions")
    rows = frozen_predictions(subjects, gt)
    save_json(PRED_OUT, {"model": "checkpoints/oasis_finetuned.pt (Stage 2, no 64 mT adaptation)",
                         "input": "acq-axi_T2w, headline preprocessing (loocv_cross_session.load_scan)",
                         "per_subject": rows,
                         "provenance": provenance("scripts/round3/e1_fold_independent_reproducibility.py")})
    fh = np.array([r["pred_hfc"] for r in rows])
    fe = np.array([r["pred_hfe"] for r in rows])
    tr = np.array([r["true_nwbv"] for r in rows])
    res["a_frozen_model"] = agreement(fh, fe, "frozen oasis_finetuned.pt: HFC vs HFE")
    r_t, p_t = stats.pearsonr(fe, tr)
    res["a_frozen_model"]["hfe_vs_truth_pearson_r"] = round(float(r_t), 4)
    res["a_frozen_model"]["hfe_vs_truth_pearson_p"] = round(float(p_t), 4)
    res["a_frozen_model"]["hfe_mae_vs_truth"] = round(float(np.mean(np.abs(fe - tr))), 4)

    # headline for reference
    head = json.loads((EXP / "loocv_cross_session" / "results.json").read_text())["per_subject"]
    hh = np.array([r["pred_hfc"] for r in head])
    he = np.array([r["pred_hfe"] for r in head])
    assert [r["subject"] for r in head] == subjects
    res["reference_headline_same_fold_model"] = agreement(hh, he, "headline LOOCV: same fold model predicts HFC and HFE")
    loo_mean = np.array([np.mean(np.delete(tr, i)) for i in range(len(tr))])
    res["reference_loo_constant_mean"] = {
        "note": "A leave-one-out constant-mean 'predictor' outputs the same value for both sessions; ICC = 1 by construction.",
        "icc_c1": round(icc_c1(loo_mean, loo_mean), 4)}

    # (b) cross-seed
    print("\n(b) cross-seed ICC")
    ms = json.loads((EXP / "multiseed_loocv" / "results.json").read_text())["per_seed"]
    pairs = []
    for sa, sb in itertools.permutations(ms, 2):
        a = np.array([r["pred_hfc"] for r in sa["per_subject"]])
        b = np.array([r["pred_hfe"] for r in sb["per_subject"]])
        pairs.append({"hfc_seed": sa["seed"], "hfe_seed": sb["seed"],
                      "icc_c1": round(icc_c1(a, b), 4),
                      "pearson_r": round(float(stats.pearsonr(a, b)[0]), 4),
                      "icc_c1_perm_p": round(perm_p_pairing(icc_c1, a, b, 5_000), 4)})
    within = [{"seed": s["seed"], "icc_c1": round(icc_c1(
        np.array([r["pred_hfc"] for r in s["per_subject"]]),
        np.array([r["pred_hfe"] for r in s["per_subject"]])), 4)} for s in ms]
    res["b_cross_seed"] = {
        "within_seed": within,
        "within_seed_mean_icc_c1": round(float(np.mean([w["icc_c1"] for w in within])), 4),
        "cross_seed_pairs": pairs,
        "cross_seed_mean_icc_c1": round(float(np.mean([p["icc_c1"] for p in pairs])), 4),
        "cross_seed_range_icc_c1": [min(p["icc_c1"] for p in pairs), max(p["icc_c1"] for p in pairs)],
        "cross_seed_mean_pearson_r": round(float(np.mean([p["pearson_r"] for p in pairs])), 4),
        "n_pairs_perm_p_below_0.05": sum(p["icc_c1_perm_p"] < 0.05 for p in pairs),
        "n_pairs": len(pairs)}

    # (c) head size
    print("\n(c) head-size partialling")
    mv = maskvol()
    tiv = synthseg_tiv()
    res["c_head_size"] = {}
    for name, cov in [("fastsurfer_maskvol", mv), ("synthseg_tiv", tiv)]:
        x = np.array([cov[s] for s in subjects])
        block = {}
        for lab, a, b in [("headline", hh, he), ("frozen", fh, fe)]:
            ra, rb = residualise(a, x), residualise(b, x)
            block[lab] = {"icc_c1_before": round(icc_c1(a, b), 4),
                          "icc_c1_after": round(icc_c1(ra, rb), 4),
                          "icc_c1_after_ci95": [round(v, 4) for v in boot_ci(lambda p, q, z: icc_c1(residualise(p, z), residualise(q, z)), a, b, x, n_boot=5_000)],
                          "r_pred_hfe_vs_covariate": round(float(stats.pearsonr(b, x)[0]), 4),
                          "r_pred_hfe_vs_covariate_p": round(float(stats.pearsonr(b, x)[1]), 4)}
        res["c_head_size"][name] = block

    # (d) QC
    print("\n(d) image-QC covariates")
    qc = {}
    for s in subjects:
        qc[s] = {ses: qc_metrics(scan_path(s, ses)) for ses in ("HFC", "HFE")}
    Xc = np.array([[qc[s]["HFC"][k] for k in QC_KEYS] for s in subjects])
    Xe = np.array([[qc[s]["HFE"][k] for k in QC_KEYS] for s in subjects])
    cov_tbl = {}
    for j, k in enumerate(QC_KEYS):
        cov_tbl[k] = {
            "between_session_r": round(float(stats.pearsonr(Xc[:, j], Xe[:, j])[0]), 4),
            "hfc_mean": round(float(Xc[:, j].mean()), 3), "hfe_mean": round(float(Xe[:, j].mean()), 3),
            "paired_wilcoxon_p": round(float(stats.wilcoxon(Xc[:, j], Xe[:, j]).pvalue), 4),
            "r_with_headline_pred_hfe": round(float(stats.pearsonr(Xe[:, j], he)[0]), 4),
            "p_with_headline_pred_hfe": round(float(stats.pearsonr(Xe[:, j], he)[1]), 4),
            "r_with_frozen_pred_hfe": round(float(stats.pearsonr(Xe[:, j], fe)[0]), 4),
            "p_with_frozen_pred_hfe": round(float(stats.pearsonr(Xe[:, j], fe)[1]), 4),
            "r_with_truth": round(float(stats.pearsonr(Xe[:, j], tr)[0]), 4),
        }
    # standardise, then partial on (i) all QC covariates, (ii) position only, (iii) single strongest
    def z(X):
        return (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    Zc, Ze = z(Xc), z(Xe)
    pos = [QC_KEYS.index(k) for k in ("com_offset_x_mm", "com_offset_y_mm", "com_offset_z_mm")]
    partial = {}
    for lab, a, b in [("headline", hh, he), ("frozen", fh, fe)]:
        partial[lab] = {
            "icc_c1_before": round(icc_c1(a, b), 4),
            "all_qc_covariates": partial_icc_with_null(a, b, Zc, Ze),
            "head_position_only": partial_icc_with_null(a, b, Zc[:, pos], Ze[:, pos]),
            "head_mask_volume_only": partial_icc_with_null(a, b, Zc[:, [3]], Ze[:, [3]]),
            "intensity_and_snr_only": partial_icc_with_null(a, b, Zc[:, 4:], Ze[:, 4:]),
        }
    res["d_image_qc"] = {"qc_keys": QC_KEYS,
                         "method": "Head mask = Otsu threshold on raw intensities, largest component, holes filled. "
                                   "Position = mask centre of mass minus FOV centre (mm; affines are identical across sessions). "
                                   "Noise = SD of raw voxels outside the 4-voxel-dilated head mask.",
                         "covariate_table": cov_tbl,
                         "partial_icc": partial,
                         "per_scan": qc}

    save_json(OUT, res)

    # console summary
    print("\nSUMMARY")
    print(f"  headline same-fold ICC(C,1) = {res['reference_headline_same_fold_model']['icc_c1']}")
    print(f"  frozen model ICC(C,1)       = {res['a_frozen_model']['icc_c1']}  perm p = {res['a_frozen_model']['icc_c1_perm_p']}  "
          f"(SD HFC {res['a_frozen_model']['sd_session_a']}, HFE {res['a_frozen_model']['sd_session_b']})")
    print(f"  cross-seed ICC(C,1) mean    = {res['b_cross_seed']['cross_seed_mean_icc_c1']} range {res['b_cross_seed']['cross_seed_range_icc_c1']}")
    for k, v in res["c_head_size"].items():
        print(f"  {k}: headline {v['headline']['icc_c1_before']} -> {v['headline']['icc_c1_after']}; frozen {v['frozen']['icc_c1_before']} -> {v['frozen']['icc_c1_after']}")
    for lab in ("headline", "frozen"):
        p = partial[lab]
        print(f"  QC {lab}: all {p['all_qc_covariates']}  position {p['head_position_only']['icc_after']}")


if __name__ == "__main__":
    main()
