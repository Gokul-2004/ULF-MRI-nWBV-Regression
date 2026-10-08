"""
E3 — External validation with the HEADLINE adapter recipe + shift diagnostics (R2.6, R4.5, R2.9)
==============================================================================================
The submitted external result (MAE 0.0731) used checkpoints/real64mt_finetuned.pt,
an adapter trained on HFC of a fixed 15-subject split with 6 augmentations —
NOT the headline cross-session recipe. This script:

 1. Trains the headline LN+head adapter (loocv_cross_session.train_adapter,
    identical hyper-parameters) on the HFC scans of ALL 23 ds006557 subjects,
    for 5 seeds (42, 1, 7, 123, 2024 — the multiseed_loocv seeds); checkpoints saved.
 2. Predicts the 10 van den Broek 64 mT T2w scans with: headline adapters
    (per seed + seed-mean), unadapted Stage-2 model, the old 15-subject adapter
    (reproduces the submitted 0.0731), and the frozen Swin-UNETR feature readout
    (E2) fitted on all 23 HFC scans.
 3. Each in the native orientation (as submitted) AND reoriented to RAS
    (ds006557 is RAS; the Zenodo 64 mT files are LAS and were never reoriented).
 4. Metrics: MAE, bias, r, calibration slope; constant baselines (ds006557
    training mean = the honest external constant; Zenodo in-sample mean = oracle);
    offset-removed MAE (oracle diagnostic: subtracts the test-set mean error).
 5. External conformal coverage/width using each model's ds006557 LOOCV residual
    quantiles (same rule as scripts/conformal_calibration.py, m = 23).
 6. Shift diagnostics: label/age distributions; acquisition headers (matrix,
    voxel size, orientation, TR/TE, series, software) for both cohorts;
    simulated-domain check (simulate each cohort's 3T T2w to 64 mT with the
    paper's simulator and predict).

Output: experiments/r3_e3_external/results.json  (+ checkpoints/r3_adapter_all23_seed*.pt)
"""

import json
import math
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import stats

from common import (DS_DIR, EXP, ROOT, lcs, provenance, save_json, scan_path,
                    subjects_and_gt)
from e2_nonneural_baselines import nested_alpha, ridge_fit_predict, swin_model

OUT = EXP / "r3_e3_external" / "results.json"
ZEN = ROOT / "Website_dat" / "extracted" / \
    "Paired 64mT and 3T Brain MRI Scans of Healthy Subjects for Neuroimaging Research v2" / "Data"
SEEDS = [42, 1, 7, 123, 2024]
ALPHAS_CONF = (0.10, 0.05)


def zen_subjects():
    import csv
    gt = {}
    with open(EXP / "fastsurfer_zenodo" / "nwbv_ground_truth_zenodo.csv") as f:
        for r in csv.DictReader(f):
            gt[r["subject"]] = float(r["nWBV_freesurfer"])
    subs = []
    for s in sorted(gt):
        p = zen64_path(s)
        if p is not None:
            subs.append(s)
    return subs, gt


def zen64_path(s):
    anat = ZEN / "64mT data" / s / "ses-01" / "anat"
    for f in sorted(anat.glob(f"{s}_ses-01_run-*_T2w.nii.gz")):
        if "localizer" not in f.name:
            return f
    return None


def load(path, ras=False):
    img = nib.load(str(path))
    if ras:
        img = nib.as_closest_canonical(img)
    vol = img.get_fdata(dtype=np.float32)
    vol = vol[..., 0] if vol.ndim == 4 else vol
    lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
    if hi > lo:
        vol = (np.clip(vol, lo, hi) - lo) / (hi - lo)
    return lcs.resize64(vol.astype(np.float32))


@torch.no_grad()
def vit_pred(model, vol):
    return float(model(torch.tensor(vol)[None, None].float()))


@torch.no_grad()
def swin_feat(m, vol):
    return m.gap(m.backbone(torch.tensor(vol)[None, None].float())).flatten().numpy()


def conformal_q(residuals, alpha):
    m = len(residuals)
    s = sorted(residuals)
    k = math.ceil((m + 1) * (1 - alpha))
    return s[min(k, m) - 1]


def evaluate(pred, y, train_mean, q):
    pred, y = np.asarray(pred), np.asarray(y)
    err = pred - y
    out = {"mae": round(float(np.mean(np.abs(err))), 4), "bias": round(float(err.mean()), 4),
           "offset_removed_mae_oracle": round(float(np.mean(np.abs(err - err.mean()))), 4),
           "pred_mean": round(float(pred.mean()), 4), "pred_sd": round(float(pred.std(ddof=1)), 4)}
    if pred.std() > 0:
        r, p = stats.pearsonr(pred, y)
        lr = stats.linregress(pred, y)
        out.update({"pearson_r": round(float(r), 4), "pearson_p": round(float(p), 4),
                    "calibration_slope": round(float(lr.slope), 3), "calibration_slope_se": round(float(lr.stderr), 3)})
    out["conformal"] = {}
    for a in ALPHAS_CONF:
        cov = np.abs(err) <= q[a]
        out["conformal"][f"level_{int(round((1 - a) * 100))}"] = {
            "halfwidth_from_ds006557": round(float(q[a]), 4), "width": round(float(2 * q[a]), 4),
            "n_covered": int(cov.sum()), "n": len(y), "coverage": round(float(cov.mean()), 3),
            "width_over_cohort_range": round(float(2 * q[a] / (y.max() - y.min())), 2)}
    return out


def header(path):
    img = nib.load(str(path))
    js = Path(str(path).replace(".nii.gz", ".json"))
    side = json.loads(js.read_text()) if js.exists() else {}
    keep = ["RepetitionTime", "EchoTime", "SeriesDescription", "ProtocolName", "SoftwareVersions",
            "MagneticFieldStrength", "AcquisitionDuration", "Manufacturer", "ManufacturersModelName",
            "SliceThickness", "PixelBandwidth", "FlipAngle", "NumberOfAverages"]
    return {"shape": list(img.shape[:3]), "voxel_mm": [round(float(z), 3) for z in img.header.get_zooms()[:3]],
            "orientation": "".join(nib.aff2axcodes(img.affine)), **{k: side.get(k) for k in keep}}


def main():
    subjects, gt = subjects_and_gt()
    y_ds = np.array([gt[s] for s in subjects])
    zsubs, zgt = zen_subjects()
    y_z = np.array([zgt[s] for s in zsubs])
    print(f"ds006557 n={len(subjects)}; external n={len(zsubs)}: {zsubs}")
    res = {"experiment": "r3_e3_external_headline_adapter", "answers": ["R2.6", "R4.5", "R2.9"],
           "external_subjects": zsubs,
           "excluded": {s: "no 64 mT T2w scan in the release" for s in zgt if s not in zsubs},
           "provenance": provenance("scripts/round3/e3_external_headline_adapter.py")}

    # 1. headline adapters on all 23 HFC
    base = lcs.build_base_model()
    adapters = {}
    for sd in SEEDS:
        ck = ROOT / "checkpoints" / f"r3_adapter_all23_seed{sd}.pt"
        m = lcs.build_base_model()
        if ck.exists():
            m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model_state_dict"])
        else:
            random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
            print(f"Training all-23 adapter, seed {sd}")
            m = lcs.train_adapter(base, subjects, gt, fold_idx=-1)
            torch.save({"model_state_dict": m.state_dict(), "recipe": "loocv_cross_session.train_adapter on HFC of all 23",
                        "seed": sd}, ck)
        m.eval()
        adapters[sd] = m
    old = lcs.build_base_model()
    old.load_state_dict(torch.load(ROOT / "checkpoints" / "real64mt_finetuned.pt", map_location="cpu",
                                   weights_only=False)["model_state_dict"])
    old.eval()

    # Swin readout on all 23 HFC features (E2 cache)
    F = dict(np.load(EXP / "r3_e2_nonneural_baselines" / "features.npz"))
    swin = swin_model()
    a_sw = nested_alpha(F["HFC_swin"], y_ds)

    # ds006557 LOOCV residuals for conformal
    head_res = [abs(r["pred_hfe"] - r["true_nwbv"]) for r in
                json.loads((EXP / "loocv_cross_session" / "results.json").read_text())["per_subject"]]
    swin_res = [abs(r["swin_ridge_hfe"] - r["true_nwbv"]) for r in
                json.loads((EXP / "r3_e2b_swin_verification" / "results.json").read_text())["per_subject"]]
    q_head = {a: conformal_q(head_res, a) for a in ALPHAS_CONF}
    q_swin = {a: conformal_q(swin_res, a) for a in ALPHAS_CONF}

    train_mean = float(y_ds.mean())
    res["constant_baselines"] = {
        "ds006557_training_mean": {"value": round(train_mean, 4), "mae": round(float(np.mean(np.abs(train_mean - y_z))), 4)},
        "external_in_sample_mean_oracle": {"value": round(float(y_z.mean()), 4), "mae": round(float(np.mean(np.abs(y_z.mean() - y_z))), 4)},
        "external_loo_mean": {"mae": round(float(np.mean([abs(np.delete(y_z, i).mean() - y_z[i]) for i in range(len(y_z))])), 4)}}

    per_subject = {s: {"true_nwbv": zgt[s]} for s in zsubs}
    res["models"] = {}
    for orient in ("native_as_submitted", "reoriented_RAS"):
        ras = orient == "reoriented_RAS"
        vols = [load(zen64_path(s), ras=ras) for s in zsubs]
        preds = {}
        for sd, m in adapters.items():
            preds[f"headline_adapter_seed{sd}"] = [vit_pred(m, v) for v in vols]
        preds["headline_adapter_seedmean"] = list(np.mean([preds[f"headline_adapter_seed{sd}"] for sd in SEEDS], axis=0))
        preds["unadapted_stage2"] = [vit_pred(base, v) for v in vols]
        preds["old_15subject_adapter_as_submitted"] = [vit_pred(old, v) for v in vols]
        Xz = np.array([swin_feat(swin, v) for v in vols])
        preds["swin_ridge_readout"] = list(ridge_fit_predict(F["HFC_swin"], y_ds, Xz, a_sw))
        res["models"][orient] = {}
        for k, p in preds.items():
            q = q_swin if k.startswith("swin") else q_head
            res["models"][orient][k] = evaluate(p, y_z, train_mean, q)
            for s, v in zip(zsubs, p):
                per_subject[s][f"{orient}:{k}"] = round(float(v), 4)
        print(f"\n[{orient}]")
        for k, v in res["models"][orient].items():
            print(f"  {k:36s} MAE {v['mae']:.4f} bias {v['bias']:+.4f} r {v.get('pearson_r', float('nan')):+.3f} "
                  f"offset-removed {v['offset_removed_mae_oracle']:.4f}  cov95 {v['conformal']['level_95']['n_covered']}/10")
    res["per_subject"] = [{"subject": s, **v} for s, v in per_subject.items()]

    # sanity: reproduce the submitted 0.0731
    res["reproduces_submitted_0.0731"] = res["models"]["native_as_submitted"]["old_15subject_adapter_as_submitted"]["mae"]

    # 6a. distributions
    import csv
    with open(DS_DIR / "participants.tsv") as f:
        ages_ds = [float(r["age"]) for r in csv.DictReader(f, delimiter="\t")]
    res["distribution_shift"] = {
        "nwbv_ds006557": {"mean": round(float(y_ds.mean()), 4), "sd": round(float(y_ds.std(ddof=1)), 4),
                          "min": round(float(y_ds.min()), 4), "max": round(float(y_ds.max()), 4)},
        "nwbv_external": {"mean": round(float(y_z.mean()), 4), "sd": round(float(y_z.std(ddof=1)), 4),
                          "min": round(float(y_z.min()), 4), "max": round(float(y_z.max()), 4)},
        "n_external_within_ds006557_range": int(np.sum((y_z >= y_ds.min()) & (y_z <= y_ds.max()))),
        "age_ds006557": {"mean": round(float(np.mean(ages_ds)), 1), "min": round(min(ages_ds), 1), "max": round(max(ages_ds), 1)},
        "age_external": "cohort-level only in the release (mean 30, range 19-65); no per-subject ages"}

    # 6b. headers
    res["acquisition_headers"] = {
        "ds006557_hfc_example": header(scan_path(subjects[0], "HFC")),
        "ds006557_hfe_example": header(scan_path(subjects[0], "HFE")),
        "external_64mt_example": header(zen64_path(zsubs[0])),
        "external_64mt_all_orientations": sorted({"".join(nib.aff2axcodes(nib.load(str(zen64_path(s))).affine)) for s in zsubs}),
        "ds006557_all_orientations": sorted({"".join(nib.aff2axcodes(nib.load(str(scan_path(s, ses))).affine))
                                             for s in subjects for ses in ("HFC", "HFE")})}

    # 6c. simulated-domain check
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from utils.field_conversion import FieldConverter
        conv = FieldConverter({})

        def sim_pred(path, model):
            img = nib.as_closest_canonical(nib.load(str(path)))
            v = img.get_fdata(dtype=np.float32)
            lo, hi = np.percentile(v, 1), np.percentile(v, 99)
            v = (np.clip(v, lo, hi) - lo) / (hi - lo)
            s = conv.convert(v, method="hyperfine")
            s = np.asarray(s, dtype=np.float32)
            lo, hi = np.percentile(s, 1), np.percentile(s, 99)
            s = (np.clip(s, lo, hi) - lo) / max(hi - lo, 1e-6)
            return vit_pred(model, lcs.resize64(s))

        np.random.seed(0)
        m42 = adapters[42]
        z_sim = [sim_pred(ZEN / "3T data" / s / "anat" / f"{s}_acq-highres_T2w.nii.gz", m42) for s in zsubs]
        d_sim = [sim_pred(DS_DIR / s / "ses-GE" / "anat" / f"{s}_ses-GE_T2w.nii.gz", m42) for s in subjects]
        d_real = [vit_pred(m42, lcs.load_scan(s, "HFC")) for s in subjects]
        z_real = res["models"]["reoriented_RAS"]["headline_adapter_seed42"]
        res["simulated_domain_check"] = {
            "model": "headline adapter seed 42 (all 23 HFC)",
            "external_sim_from_3T_T2w": evaluate(z_sim, y_z, train_mean, q_head),
            "external_real_64mt_RAS": {"mae": z_real["mae"], "bias": z_real["bias"]},
            "ds006557_sim_from_GE_T2w": evaluate(d_sim, y_ds, train_mean, q_head),
            "ds006557_real_hfc_in_sample": {"mae": round(float(np.mean(np.abs(np.array(d_real) - y_ds))), 4),
                                            "note": "in-sample (adapter trained on these scans)"},
            "per_subject_external_sim": dict(zip(zsubs, [round(v, 4) for v in z_sim]))}
        print("\nsim check:", {k: (v.get("mae"), v.get("bias")) for k, v in res["simulated_domain_check"].items() if isinstance(v, dict) and "mae" in v})
    except Exception as e:
        res["simulated_domain_check"] = {"error": repr(e)}
        print("sim check failed:", e)

    save_json(OUT, res)


if __name__ == "__main__":
    main()
