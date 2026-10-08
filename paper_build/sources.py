"""
Every input the paper is built from. Each loader reads one committed result
file under experiments/ and returns plain numpy arrays aligned by subject.
Nothing here computes a statistic; it only reads.
"""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"

USED = set()  # every file read during the build, for the provenance report


def load(rel):
    p = EXP / rel
    USED.add(str(p.relative_to(ROOT)))
    return json.loads(p.read_text())


def _col(rows, key, subjects, skey="subject"):
    d = {r[skey]: r[key] for r in rows}
    return np.array([d[s] for s in subjects], float)


# ── ds006557 (real 64 mT, n = 23) ─────────────────────────────────────────────

def ds006557():
    """Truth, ages and the HFE prediction of every readout evaluated cross-session (HFC of 22 -> HFE of held-out)."""
    head = load("loocv_cross_session/results.json")["per_subject"]
    subjects = [r["subject"] for r in head]
    y = _col(head, "true_nwbv", subjects)
    preds = {"vit_ln_head_headline": _col(head, "pred_hfe", subjects)}

    e2 = load("r3_e2_nonneural_baselines/results.json")["per_subject"]
    for k in ("vit_frozen_offset", "vit_frozen_affine", "ridge_vit_cls_prenorm", "ridge_vit_cls_postnorm",
              "ridge_vit_meantokens", "ridge_image_stats", "ridge_swin_gap"):
        preds[k] = _col(e2, k, subjects)

    e2c = load("r3_e2c_cnn3d_target/results.json")["per_subject"]
    for k in ("cnn_unadapted_hfe", "cnn_frozen_offset", "cnn_frozen_affine",
              "ridge_cnn_globalpool256", "ridge_cnn_penultimate128"):
        preds[k] = _col(e2c, k, subjects)

    e2b = load("r3_e2b_swin_verification/results.json")["per_subject"]
    preds["age_only_ridge"] = _col(e2b, "age_only_ridge", subjects)
    preds["swin_plus_age_ridge"] = _col(e2b, "swin_plus_age_ridge", subjects)
    age = _col(e2b, "age", subjects)

    fr = load("r3_frozen_predictions/predictions.json")["per_subject"]
    preds["vit_unadapted"] = _col(fr, "pred_hfe", subjects)

    ab = load("ablation_adapter/results.json")["results"]
    for k in ("head_only", "ln_head", "full_ft"):
        preds[f"adapter_{k}"] = _col(ab[k]["per_subject"], "pred", subjects)
    preds["adapter_lora"] = _col(load("ablation_lora/results.json")["lora"]["per_subject"], "pred", subjects)
    preds["dino_ln_head"] = _col(load("dino_headline_loocv/results.json")["per_subject"], "pred_hfe", subjects)

    for k in preds:
        assert len(preds[k]) == len(y), k
    for rows in (e2, e2c, e2b, fr):
        assert np.allclose(_col(rows, "true_nwbv", subjects), y), "truth mismatch across result files"
    return subjects, y, age, preds


def sessions():
    """HFC and HFE predictions for the reproducibility analysis."""
    head = load("loocv_cross_session/results.json")["per_subject"]
    subjects = [r["subject"] for r in head]
    out = {"headline_same_fold": (_col(head, "pred_hfc", subjects), _col(head, "pred_hfe", subjects))}
    fr = load("r3_frozen_predictions/predictions.json")["per_subject"]
    out["frozen_single_model"] = (_col(fr, "pred_hfc", subjects), _col(fr, "pred_hfe", subjects))
    seeds = {}
    for s in load("multiseed_loocv/results.json")["per_seed"]:
        seeds[s["seed"]] = (_col(s["per_subject"], "pred_hfc", subjects), _col(s["per_subject"], "pred_hfe", subjects))
    qc = load("r3_e1_reproducibility/results.json")["d_image_qc"]["per_scan"]
    z_hfe = np.array([qc[s]["HFE"]["com_offset_z_mm"] for s in subjects])
    z_hfc = np.array([qc[s]["HFC"]["com_offset_z_mm"] for s in subjects])
    return subjects, out, seeds, z_hfc, z_hfe


def e1():
    return load("r3_e1_reproducibility/results.json")


def e2b():
    return load("r3_e2b_swin_verification/results.json")


# ── external cohort (van den Broek, n = 10) ───────────────────────────────────

def external():
    e3 = load("r3_e3_external/results.json")
    rows = e3["per_subject"]
    subjects = [r["subject"] for r in rows]
    y = np.array([r["true_nwbv"] for r in rows])
    keys = {"vit_ln_head_seedmean": "native_as_submitted:headline_adapter_seedmean",
            "vit_unadapted": "native_as_submitted:unadapted_stage2",
            "ridge_swin_gap": "native_as_submitted:swin_ridge_readout",
            "vit_ln_head_old15_as_submitted": "native_as_submitted:old_15subject_adapter_as_submitted"}
    for s in (42, 1, 7, 123, 2024):
        keys[f"vit_ln_head_seed{s}"] = f"native_as_submitted:headline_adapter_seed{s}"
    preds = {k: np.array([r[v] for r in rows]) for k, v in keys.items()}
    ras = {k: np.array([r[v.replace("native_as_submitted", "reoriented_RAS")] for r in rows]) for k, v in keys.items()}
    return subjects, y, preds, ras, e3


# ── OASIS-1 ───────────────────────────────────────────────────────────────────

def oasis_fixed():
    return load("r3_e8_oasis_fixed_inputs/results.json")


def oasis_single_draw():
    """Comparators whose fine-tuned weights were not saved: single on-the-fly noise draw, as reported."""
    out = {}
    for name, rel in (("unetr", "arch_comparator_unetr/results.json"), ("ssl_mae", "ssl_comparator_mae/results.json"),
                      ("ssl_simmim", "ssl_comparator_simmim/results.json"),
                      ("ssl_contrastive", "ssl_comparator_contrastive/results.json"),
                      ("swin_unetr", "arch_comparator_swin/results.json"), ("ssl_dino", "ssl_comparator_dino/results.json")):
        d = load(rel)
        t = np.array([r["true"] for r in d["per_test"]]); p = np.array([r["pred"] for r in d["per_test"]])
        out[name] = {"true": t, "pred": p}
    return out


def params():
    return load("model_param_counts/results.json")["counts"]


# ── other experiments ─────────────────────────────────────────────────────────

def e4():
    d = load("r3_e4_physics_vs_blur/summary.json")
    arms = {}
    for seed in (42, 1, 7):
        for arm in ("physics", "blur"):
            arms[(arm, seed)] = load(f"r3_e4_physics_vs_blur/{arm}_seed{seed}.json")
    return d, arms


def sensitivity():
    return load("simulation_sensitivity/results.json")


def multiseed():
    return load("multiseed_loocv/results.json")


def e5_and_r2():
    return load("r3_existing_data_analyses/results.json")


def e6():
    return load("r3_e6_time_budget/results.json")


def e7():
    return load("r3_e7_conformal/results.json")


def mc_dropout():
    return {"oasis": load("oasis_mc_dropout/results.json"), "real64mt": load("real64mt_eval/mc_dropout_ci.json")}


def latency():
    return load("inference_latency/results.json")


def label_table():
    """Per-cohort label definitions, checked against the committed label files."""
    rows = []
    gt = EXP / "fastsurfer_output" / "nwbv_ground_truth.csv"
    USED.add(str(gt.relative_to(ROOT)))
    with open(gt) as f:
        r = list(csv.DictReader(f))
    ok = all(abs((float(x["gray_matter_vol"]) + float(x["white_matter_vol"])) / float(x["etiv"])
                 - float(x["nWBV_freesurfer"])) < 6e-5 for x in r)
    gz = EXP / "fastsurfer_zenodo" / "nwbv_ground_truth_zenodo.csv"
    USED.add(str(gz.relative_to(ROOT)))
    with open(gz) as f:
        z = list(csv.DictReader(f))
    okz = all(abs(float(x["brainseg_vol"]) / float(x["etiv_mask"]) - float(x["nWBV_freesurfer"])) < 6e-5 for x in z)
    assert ok and okz, "label files do not match their stated definition"
    rows.append({"cohort": "IXI (Stage 1)", "target": "4 proxy targets (BTF, TCR, VBR, MCI)",
                 "source": "FreeSurfer label maps shipped with the preprocessed IXI release (Chen et al.)", "acquired_on": "1.5 T / 3 T T1w"})
    rows.append({"cohort": "OASIS-1 (Stage 2)", "target": "nWBV",
                 "source": "OASIS-supplied spreadsheet: FSL FAST segmentation of the atlas-registered, masked image (Fotenos et al. 2005)",
                 "acquired_on": "1.5 T T1w"})
    rows.append({"cohort": "ds006557 (Stage 3)", "target": "nWBV = BrainSegVol / MaskVol",
                 "source": "FastSurfer --seg_only on paired 3 T MPRAGE", "acquired_on": "3 T (labels); 64 mT T2w (inputs)",
                 "verified_rows": len(r)})
    rows.append({"cohort": "van den Broek (external)", "target": "nWBV = BrainSegVol / MaskVol",
                 "source": "FastSurfer --seg_only on paired 3 T", "acquired_on": "3 T (labels); 64 mT T2w (inputs)",
                 "verified_rows": len(z)})
    rows.append({"cohort": "SynthSeg+ comparator", "target": "nWBV = (GM + WM) / TIV",
                 "source": "FreeSurfer 7.4.1 mri_synthseg --robust on 64 mT T2w", "acquired_on": "64 mT T2w"})
    return rows
