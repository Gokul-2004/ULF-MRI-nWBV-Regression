"""
Build-time guards.

lint()        refuses to build if paper_build code draws random numbers outside
              stats.py's seeded bootstrap, or contains a number with four or more
              decimals (the precision results are reported at), i.e. a typed-in result.
crosscheck()  recomputes key results independently in paper_build/stats.py and
              asserts they match what each experiment script saved itself.
committed()   lists every input file and whether git tracks it.
"""

import re
import subprocess
from pathlib import Path

import numpy as np

from . import sources as S
from .stats import icc_c1, jackknife_conformal

HERE = Path(__file__).resolve().parent


def lint():
    problems = []
    for f in sorted(HERE.glob("*.py")):
        if f.name == "checks.py":
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            code = line.split("#")[0]
            if re.search(r"\bnp\.random\.(?!default_rng)", code) or re.search(r"\brandom\.(seed|random|choice|shuffle)\b", code):
                problems.append(f"{f.name}:{i}: random number generation outside the seeded bootstrap")
            if f.name != "stats.py" and "default_rng" in code:
                problems.append(f"{f.name}:{i}: RNG used outside stats.py")
            if re.search(r"(?<![\w.])\d+\.\d{4,}\b", code):
                problems.append(f"{f.name}:{i}: literal with >= 4 decimals (typed-in result?): {line.strip()[:80]}")
    return problems


def _close(name, a, b, tol, out):
    ok = abs(a - b) <= tol
    out.append({"check": name, "paper_build": round(float(a), 5), "experiment_file": round(float(b), 5), "ok": bool(ok)})


def crosscheck(numbers):
    out = []
    v = lambda k: numbers[k]["value"]  # noqa: E731
    e2 = S.load("r3_e2_nonneural_baselines/results.json")
    for k in ("ridge_swin_gap", "ridge_image_stats", "ridge_vit_meantokens", "vit_frozen_offset", "constant_loo_mean"):
        _close(f"MAE {k} vs E2", v(f"ds006557.{k}.mae"), e2["arms"][k]["mae"], 6e-5, out)
        _close(f"MSE skill {k} vs E2", v(f"ds006557.{k}.mse_skill"), e2["arms"][k]["mse_skill_vs_loo_mean"], 6e-4, out)
    _close("MAE headline vs E2 reference", v("ds006557.vit_ln_head_headline.mae"), e2["headline_ln_head_reference"]["mae"], 6e-5, out)
    e2c = S.load("r3_e2c_cnn3d_target/results.json")["cross_session_arms"]
    for k in ("ridge_cnn_globalpool256", "cnn_frozen_affine"):
        _close(f"MAE {k} vs E2c", v(f"ds006557.{k}.mae"), e2c[k]["mae"], 6e-5, out)
    nb = S.load("r3_e2b_swin_verification/results.json")["nuisance_only_readouts"]
    _close("MAE age-only vs E2b", v("ds006557.age_only_ridge.mae"), nb["age_only"]["mae"], 6e-5, out)
    _close("MAE Swin+age vs E2b", v("ds006557.swin_plus_age_ridge.mae"), nb["swin_plus_age_vs_age"]["swin_plus_age"]["mae"], 6e-5, out)
    e1 = S.load("r3_e1_reproducibility/results.json")
    _close("ICC frozen vs E1", v("repro.icc_frozen_single_model"), e1["a_frozen_model"]["icc_c1"], 6e-5, out)
    _close("ICC headline vs E1", v("repro.icc_headline_same_fold"), e1["reference_headline_same_fold_model"]["icc_c1"], 6e-5, out)
    _close("ICC cross-seed mean vs E1", v("repro.icc_cross_seed_mean"), e1["b_cross_seed"]["cross_seed_mean_icc_c1"], 6e-4, out)
    e7 = S.load("r3_e7_conformal/results.json")["cohorts"]
    for lv in ("90", "95"):
        c = v(f"conformal.ds006557.vit_ln_head_headline.{lv}")
        _close(f"conformal coverage {lv} vs E7", c["coverage"], e7["ds006557_headline_adapter"][f"level_{lv}"]["coverage"], 6e-4, out)
        _close(f"conformal width {lv} vs E7", c["width"], e7["ds006557_headline_adapter"][f"level_{lv}"]["mean_width"], 6e-5, out)
        ce = v(f"conformal.external.vit_ln_head_seedmean.{lv}")
        _close(f"external conformal width {lv} vs E7", ce["width"],
               e7["external_native_as_submitted_headline_adapter_seedmean"][f"level_{lv}"]["mean_width"], 6e-5, out)
    e3 = S.load("r3_e3_external/results.json")["models"]["native_as_submitted"]
    _close("external adapter MAE vs E3", v("external.vit_ln_head_seedmean.mae"), e3["headline_adapter_seedmean"]["mae"], 6e-5, out)
    _close("external Swin MAE vs E3", v("external.ridge_swin_gap.mae"), e3["swin_ridge_readout"]["mae"], 6e-5, out)
    _close("external old adapter = submitted 0.0731", v("external.vit_ln_head_old15_as_submitted.mae"),
           S.load("r3_e3_external/results.json")["reproduces_submitted_0.0731"], 6e-5, out)
    an = S.load("r3_existing_data_analyses/results.json")["r2_2"]["headline"]
    _close("MSE skill headline vs r2_2", v("ds006557.vit_ln_head_headline.mse_skill"), an["mse_skill_vs_loo_mean"], 6e-4, out)
    _close("calibration slope headline vs r2_2", v("ds006557.vit_ln_head_headline.calibration_slope"), an["calibration_slope_true_on_pred"], 6e-3, out)
    lc = S.load("loocv_cross_session/results.json")
    _close("headline MAE vs submitted LOOCV file", v("ds006557.vit_ln_head_headline.mae"), lc["mae"], 6e-5, out)
    # the construction check: internal jackknife coverage depends only on rank
    _, y, _, preds = S.ds006557()
    covs = {round(jackknife_conformal(p, y, 0.05)["coverage"], 4) for p in preds.values()}
    out.append({"check": "internal 95% conformal coverage identical for every readout (fixed by construction)",
                "paper_build": sorted(covs), "experiment_file": "—", "ok": len(covs) == 1})
    a, b = S.sessions()[1]["frozen_single_model"]
    _close("ICC frozen recomputed twice", icc_c1(a, b), icc_c1(b, a), 1e-12, out)
    return out


def committed():
    root = S.ROOT
    try:
        tracked = set(subprocess.check_output(["git", "-C", str(root), "ls-files"], text=True).splitlines())
    except Exception:
        tracked = set()
    return {f: (f in tracked) for f in sorted(S.USED)}


def git_state():
    root = S.ROOT
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--", "paper_build", "experiments",
                                         "scripts"], text=True).strip()
        tag = subprocess.run(["git", "-C", str(root), "describe", "--tags", "--exact-match"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit, dirty, tag = "unknown", "unknown", ""
    return {"commit": commit, "tag": tag or None, "uncommitted_changes": bool(dirty)}


np.seterr(all="ignore")
