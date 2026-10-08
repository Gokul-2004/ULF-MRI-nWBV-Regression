"""
Shared helpers for the round-3 (Access-2026-45125 resubmission) analyses.

Preprocessing and model construction are imported from
scripts/loocv_cross_session.py so every round-3 number uses exactly the
headline pipeline's NIfTI loading, normalisation, resizing and checkpoint.
"""

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import loocv_cross_session as lcs  # noqa: E402  (seeds RNGs at import)

EXP = ROOT / "experiments"
DS_DIR = lcs.DS_DIR
SEED = 42


# ── provenance ────────────────────────────────────────────────────────────────

def provenance(script: str) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--", "scripts", "models", "utils"],
            text=True).strip())
    except Exception:
        commit, dirty = "unknown", None
    return {
        "script": script,
        "git_commit": commit,
        "code_tree_dirty": dirty,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    print(f"Saved -> {path.relative_to(ROOT)}")


# ── data ──────────────────────────────────────────────────────────────────────

def subjects_and_gt():
    return lcs.load_subjects()


def scan_path(subject: str, session: str) -> Path:
    return DS_DIR / subject / f"ses-{session}" / "anat" / f"{subject}_ses-{session}_acq-axi_T2w.nii.gz"


def read_csv_col(path: Path, key: str, col: str) -> dict:
    with open(path) as f:
        return {r[key]: float(r[col]) for r in csv.DictReader(f)}


def maskvol() -> dict:
    """FastSurfer MaskVol (the CSV column is named 'etiv' but holds MaskVol)."""
    return read_csv_col(EXP / "fastsurfer_output" / "nwbv_ground_truth.csv", "subject", "etiv")


def synthseg_tiv() -> dict:
    return read_csv_col(EXP / "synthseg_output" / "nwbv_synthseg.csv", "subject", "tiv")


# ── statistics ────────────────────────────────────────────────────────────────

def _anova2(a: np.ndarray, b: np.ndarray):
    data = np.column_stack([a, b])
    n, k = data.shape
    gm = data.mean()
    ss_r = k * np.sum((data.mean(axis=1) - gm) ** 2)
    ss_c = n * np.sum((data.mean(axis=0) - gm) ** 2)
    ss_t = np.sum((data - gm) ** 2)
    ss_e = ss_t - ss_r - ss_c
    ms_r = ss_r / (n - 1)
    ms_c = ss_c / (k - 1)
    ms_e = ss_e / ((n - 1) * (k - 1))
    return n, k, ms_r, ms_c, ms_e


def icc_c1(a, b) -> float:
    """ICC(3,1) consistency (McGraw & Wong ICC(C,1)) — what the paper's code computes."""
    n, k, ms_r, ms_c, ms_e = _anova2(np.asarray(a, float), np.asarray(b, float))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e))


def icc_a1(a, b) -> float:
    """ICC(A,1) absolute agreement (McGraw & Wong)."""
    n, k, ms_r, ms_c, ms_e = _anova2(np.asarray(a, float), np.asarray(b, float))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n))


def boot_ci(fn, *arrays, n_boot=10_000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = fn(*[np.asarray(x)[idx] for x in arrays])
        if np.isfinite(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def perm_p_pairing(fn, a, b, n_perm=20_000, seed=SEED):
    """One-sided permutation p for fn(a, b) under random re-pairing of b."""
    rng = np.random.default_rng(seed)
    obs = fn(a, b)
    b = np.asarray(b)
    hits = sum(fn(a, b[rng.permutation(len(b))]) >= obs for _ in range(n_perm))
    return float((hits + 1) / (n_perm + 1))


def residualise(y, X):
    """OLS residuals of y on [1, X] (X may be 1-D or 2-D)."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta
