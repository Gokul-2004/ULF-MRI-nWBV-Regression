"""
Cross-Session Leave-One-Out Cross-Validation
=============================================
Primary evaluation protocol for the paper.

For each subject i in {0..22}:
  - Train adapter (final LayerNorm + head, 769 params) on HFC scans
    of all OTHER subjects (n=22)
  - Test on HFE scan of subject i

Subject i contributes zero data to the adapter that evaluates them:
  - HFC_i  not in training
  - HFE_i  not in training
  => fully held-out per subject

Secondary output (no extra training):
  - Also predict HFC_i with the same held-out adapter → ICC(3,1)

Usage:
    python3 scripts/loocv_cross_session.py

Output:
    experiments/loocv_cross_session/results.json
"""

import sys
import json
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from scipy.ndimage import zoom
from scipy import stats

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.baselines import BaselineViT3D

# ── Config ────────────────────────────────────────────────────────────────────
DS_DIR       = project_root / "data" / "ds006557_data"
GT_CSV       = project_root / "experiments" / "fastsurfer_output" / "nwbv_ground_truth.csv"
VIT_CKPT     = project_root / "checkpoints" / "oasis_finetuned.pt"
OUT_DIR      = project_root / "experiments" / "loocv_cross_session"
TARGET_SHAPE = (64, 64, 64)
DEVICE       = torch.device("cpu")   # Quadro P1000 CC6.1 not supported

EPOCHS       = 80
LR           = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 4
N_AUGMENTS   = 5       # augmented copies per original scan
N_BOOT       = 10_000  # bootstrap resamples for CI
SEED         = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Data utilities ─────────────────────────────────────────────────────────────

def load_nifti_norm(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    vol = img.get_fdata(dtype=np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
    if hi > lo:
        vol = np.clip(vol, lo, hi)
        vol = (vol - lo) / (hi - lo)
    return vol.astype(np.float32)


def resize64(vol: np.ndarray) -> np.ndarray:
    if vol.shape == TARGET_SHAPE:
        return vol
    factors = [t / s for t, s in zip(TARGET_SHAPE, vol.shape)]
    return zoom(vol, factors, order=1).astype(np.float32)


def augment(vol: np.ndarray) -> np.ndarray:
    v = vol.copy()
    v += np.random.normal(0, 0.015, v.shape).astype(np.float32)
    v = v * np.random.uniform(0.90, 1.10) + np.random.uniform(-0.05, 0.05)
    if np.random.rand() > 0.5:
        v = np.flip(v, axis=np.random.randint(0, 3)).copy()
    return np.clip(v, 0, 1).astype(np.float32)


def load_scan(subject: str, session: str) -> np.ndarray | None:
    path = (DS_DIR / subject / f"ses-{session}" / "anat" /
            f"{subject}_ses-{session}_acq-axi_T2w.nii.gz")
    if not path.exists():
        return None
    return resize64(load_nifti_norm(path))


def load_ground_truth() -> dict[str, float]:
    import csv
    gt = {}
    with open(GT_CSV) as f:
        for row in csv.DictReader(f):
            gt[row["subject"]] = float(row["nWBV_freesurfer"])
    return gt


def load_subjects() -> tuple[list[str], dict[str, float]]:
    gt = load_ground_truth()
    subjects = sorted(s for s in gt if (DS_DIR / s).exists())
    return subjects, gt


# ── Model utilities ────────────────────────────────────────────────────────────

def build_base_model() -> BaselineViT3D:
    model = BaselineViT3D(
        img_size=TARGET_SHAPE, patch_size=16, num_classes=4,
        embed_dim=256, num_layers=4, num_heads=8
    )
    # checkpoint already has head shape [1, 256]
    model.head = nn.Linear(model.head.in_features, 1)
    ckpt = torch.load(VIT_CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def freeze_to_adapter(model: nn.Module) -> int:
    """Freeze all params; unfreeze final LayerNorm + head (769 params)."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.norm.parameters():
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Training ───────────────────────────────────────────────────────────────────

def train_adapter(
    base_model: nn.Module,
    train_subjects: list[str],
    gt: dict[str, float],
    fold_idx: int,
) -> nn.Module:
    model = copy.deepcopy(base_model)
    n_adapter = freeze_to_adapter(model)
    model.to(DEVICE)

    # Build HFC training set with augmentation
    X, y = [], []
    for subj in train_subjects:
        vol = load_scan(subj, "HFC")
        if vol is None:
            print(f"    WARNING: HFC missing for {subj}, skipping")
            continue
        nwbv = gt[subj]
        X.append(vol)
        y.append(nwbv)
        for _ in range(N_AUGMENTS):
            X.append(augment(vol))
            y.append(nwbv)

    if not X:
        raise RuntimeError(f"Fold {fold_idx + 1}: no training scans found")

    X_t = torch.tensor(np.array(X)).unsqueeze(1).float()
    y_t = torch.tensor(y).float().unsqueeze(1)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_loss  = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(X_t))
        X_t, y_t = X_t[perm], y_t[perm]

        epoch_loss, n_batches = 0.0, 0
        for i in range(0, len(X_t), BATCH_SIZE):
            xb = X_t[i : i + BATCH_SIZE].to(DEVICE)
            yb = y_t[i : i + BATCH_SIZE].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        avg = epoch_loss / max(n_batches, 1)
        scheduler.step()

        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"      Epoch {epoch+1:3d}/{EPOCHS}  loss={avg:.6f}")

    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict(model: nn.Module, subject: str, session: str) -> float | None:
    vol = load_scan(subject, session)
    if vol is None:
        return None
    x = torch.tensor(vol).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    return float(model(x).item())


# ── Statistics ─────────────────────────────────────────────────────────────────

def bootstrap_mae_ci(abs_errors: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
    rng   = np.random.default_rng(SEED)
    means = [np.mean(rng.choice(abs_errors, size=len(abs_errors), replace=True))
             for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compute_icc31(session_a: np.ndarray, session_b: np.ndarray) -> float:
    """
    ICC(3,1): two-way mixed-effects, single-measures, absolute agreement.
    session_a, session_b: paired prediction arrays of length n.
    """
    n    = len(session_a)
    data = np.column_stack([session_a, session_b])   # (n, 2)

    grand_mean = data.mean()
    row_means  = data.mean(axis=1)   # per-subject
    col_means  = data.mean(axis=0)   # per-session

    ss_r = 2 * np.sum((row_means - grand_mean) ** 2)   # k=2
    ss_c = n * np.sum((col_means - grand_mean) ** 2)
    ss_t = np.sum((data - grand_mean) ** 2)
    ss_e = ss_t - ss_r - ss_c

    ms_r = ss_r / (n - 1)
    ms_e = ss_e / (n - 1)                              # (n-1)(k-1), k=2

    return float((ms_r - ms_e) / (ms_r + ms_e))


def bootstrap_icc_ci(
    session_a: np.ndarray,
    session_b: np.ndarray,
    n_boot: int = N_BOOT,
) -> tuple[float, float]:
    rng  = np.random.default_rng(SEED)
    n    = len(session_a)
    iccs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        iccs.append(compute_icc31(session_a[idx], session_b[idx]))
    return float(np.percentile(iccs, 2.5)), float(np.percentile(iccs, 97.5))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("Cross-Session LOOCV  (HFC train  →  HFE test)")
    print(f"Device: {DEVICE}  |  Epochs: {EPOCHS}  |  LR: {LR}")
    print("=" * 62)

    subjects, gt = load_subjects()
    n = len(subjects)
    print(f"Subjects loaded: {n}")

    base_model = build_base_model()

    records = []

    for i, held_out in enumerate(subjects):
        train_subjs = [s for s in subjects if s != held_out]
        print(f"\n[Fold {i+1:2d}/{n}]  held-out={held_out}  "
              f"train n={len(train_subjs)}")

        model = train_adapter(base_model, train_subjs, gt, i)

        pred_hfe = predict(model, held_out, "HFE")
        pred_hfc = predict(model, held_out, "HFC")   # for ICC
        true_val = gt[held_out]

        if pred_hfe is None:
            print(f"  SKIP: HFE scan missing for {held_out}")
            continue

        err = pred_hfe - true_val
        print(f"  HFE  pred={pred_hfe:.4f}  GT={true_val:.4f}  "
              f"err={err:+.4f}  |absErr|={abs(err):.4f}")

        records.append({
            "subject":       held_out,
            "true_nwbv":     round(true_val, 4),
            "pred_hfe":      round(pred_hfe, 4),
            "pred_hfc":      round(pred_hfc, 4) if pred_hfc is not None else None,
            "abs_error_hfe": round(abs(err), 4),
        })

    # ── Aggregate ──────────────────────────────────────────────────────────────
    trues     = np.array([r["true_nwbv"] for r in records])
    preds_hfe = np.array([r["pred_hfe"]  for r in records])
    abs_errs  = np.abs(preds_hfe - trues)

    mae  = float(np.mean(abs_errs))
    rmse = float(np.sqrt(np.mean((preds_hfe - trues) ** 2)))
    bias = float(np.mean(preds_hfe - trues))
    ci_lo, ci_hi = bootstrap_mae_ci(abs_errs)

    r_val,  p_r   = stats.pearsonr(preds_hfe, trues)
    rho,    p_rho = stats.spearmanr(preds_hfe, trues)

    # ICC(3,1): only where HFC prediction is also available
    icc_val = icc_ci = None
    hfc_available = [r for r in records if r["pred_hfc"] is not None]
    if len(hfc_available) == n:
        preds_hfc_arr = np.array([r["pred_hfc"] for r in records])
        icc_val = compute_icc31(preds_hfc_arr, preds_hfe)
        icc_lo, icc_hi = bootstrap_icc_ci(preds_hfc_arr, preds_hfe)
        icc_ci = [round(icc_lo, 4), round(icc_hi, 4)]

    n_below = int(np.sum(abs_errs < 0.020))

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"CROSS-SESSION LOOCV RESULTS  (n={len(records)})")
    print("=" * 62)
    print(f"MAE           = {mae:.4f}  [95% CI: {ci_lo:.4f}–{ci_hi:.4f}]")
    print(f"RMSE          = {rmse:.4f}")
    print(f"Bias          = {bias:+.4f}")
    print(f"Subjects MAE<0.020: {n_below}/{len(records)}")
    print(f"Pearson r     = {r_val:.4f}  (p={p_r:.4f})"
          "  [narrow range — informational only]")
    print(f"Spearman ρ    = {rho:.4f}  (p={p_rho:.4f})")
    if icc_val is not None:
        print(f"ICC(3,1) HFC vs HFE = {icc_val:.4f}  [95% CI: {icc_ci}]")

    # ── Save ───────────────────────────────────────────────────────────────────
    results = {
        "protocol":              "cross_session_loocv",
        "description":           "Adapter trained on HFC of n-1 subjects; tested on HFE of held-out subject",
        "n_subjects":            len(records),
        "n_train_per_fold":      n - 1,
        "adapter_params":        769,
        "adapter_components":    "final_layernorm + regression_head",
        "epochs_per_fold":       EPOCHS,
        "mae":                   round(mae, 4),
        "rmse":                  round(rmse, 4),
        "bias":                  round(bias, 4),
        "mae_bootstrap_ci_95":   [round(ci_lo, 4), round(ci_hi, 4)],
        "n_bootstrap":           N_BOOT,
        "pearson_r":             round(float(r_val), 4),
        "pearson_p":             round(float(p_r), 4),
        "spearman_rho":          round(float(rho), 4),
        "spearman_p":            round(float(p_rho), 4),
        "icc_31_hfc_hfe":        round(float(icc_val), 4) if icc_val is not None else None,
        "icc_31_bootstrap_ci_95": icc_ci,
        "subjects_below_mae020": n_below,
        "per_subject":           records,
    }

    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
