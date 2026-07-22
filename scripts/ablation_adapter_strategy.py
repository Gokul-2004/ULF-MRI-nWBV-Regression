"""
Adapter Strategy Ablation (E5)
================================
Compares three domain-adaptation strategies using cross-session LOOCV:

  (A) Head-only       — only regression head (~257 params)
  (B) LN + head       — final LayerNorm + head (769 params, current)
  (C) Full fine-tune  — all 4.23M parameters

For each strategy, runs the full 23-fold cross-session LOOCV
(train on HFC of 22 subjects, test on HFE of held-out subject).

Reports per-fold MAE and fold-to-fold variance to show stability.

Output: experiments/ablation_adapter/results.json
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

# ── Config ─────────────────────────────────────────────────────────────────────
DS_DIR       = project_root / "data" / "ds006557_data"
GT_CSV       = project_root / "experiments" / "fastsurfer_output" / "nwbv_ground_truth.csv"
VIT_CKPT     = project_root / "checkpoints" / "oasis_finetuned.pt"
OUT_DIR      = project_root / "experiments" / "ablation_adapter"
TARGET_SHAPE = (64, 64, 64)
DEVICE       = torch.device("cpu")

EPOCHS       = 80
LR           = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 4
N_AUGMENTS   = 5
N_BOOT       = 10_000
SEED         = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Data ───────────────────────────────────────────────────────────────────────

def load_nifti_norm(path):
    img = nib.load(str(path))
    vol = img.get_fdata(dtype=np.float32)
    if vol.ndim == 4: vol = vol[..., 0]
    lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
    if hi > lo:
        vol = np.clip(vol, lo, hi)
        vol = (vol - lo) / (hi - lo)
    return vol.astype(np.float32)


def resize64(vol):
    if vol.shape == TARGET_SHAPE: return vol
    from scipy.ndimage import zoom as zm
    factors = [t / s for t, s in zip(TARGET_SHAPE, vol.shape)]
    return zm(vol, factors, order=1).astype(np.float32)


def augment(vol):
    v = vol.copy()
    v += np.random.normal(0, 0.015, v.shape).astype(np.float32)
    v = v * np.random.uniform(0.90, 1.10) + np.random.uniform(-0.05, 0.05)
    if np.random.rand() > 0.5:
        v = np.flip(v, axis=np.random.randint(0, 3)).copy()
    return np.clip(v, 0, 1).astype(np.float32)


def load_scan(subject, session):
    path = (DS_DIR / subject / f"ses-{session}" / "anat" /
            f"{subject}_ses-{session}_acq-axi_T2w.nii.gz")
    if not path.exists(): return None
    return resize64(load_nifti_norm(path))


def load_gt():
    import csv
    gt = {}
    with open(GT_CSV) as f:
        for row in csv.DictReader(f):
            gt[row["subject"]] = float(row["nWBV_freesurfer"])
    return gt


def load_subjects():
    gt = load_gt()
    return sorted(s for s in gt if (DS_DIR / s).exists()), gt


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model():
    model = BaselineViT3D(img_size=TARGET_SHAPE, patch_size=16, num_classes=4,
                          embed_dim=256, num_layers=4, num_heads=8)
    model.head = nn.Linear(model.head.in_features, 1)
    ckpt = torch.load(VIT_CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def configure_adapter(model, strategy):
    """Freeze/unfreeze params based on strategy. Returns n_trainable."""
    for p in model.parameters():
        p.requires_grad = False

    if strategy == "head_only":
        for p in model.head.parameters():
            p.requires_grad = True

    elif strategy == "ln_head":
        for p in model.norm.parameters():
            p.requires_grad = True
        for p in model.head.parameters():
            p.requires_grad = True

    elif strategy == "full_ft":
        for p in model.parameters():
            p.requires_grad = True

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Training ───────────────────────────────────────────────────────────────────

def train_fold(base_model, train_subjects, gt, strategy, fold_idx):
    model = copy.deepcopy(base_model)
    n_trainable = configure_adapter(model, strategy)
    model.to(DEVICE)

    X, y = [], []
    for subj in train_subjects:
        vol = load_scan(subj, "HFC")
        if vol is None: continue
        nwbv = gt[subj]
        X.append(vol); y.append(nwbv)
        for _ in range(N_AUGMENTS):
            X.append(augment(vol)); y.append(nwbv)

    X_t = torch.tensor(np.array(X)).unsqueeze(1).float()
    y_t = torch.tensor(y).float().unsqueeze(1)

    # Lower LR for full fine-tune to prevent catastrophic forgetting
    lr = LR / 10 if strategy == "full_ft" else LR
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_loss  = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(X_t))
        X_t, y_t = X_t[perm], y_t[perm]
        epoch_loss, nb = 0.0, 0
        for i in range(0, len(X_t), BATCH_SIZE):
            xb = X_t[i:i+BATCH_SIZE].to(DEVICE)
            yb = y_t[i:i+BATCH_SIZE].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item(); nb += 1
        avg = epoch_loss / max(nb, 1)
        scheduler.step()
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, n_trainable


@torch.no_grad()
def predict(model, subject, session):
    vol = load_scan(subject, session)
    if vol is None: return None
    x = torch.tensor(vol).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    return float(model(x).item())


def bootstrap_mae_ci(abs_errors):
    rng = np.random.default_rng(SEED)
    means = [np.mean(rng.choice(abs_errors, size=len(abs_errors), replace=True))
             for _ in range(N_BOOT)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ── Main ───────────────────────────────────────────────────────────────────────

def run_loocv(base_model, subjects, gt, strategy):
    print(f"\n  Strategy: {strategy}")
    records = []
    n_trainable = None

    for i, held_out in enumerate(subjects):
        train_subjs = [s for s in subjects if s != held_out]
        model, n_trainable = train_fold(base_model, train_subjs, gt, strategy, i)
        pred = predict(model, held_out, "HFE")
        true = gt[held_out]
        if pred is None: continue
        err = abs(pred - true)
        records.append({"subject": held_out, "pred": round(pred,4),
                        "true": round(true,4), "abs_error": round(err,4)})
        print(f"    [{i+1:2d}/23] {held_out}  err={err:.4f}")

    abs_errs = np.array([r["abs_error"] for r in records])
    mae      = float(np.mean(abs_errs))
    mae_std  = float(np.std(abs_errs))
    ci_lo, ci_hi = bootstrap_mae_ci(abs_errs)
    bias     = float(np.mean([r["pred"] - r["true"] for r in records]))

    print(f"  → MAE={mae:.4f} ± {mae_std:.4f}  CI=[{ci_lo:.4f}-{ci_hi:.4f}]  "
          f"bias={bias:+.4f}  trainable_params={n_trainable}")

    return {
        "strategy":        strategy,
        "trainable_params": n_trainable,
        "mae":             round(mae, 4),
        "mae_std":         round(mae_std, 4),
        "mae_ci_95":       [round(ci_lo, 4), round(ci_hi, 4)],
        "bias":            round(bias, 4),
        "n_subjects":      len(records),
        "per_subject":     records,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 62)
    print("Adapter Strategy Ablation (cross-session LOOCV)")
    print(f"Strategies: head_only | ln_head | full_ft")
    print("=" * 62)

    subjects, gt = load_subjects()
    base_model   = build_model()

    strategies = ["head_only", "ln_head", "full_ft"]
    results    = {}

    for strategy in strategies:
        results[strategy] = run_loocv(base_model, subjects, gt, strategy)

    # Summary table
    print("\n" + "=" * 62)
    print("ADAPTER ABLATION SUMMARY")
    print(f"{'Strategy':<15} {'Params':>10} {'MAE':>8} {'±SD':>8} {'95% CI':>18} {'Bias':>8}")
    print("-" * 62)
    for s, r in results.items():
        ci = f"[{r['mae_ci_95'][0]:.4f}-{r['mae_ci_95'][1]:.4f}]"
        print(f"  {s:<13} {r['trainable_params']:>10,} {r['mae']:>8.4f} "
              f"{r['mae_std']:>8.4f} {ci:>18} {r['bias']:>+8.4f}")

    out = {
        "experiment":  "ablation_adapter_strategy",
        "protocol":    "cross_session_loocv",
        "description": "head_only vs ln_head vs full_ft adapter on HFC→HFE LOOCV",
        "results":     results,
    }
    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
