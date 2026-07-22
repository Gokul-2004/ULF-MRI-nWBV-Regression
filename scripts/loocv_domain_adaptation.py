"""
Leave-One-Out Cross-Validation: Domain Adaptation on Real 64mT
==============================================================
For each of 23 subjects:
  - Adapt on the other 22 (norm + head only, 769 params)
  - Predict on the left-out subject
Final metrics computed across all 23 predictions — fully comparable
to SynthSeg+ which is also evaluated on all 23 subjects.

Usage: python3 scripts/loocv_domain_adaptation.py
Output: experiments/real64mt_loocv/loocv_results.json
        paper_figures/fig_loocv_comparison.{pdf,png}
"""

import sys, json, random, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from scipy.ndimage import zoom
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.baselines import BaselineViT3D

# ── Config ────────────────────────────────────────────────────────────────────
DS_DIR       = project_root / "data" / "ds006557_data"
GT_CSV       = project_root / "experiments" / "fastsurfer_output" / "nwbv_ground_truth.csv"
OASIS_CKPT   = project_root / "checkpoints" / "oasis_finetuned.pt"
OUT_DIR      = project_root / "experiments" / "real64mt_loocv"
FIGURES_DIR  = project_root / "paper_figures"
TARGET_SHAPE = (64, 64, 64)
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fine-tuning hyperparams (same as finetune_real64mt.py)
EPOCHS       = 80
LR           = 1e-4
WEIGHT_DECAY = 1e-4
AUG_PER_VOL  = 6
SEED         = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_nifti_norm(path):
    img = nib.load(str(path))
    vol = img.get_fdata(dtype=np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
    if hi > lo:
        vol = np.clip(vol, lo, hi)
        vol = (vol - lo) / (hi - lo)
    return vol.astype(np.float32)


def resize64(vol):
    if vol.shape == TARGET_SHAPE:
        return vol
    factors = [t / s for t, s in zip(TARGET_SHAPE, vol.shape)]
    return zoom(vol, factors, order=1).astype(np.float32)


def augment(vol):
    v = vol.copy()
    v += np.random.normal(0, 0.015, v.shape).astype(np.float32)
    v = v * np.random.uniform(0.90, 1.10) + np.random.uniform(-0.05, 0.05)
    if np.random.rand() > 0.5:
        v = np.flip(v, axis=np.random.randint(0, 3)).copy()
    return np.clip(v, 0, 1).astype(np.float32)


def load_gt():
    import csv
    gt = {}
    with open(GT_CSV) as f:
        for row in csv.DictReader(f):
            gt[row['subject']] = float(row['nWBV_freesurfer'])
    return gt


def load_base_model():
    """Always reload from OASIS checkpoint — never reuse adapted weights."""
    model = BaselineViT3D(
        img_size=TARGET_SHAPE, patch_size=16, num_classes=4,
        embed_dim=256, num_layers=4, num_heads=8
    )
    model.head = nn.Linear(model.head.in_features, 1)
    ckpt = torch.load(OASIS_CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE)
    return model


def freeze_except_last2(model):
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith('norm.') or name.startswith('head.')
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable


def adapt_and_predict(train_data, test_subj, test_vol, test_gt, fold_idx, n_folds):
    """Adapt on train_data, return prediction for test_subj."""

    # Fresh model from OASIS checkpoint every fold
    model = load_base_model()
    n_trainable = freeze_except_last2(model)

    # Build augmented training set
    X_train, y_train = [], []
    for _, vol, nwbv in train_data:
        X_train.append(vol)
        y_train.append(nwbv)
        for _ in range(AUG_PER_VOL):
            X_train.append(augment(vol))
            y_train.append(nwbv)

    X_train = torch.tensor(np.array(X_train)).unsqueeze(1).float()
    y_train = torch.tensor(y_train).float().unsqueeze(1)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    best_state = None

    model.train()
    for epoch in range(EPOCHS):
        idx = torch.randperm(len(X_train))
        X_train = X_train[idx]
        y_train = y_train[idx]

        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(X_train), 4):
            xb = X_train[i:i+4].to(DEVICE)
            yb = y_train[i:i+4].to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Predict on held-out subject
    model.load_state_dict(best_state)
    model.eval()
    x = torch.tensor(test_vol).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        pred = model(x).cpu().item()

    err = pred - test_gt
    print(f"  Fold {fold_idx+1:2d}/{n_folds}  {test_subj}  "
          f"pred={pred:.4f}  GT={test_gt:.4f}  err={err:+.4f}")
    return pred


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LOOCV Domain Adaptation — Real 64mT (n=23)")
    print(f"Device: {DEVICE} | Epochs/fold: {EPOCHS} | AUG: {AUG_PER_VOL}")
    print("=" * 60)

    # Load ground truth
    gt = load_gt()

    # Load all scans
    print("\nLoading real 64mT scans...")
    data = []
    for subj, nwbv in sorted(gt.items()):
        path = DS_DIR / subj / "ses-HFC" / "anat" / f"{subj}_ses-HFC_acq-axi_T2w.nii.gz"
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        vol = resize64(load_nifti_norm(path))
        data.append((subj, vol, nwbv))
    print(f"Loaded {len(data)} subjects\n")

    n = len(data)

    # ── LOOCV loop ────────────────────────────────────────────────────────────
    loocv_preds = []
    loocv_gts   = []
    loocv_names = []

    for i in range(n):
        test_subj, test_vol, test_gt = data[i]
        train_data = [data[j] for j in range(n) if j != i]

        pred = adapt_and_predict(
            train_data, test_subj, test_vol, test_gt,
            fold_idx=i, n_folds=n
        )
        loocv_preds.append(pred)
        loocv_gts.append(test_gt)
        loocv_names.append(test_subj.replace('sub-HYPE', 'S'))

    # ── Metrics ───────────────────────────────────────────────────────────────
    preds = np.array(loocv_preds)
    gts   = np.array(loocv_gts)
    errors = preds - gts
    abs_errors = np.abs(errors)

    r,   p_r   = stats.pearsonr(preds, gts)
    rho, p_rho = stats.spearmanr(preds, gts)
    mae        = float(np.mean(abs_errors))
    rmse       = float(np.sqrt(np.mean(errors**2)))
    bias       = float(np.mean(errors))

    # 95% CI on r via Fisher z
    z    = np.arctanh(r)
    se   = 1.0 / np.sqrt(n - 3)
    ci_lo = float(np.tanh(z - 1.96 * se))
    ci_hi = float(np.tanh(z + 1.96 * se))

    print("\n" + "=" * 60)
    print("LOOCV Results — All 23 Subjects")
    print("=" * 60)
    print(f"Pearson r     = {r:.4f} [95% CI: {ci_lo:.3f}–{ci_hi:.3f}]  (p={p_r:.4f})")
    print(f"Spearman ρ    = {rho:.4f}  (p={p_rho:.4f})")
    print(f"MAE           = {mae:.4f}")
    print(f"RMSE          = {rmse:.4f}")
    print(f"Mean bias     = {bias:+.4f}")
    print(f"\nSynthSeg+ reference: r=0.918, MAE=0.005")
    print(f"Previous (n=8):      r=0.247, MAE=0.010")
    print(f"LOOCV    (n=23):     r={r:.3f}, MAE={mae:.3f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f'LOOCV Domain Adaptation — Real 64\u202fmT (n={n})\n'
                 f'Pearson r = {r:.3f} [95% CI: {ci_lo:.3f}–{ci_hi:.3f}], '
                 f'MAE = {mae:.4f}',
                 fontsize=11, fontweight='bold', color='#1B4F72')

    # Panel A: predicted vs true
    ax = axes[0]
    sc = ax.scatter(gts, preds, c=abs_errors, cmap='RdYlGn_r',
                    s=60, alpha=0.9, edgecolors='white', lw=0.5,
                    vmin=0, vmax=0.06, zorder=3)
    lims = [min(gts.min(), preds.min()) - 0.01,
            max(gts.max(), preds.max()) + 0.01]
    ax.plot(lims, lims, 'k--', lw=1.2, alpha=0.5, label='Perfect prediction')
    m, b_coef = np.polyfit(gts, preds, 1)
    xs = np.linspace(lims[0], lims[1], 100)
    ax.plot(xs, m*xs + b_coef, color='#D35400', lw=2,
            label=f'Fit: r={r:.3f} [CI: {ci_lo:.3f}–{ci_hi:.3f}]')
    plt.colorbar(sc, ax=ax, label='Absolute error', shrink=0.85)
    ax.set_xlabel('True nWBV (FastSurfer)', fontsize=10)
    ax.set_ylabel('Predicted nWBV (LOOCV)', fontsize=10)
    ax.set_title(f'(A) Predicted vs True — LOOCV\n'
                 f'MAE = {mae:.4f}, bias = {bias:+.4f}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)

    # Panel B: per-subject error bar chart
    ax2 = axes[1]
    order = np.argsort(gts)
    colors = ['#C0392B' if e > 0.020 else '#1A7A8A' for e in abs_errors[order]]
    bars = ax2.bar(range(n), abs_errors[order], color=colors,
                   alpha=0.85, edgecolor='white', linewidth=0.8)
    ax2.axhline(0.020, color='#27AE60', ls='--', lw=1.5,
                label='Clinical threshold (0.020)')
    ax2.set_xticks(range(n))
    ax2.set_xticklabels([loocv_names[i] for i in order],
                        rotation=45, ha='right', fontsize=6.5)
    ax2.set_ylabel('Absolute Error (nWBV)', fontsize=10)
    ax2.set_title(f'(B) Per-Subject Absolute Error\n'
                  f'Red = above clinical threshold',
                  fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8.5)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, alpha=0.3)

    fig.text(0.5, -0.02,
             f'LOOCV: adapt on 22 subjects, predict on 1, repeat 23×  |  '
             f'769 trainable params (LayerNorm + head)  |  '
             f'{EPOCHS} epochs/fold',
             ha='center', fontsize=8.5, color='#5D6D7E', style='italic')

    plt.tight_layout(pad=1.5)
    for ext in ['pdf', 'png']:
        fig.savefig(FIGURES_DIR / f"fig_loocv.{ext}",
                    dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nFigure saved: {FIGURES_DIR}/fig_loocv.pdf/png")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    results = {
        'method': 'LOOCV domain adaptation',
        'n_subjects': n,
        'epochs_per_fold': EPOCHS,
        'trainable_params': 769,
        'pearson_r': round(r, 4),
        'pearson_r_ci_lo': round(ci_lo, 4),
        'pearson_r_ci_hi': round(ci_hi, 4),
        'pearson_p': round(p_r, 4),
        'spearman_rho': round(rho, 4),
        'spearman_p': round(p_rho, 4),
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'bias': round(bias, 4),
        'synthseg_reference': {'r': 0.918, 'mae': 0.005, 'n': 23},
        'previous_split': {'r': 0.247, 'mae': 0.010, 'n_test': 8},
        'per_subject': [
            {'subject': loocv_names[i], 'pred': round(float(preds[i]), 4),
             'gt': round(float(gts[i]), 4), 'error': round(float(errors[i]), 4),
             'abs_error': round(float(abs_errors[i]), 4)}
            for i in range(n)
        ]
    }
    with open(OUT_DIR / "loocv_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {OUT_DIR}/loocv_results.json")
    print("\nUpdate manuscript Table 1 with LOOCV r and MAE (n=23 instead of n=8).")


if __name__ == '__main__':
    main()
