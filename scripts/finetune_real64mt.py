"""
Domain Adaptation: Fine-tune last 2 layers of OASIS model on real 64mT + FreeSurfer GT
========================================================================================
Uses 15 subjects for fine-tuning, 8 held-out for testing.
Only the final norm + head layers are updated (2 layers).

Usage: python3 scripts/finetune_real64mt.py
Output: checkpoints/real64mt_finetuned.pt
        experiments/real64mt_finetune/results.json
"""

import sys, json, random
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
OASIS_CKPT   = project_root / "checkpoints" / "oasis_finetuned.pt"
OUT_CKPT     = project_root / "checkpoints" / "real64mt_finetuned.pt"
OUT_DIR      = project_root / "experiments" / "real64mt_finetune"
TARGET_SHAPE = (64, 64, 64)
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fine-tuning hyperparams
EPOCHS       = 80
LR           = 1e-4
WEIGHT_DECAY = 1e-4
TRAIN_N      = 15   # subjects for fine-tuning
AUG_PER_VOL  = 6    # augmentations per volume
SEED         = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Data helpers ──────────────────────────────────────────────────────────────

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
    """Light augmentation: noise + intensity shift + random flip."""
    v = vol.copy()
    # Gaussian noise
    v += np.random.normal(0, 0.015, v.shape).astype(np.float32)
    # Intensity scale+shift
    v = v * np.random.uniform(0.90, 1.10) + np.random.uniform(-0.05, 0.05)
    # Random axis flip
    if np.random.rand() > 0.5:
        v = np.flip(v, axis=np.random.randint(0, 3)).copy()
    return np.clip(v, 0, 1).astype(np.float32)


def load_gt(csv_path: Path):
    import csv
    gt = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[row['subject']] = float(row['nWBV_freesurfer'])
    return gt


# ── Model helpers ─────────────────────────────────────────────────────────────

def load_model():
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
    """Freeze all params except norm and head (last 2 layers)."""
    for name, param in model.named_parameters():
        if name.startswith('norm.') or name.startswith('head.'):
            param.requires_grad = True
        else:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Real 64mT Domain Adaptation Fine-tuning")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Load ground truth
    gt = load_gt(GT_CSV)
    print(f"Ground truth loaded: {len(gt)} subjects")

    # Load real 64mT scans
    print("\nLoading real 64mT scans (axial T2w)...")
    data = []
    for subj, nwbv in sorted(gt.items()):
        hfc_path = DS_DIR / subj / "ses-HFC" / "anat" / f"{subj}_ses-HFC_acq-axi_T2w.nii.gz"
        if not hfc_path.exists():
            print(f"  MISSING: {hfc_path}")
            continue
        vol = resize64(load_nifti_norm(hfc_path))
        data.append((subj, vol, nwbv))
        print(f"  {subj}: nWBV={nwbv:.4f}")

    print(f"\nLoaded {len(data)} subjects")

    # Split: 15 train, rest test
    random.shuffle(data)
    train_data = data[:TRAIN_N]
    test_data  = data[TRAIN_N:]
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    print(f"Test subjects: {[d[0] for d in test_data]}")

    # Build augmented training set
    X_train, y_train = [], []
    for subj, vol, nwbv in train_data:
        X_train.append(vol)
        y_train.append(nwbv)
        for _ in range(AUG_PER_VOL):
            X_train.append(augment(vol))
            y_train.append(nwbv)

    X_train = torch.tensor(np.array(X_train)).unsqueeze(1).float()
    y_train = torch.tensor(y_train).float().unsqueeze(1)
    print(f"\nTraining samples: {len(X_train)} ({len(train_data)} subjects × {AUG_PER_VOL+1} aug)")

    # Load model, freeze all but last 2 layers
    model = load_model()
    freeze_except_last2(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    # ── Training loop ──
    print(f"\nFine-tuning for {EPOCHS} epochs...")
    best_train_loss = float('inf')
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        # shuffle
        idx = torch.randperm(len(X_train))
        X_train = X_train[idx]
        y_train = y_train[idx]

        epoch_loss = 0.0
        batch_size = 4
        n_batches = 0
        for i in range(0, len(X_train), batch_size):
            xb = X_train[i:i+batch_size].to(DEVICE)
            yb = y_train[i:i+batch_size].to(DEVICE)
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

        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS}  loss={avg_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

    # ── Evaluation on held-out test set ──
    print("\n" + "=" * 60)
    print("Evaluation on held-out test subjects")
    print("=" * 60)

    model.load_state_dict(best_state)
    model.eval()

    preds, trues, names = [], [], []
    with torch.no_grad():
        for subj, vol, nwbv in test_data:
            x = torch.tensor(vol).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
            p = model(x).cpu().item()
            preds.append(p)
            trues.append(nwbv)
            names.append(subj)
            print(f"  {subj}: pred={p:.4f}  GT={nwbv:.4f}  err={p-nwbv:+.4f}")

    preds = np.array(preds)
    trues = np.array(trues)

    r, p_val   = stats.pearsonr(preds, trues)
    rho, p_sp  = stats.spearmanr(preds, trues)
    mae        = float(np.mean(np.abs(preds - trues)))
    rmse       = float(np.sqrt(np.mean((preds - trues)**2)))
    bias       = float(np.mean(preds - trues))

    print(f"\n=== Fine-tuned Model: Real 64mT Test Results ===")
    print(f"n_test        = {len(preds)}")
    print(f"Pearson r     = {r:.4f}  (p={p_val:.4f})")
    print(f"Spearman ρ    = {rho:.4f}  (p={p_sp:.4f})")
    print(f"MAE           = {mae:.4f}")
    print(f"RMSE          = {rmse:.4f}")
    print(f"Mean bias     = {bias:+.4f}")

    # Compare with pre-finetune baseline
    print(f"\nPre-finetune:  r=0.2911  MAE=0.0403  bias=+0.0403")
    print(f"Post-finetune: r={r:.4f}  MAE={mae:.4f}  bias={bias:+.4f}")
    print(f"Improvement:   Δr = {r - 0.2911:+.4f}")

    # Save checkpoint
    torch.save({
        'epoch': EPOCHS,
        'model_name': 'real64mt_finetuned',
        'model_state_dict': best_state,
        'train_loss': best_train_loss,
        'test_pearson_r': round(r, 4),
        'test_mae': round(mae, 4),
        'n_train': len(train_data),
        'n_test': len(test_data),
        'train_subjects': [d[0] for d in train_data],
        'test_subjects': [d[0] for d in test_data],
    }, OUT_CKPT)
    print(f"\nCheckpoint saved: {OUT_CKPT}")

    # Save results JSON
    results = {
        'model': 'real64mt_finetuned',
        'fine_tuning': {
            'n_train_subjects': len(train_data),
            'n_test_subjects': len(test_data),
            'aug_per_vol': AUG_PER_VOL,
            'total_train_samples': len(X_train),
            'epochs': EPOCHS,
            'layers_trained': 'norm + head (last 2)',
            'lr': LR,
        },
        'test_results': {
            'pearson_r': round(r, 4),
            'pearson_p': round(p_val, 4),
            'spearman_r': round(rho, 4),
            'spearman_p': round(p_sp, 4),
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'mean_bias': round(bias, 4),
            'n_test': len(preds),
        },
        'baseline_before_finetune': {
            'pearson_r': 0.2911,
            'pearson_p': 0.1778,
            'mae': 0.0403,
        },
        'per_subject': [
            {'subject': n, 'pred': round(float(p), 4), 'gt': round(float(t), 4),
             'error': round(float(p-t), 4)}
            for n, p, t in zip(names, preds, trues)
        ]
    }

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
