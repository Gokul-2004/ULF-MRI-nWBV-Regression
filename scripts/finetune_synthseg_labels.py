"""
Full Fine-tune on Real 64mT using SynthSeg+ Pseudo-Labels
==========================================================
Strategy:
  - Use SynthSeg+ nWBV (r=0.918 vs FreeSurfer) as supervision signal
  - Train on 15 real 64mT subjects, test on 8 held-out
  - Full fine-tune (all layers, low LR) — not just head
  - Heavy augmentation to compensate for small n

This directly addresses the sim-to-real gap using real Hyperfine
images with reliable pseudo-labels from SynthSeg+.

Usage: python3 scripts/finetune_synthseg_labels.py
Output: checkpoints/synthseg_finetuned.pt
        experiments/synthseg_finetune/results.json
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

# ── Config ─────────────────────────────────────────────────────────────────
DS_DIR        = project_root / "data" / "ds006557_data"
SYNTHSEG_CSV  = project_root / "experiments" / "synthseg_output" / "nwbv_synthseg.csv"
FASTSURFER_CSV= project_root / "experiments" / "fastsurfer_output" / "nwbv_ground_truth.csv"
OASIS_CKPT    = project_root / "checkpoints" / "oasis_finetuned.pt"
OUT_CKPT      = project_root / "checkpoints" / "synthseg_finetuned.pt"
OUT_DIR       = project_root / "experiments" / "synthseg_finetune"
TARGET_SHAPE  = (64, 64, 64)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fine-tuning hyperparams — full fine-tune, low LR
EPOCHS        = 120
LR            = 3e-5       # low LR for full fine-tune
WEIGHT_DECAY  = 1e-4
TRAIN_N       = 15
AUG_PER_VOL   = 10         # more augmentation for small dataset
SEED          = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Helpers ────────────────────────────────────────────────────────────────

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
    factors = [t / s for t, s in zip(TARGET_SHAPE, vol.shape)]
    return zoom(vol, factors, order=1).astype(np.float32)


def augment(vol):
    """Heavier augmentation for small dataset."""
    v = vol.copy()
    # Gaussian noise
    v += np.random.normal(0, np.random.uniform(0.01, 0.03), v.shape).astype(np.float32)
    # Intensity scale + shift
    v = v * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.07, 0.07)
    # Random flip (all axes)
    for ax in range(3):
        if np.random.rand() > 0.5:
            v = np.flip(v, axis=ax).copy()
    # Gamma correction
    gamma = np.random.uniform(0.8, 1.2)
    v = np.power(np.clip(v, 0, 1), gamma)
    # Random crop + resize (simulates zoom variation)
    if np.random.rand() > 0.5:
        crop = np.random.randint(2, 6)
        v = v[crop:-crop, crop:-crop, crop:-crop]
        v = resize64(v)
    return np.clip(v, 0, 1).astype(np.float32)


def load_csv(path):
    import csv
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            subj = row['subject']
            # SynthSeg CSV
            if 'nWBV_synthseg' in row:
                data[subj] = float(row['nWBV_synthseg'])
            # FastSurfer CSV
            elif 'nWBV_freesurfer' in row:
                data[subj] = float(row['nWBV_freesurfer'])
    return data


def load_ages():
    tsv = DS_DIR / "participants.tsv"
    ages = {}
    if tsv.exists():
        with open(tsv) as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try: ages['sub-' + parts[0]] = float(parts[1])
                    except: pass
    return ages


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Full Fine-tune: Real 64mT + SynthSeg+ Pseudo-Labels")
    print(f"Device: {DEVICE} | Epochs: {EPOCHS} | LR: {LR}")
    print("=" * 60)

    # Load labels
    synthseg_labels = load_csv(SYNTHSEG_CSV)
    fastsurfer_labels = load_csv(FASTSURFER_CSV)
    ages = load_ages()
    print(f"SynthSeg labels: {len(synthseg_labels)}")
    print(f"FastSurfer GT:   {len(fastsurfer_labels)}")

    # Load all 23 real 64mT scans
    print("\nLoading real 64mT scans...")
    data = []
    for subj, nwbv_ss in sorted(synthseg_labels.items()):
        path = DS_DIR / subj / "ses-HFC" / "anat" / f"{subj}_ses-HFC_acq-axi_T2w.nii.gz"
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        vol = resize64(load_nifti_norm(path))
        nwbv_fs = fastsurfer_labels.get(subj, nwbv_ss)  # prefer FastSurfer GT
        data.append((subj, vol, nwbv_ss, nwbv_fs))

    print(f"Loaded: {len(data)} subjects")

    # Split
    random.shuffle(data)
    train_data = data[:TRAIN_N]
    test_data  = data[TRAIN_N:]
    print(f"Train: {len(train_data)} | Test: {len(test_data)}")
    print(f"Test:  {[d[0] for d in test_data]}")

    # Build augmented training set using SynthSeg pseudo-labels
    X_train, y_train = [], []
    for subj, vol, nwbv_ss, _ in train_data:
        X_train.append(vol)
        y_train.append(nwbv_ss)
        for _ in range(AUG_PER_VOL):
            X_train.append(augment(vol))
            y_train.append(nwbv_ss)

    X_train = torch.tensor(np.array(X_train)).unsqueeze(1).float()
    y_train = torch.tensor(y_train).float().unsqueeze(1)
    print(f"Train samples: {len(X_train)} ({TRAIN_N} × {AUG_PER_VOL+1} aug)")

    # Load OASIS fine-tuned model — full fine-tune all layers
    model = BaselineViT3D(
        img_size=TARGET_SHAPE, patch_size=16, num_classes=4,
        embed_dim=256, num_layers=4, num_heads=8
    )
    model.head = nn.Linear(model.head.in_features, 1)
    ckpt = torch.load(OASIS_CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE)

    # ALL layers trainable
    for p in model.parameters():
        p.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: ALL {total:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.MSELoss()

    # ── Training ──
    print(f"\nFine-tuning for {EPOCHS} epochs...")
    best_loss = float('inf')
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(X_train))
        X_s, y_s = X_train[idx], y_train[idx]

        epoch_loss = 0.0; n_batches = 0
        for i in range(0, len(X_s), 4):
            xb = X_s[i:i+4].to(DEVICE)
            yb = y_s[i:i+4].to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item(); n_batches += 1

        scheduler.step()
        avg = epoch_loss / n_batches

        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS}  loss={avg:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

    # ── Evaluation ──
    model.load_state_dict(best_state)
    model.eval()

    print(f"\n{'='*60}")
    print("Evaluation on held-out test subjects")
    print(f"{'='*60}")

    preds_ss, preds_fs, trues_ss, trues_fs, names, age_pairs = [], [], [], [], [], []

    with torch.no_grad():
        for subj, vol, nwbv_ss, nwbv_fs in test_data:
            x = torch.tensor(vol).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
            p = float(model(x).cpu().item())
            preds_ss.append(p); preds_fs.append(p)
            trues_ss.append(nwbv_ss); trues_fs.append(nwbv_fs)
            names.append(subj)
            if subj in ages: age_pairs.append((ages[subj], p))
            print(f"  {subj}: pred={p:.4f}  SynthSeg={nwbv_ss:.4f}  FreeSurfer={nwbv_fs:.4f}")

    preds = np.array(preds_fs)
    trues = np.array(trues_fs)

    r,   p_val = stats.pearsonr(preds, trues)
    rho, p_sp  = stats.spearmanr(preds, trues)
    mae        = float(np.mean(np.abs(preds - trues)))
    rmse       = float(np.sqrt(np.mean((preds - trues)**2)))
    bias       = float(np.mean(preds - trues))

    age_r = age_rho = age_p = None
    if len(age_pairs) >= 4:
        a = np.array([x[0] for x in age_pairs])
        p_a = np.array([x[1] for x in age_pairs])
        age_r,  _     = stats.pearsonr(a, p_a)
        age_rho, age_p = stats.spearmanr(a, p_a)

    print(f"\n{'='*60}")
    print("RESULTS vs FreeSurfer Ground Truth")
    print(f"{'='*60}")
    print(f"n_test        = {len(preds)}")
    print(f"Pearson r     = {r:.4f}  (p={p_val:.4f})")
    print(f"Spearman ρ    = {rho:.4f}  (p={p_sp:.4f})")
    print(f"MAE           = {mae:.4f}")
    print(f"RMSE          = {rmse:.4f}")
    print(f"Mean bias     = {bias:+.4f}")
    if age_r:
        print(f"Age Pearson r = {age_r:.4f}")
        print(f"Age Spearman ρ= {age_rho:.4f}  (p={age_p:.4f})")

    print(f"\n--- Comparison ---")
    print(f"Before (OASIS only):     r=0.2911  MAE=0.0403")
    print(f"After (SynthSeg adapt):  r={r:.4f}  MAE={mae:.4f}")
    print(f"Improvement:             Δr={r-0.2911:+.4f}  ΔMAE={mae-0.0403:+.4f}")

    # Save
    torch.save({
        'epoch': EPOCHS,
        'model_name': 'synthseg_finetuned',
        'model_state_dict': best_state,
        'train_loss': best_loss,
        'test_pearson_r': round(r, 4),
        'test_mae': round(mae, 4),
        'train_subjects': [d[0] for d in train_data],
        'test_subjects':  [d[0] for d in test_data],
    }, OUT_CKPT)
    print(f"\nCheckpoint: {OUT_CKPT}")

    results = {
        'model': 'ViT3D full fine-tune on real 64mT (SynthSeg+ labels)',
        'supervision': 'SynthSeg+ nWBV pseudo-labels (r=0.918 vs FreeSurfer)',
        'fine_tuning': {
            'n_train': TRAIN_N, 'n_test': len(test_data),
            'aug_per_vol': AUG_PER_VOL,
            'total_train_samples': len(X_train),
            'epochs': EPOCHS, 'lr': LR,
            'layers': 'all (full fine-tune)',
        },
        'test_vs_freesurfer_gt': {
            'pearson_r': round(r, 4), 'pearson_p': round(p_val, 4),
            'spearman_r': round(rho, 4), 'spearman_p': round(p_sp, 4),
            'mae': round(mae, 4), 'rmse': round(rmse, 4),
            'mean_bias': round(bias, 4), 'n': len(preds),
        },
        'age_correlation': {
            'pearson_r': round(float(age_r), 4) if age_r else None,
            'spearman_r': round(float(age_rho), 4) if age_rho else None,
            'spearman_p': round(float(age_p), 4) if age_p else None,
        },
        'baseline_oasis_only': {'pearson_r': 0.2911, 'mae': 0.0403},
        'per_subject': [
            {'subject': n, 'pred': round(float(p), 4),
             'synthseg_gt': round(float(s), 4), 'freesurfer_gt': round(float(f), 4)}
            for n, p, s, f in zip(names, preds_fs, trues_ss, trues_fs)
        ]
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results: {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
