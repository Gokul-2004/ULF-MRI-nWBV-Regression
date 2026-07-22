"""
Ablation: Physics Simulation vs Gaussian Blur Pre-training
===========================================================
Trains two ViT3D models from scratch on IXI data:
  (A) Physics-sim pre-training  → fine-tune on OASIS → evaluate
  (B) Gaussian blur pre-training → fine-tune on OASIS → evaluate

Tests the core claim: physics-constrained simulation is better
than simple degradation for transfer to low-field MRI.

Usage: python3 scripts/ablation_gaussianblur.py
Output: experiments/ablation_gaussblur/results.json
        paper_figures/fig_ablation_gaussblur.{pdf,png}
"""

import sys, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from scipy.ndimage import zoom, gaussian_filter
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.baselines import BaselineViT3D
from utils.field_conversion import FieldConverter

# ── Config ─────────────────────────────────────────────────────────────────
IXI_DIR       = project_root / "data" / "high_field"
OASIS_DIR     = project_root / "data" / "oasis_processed"
OASIS_CSV     = project_root / "data" / "oasis_raw" / "oasis_cross-sectional.xlsx"
OASIS_CKPT    = project_root / "checkpoints" / "oasis_finetuned.pt"   # physics-trained
OUT_DIR       = project_root / "experiments" / "ablation_gaussblur"
FIGURES_DIR   = project_root / "paper_figures"
TARGET_SHAPE  = (64, 64, 64)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRETRAIN_EPOCHS = 30
FINETUNE_EPOCHS = 30
LR_PRETRAIN   = 1e-4
LR_FINETUNE   = 5e-5
BATCH_SIZE    = 4
SEED          = 42
N_IXI         = 80   # subjects for pre-training ablation

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────

def load_npy_norm(path):
    vol = np.load(str(path)).astype(np.float32)
    if vol.ndim == 4: vol = vol[0]
    lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
    if hi > lo: vol = np.clip((vol - lo) / (hi - lo), 0, 1)
    return vol.astype(np.float32)


def resize64(vol):
    if vol.shape == TARGET_SHAPE: return vol
    factors = [t / s for t, s in zip(TARGET_SHAPE, vol.shape)]
    return zoom(vol, factors, order=1).astype(np.float32)


def gaussian_blur_degradation(vol, sigma=2.0):
    """Simple Gaussian blur + noise — the naive baseline."""
    blurred = gaussian_filter(vol, sigma=sigma)
    noise = np.random.normal(0, 0.04, vol.shape).astype(np.float32)
    return np.clip(blurred + noise, 0, 1).astype(np.float32)


def physics_sim_degradation(vol, converter):
    """Full physics-constrained simulation."""
    low = converter.convert(vol, method='combined')
    return np.clip(low, 0, 1).astype(np.float32)


def augment(vol):
    v = vol.copy()
    v += np.random.normal(0, 0.02, v.shape).astype(np.float32)
    v = v * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05)
    for ax in range(3):
        if np.random.rand() > 0.5: v = np.flip(v, axis=ax).copy()
    return np.clip(v, 0, 1).astype(np.float32)


# ── Load IXI high-field volumes ────────────────────────────────────────────

def load_ixi(n=N_IXI):
    files = sorted(Path(IXI_DIR).glob("IXI_*.npy"))[:n]
    vols = []
    for f in files:
        v = resize64(load_npy_norm(f))
        vols.append(v)
    print(f"  Loaded {len(vols)} IXI volumes")
    return vols


# ── Load OASIS test set (stored predictions from physics model for reference)
def load_oasis_test():
    """Return test set nWBV GT and stored physics-model predictions."""
    results_path = project_root / "experiments" / "oasis_finetune" / "finetune_results.json"
    with open(results_path) as f:
        d = json.load(f)
    trues = np.array([s['true_nwbv'] for s in d['per_subject_test']])
    preds_physics = np.array([s['pred_nwbv'] for s in d['per_subject_test']])
    return trues, preds_physics


# ── Load OASIS training volumes + labels ──────────────────────────────────

def load_oasis_data():
    import pandas as pd
    df = pd.read_excel(OASIS_CSV)
    df = df[df['nWBV'].notna() & df['ID'].notna()]

    data = []
    for _, row in df.iterrows():
        subj_id = str(row['ID']).strip()
        nwbv = float(row['nWBV'])
        imgs = list(Path(OASIS_DIR).rglob(f"{subj_id}*_mpr_n4_anon_111_t88_gfc.img"))
        if not imgs: continue
        try:
            import nibabel as nib
            vol_raw = nib.load(str(imgs[0])).get_fdata(dtype=np.float32)
            if vol_raw.ndim == 4: vol_raw = vol_raw[..., 0]
            lo, hi = np.percentile(vol_raw, 1), np.percentile(vol_raw, 99)
            if hi > lo: vol_raw = np.clip((vol_raw - lo) / (hi - lo), 0, 1)
            vol = resize64(vol_raw.astype(np.float32))
            data.append((subj_id, vol, nwbv))
        except Exception:
            continue
    return data


# ── Pre-training: paired high→low regression ──────────────────────────────

def pretrain(model, ixi_vols, degradation_fn, label='physics'):
    """Pre-train as encoder via reconstruction loss (predict SNR from degraded)."""
    print(f"  Pre-training ({label}) on {len(ixi_vols)} IXI volumes × 3 aug = {len(ixi_vols)*3} samples...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_PRETRAIN, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS, eta_min=1e-6)
    criterion = nn.MSELoss()

    # Build dataset: predict nWBV proxy = mean brain intensity ratio (high/low)
    # This teaches the model about signal degradation patterns
    X_pairs, y_proxy = [], []
    for vol in ixi_vols:
        brain_mask = vol > 0.1
        brain_mean = float(vol[brain_mask].mean()) if brain_mask.any() else 0.5
        for _ in range(3):
            low = degradation_fn(augment(vol))
            X_pairs.append(low)
            y_proxy.append(brain_mean)  # predict original brain intensity from degraded

    X = torch.tensor(np.array(X_pairs)).unsqueeze(1).float()
    y = torch.tensor(y_proxy).float().unsqueeze(1)

    best_loss = float('inf')
    best_state = None
    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        idx = torch.randperm(len(X))
        X_s, y_s = X[idx], y[idx]
        epoch_loss = 0; nb = 0
        for i in range(0, len(X_s), BATCH_SIZE):
            xb = X_s[i:i+BATCH_SIZE].to(DEVICE)
            yb = y_s[i:i+BATCH_SIZE].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item(); nb += 1
        scheduler.step()
        avg = epoch_loss / nb
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch+1) % 10 == 0:
            print(f"    epoch {epoch+1}/{PRETRAIN_EPOCHS}  loss={avg:.5f}")
    model.load_state_dict(best_state)
    return model


# ── Fine-tune on OASIS ─────────────────────────────────────────────────────

def finetune_oasis(model, oasis_data, label=''):
    """Fine-tune on OASIS nWBV — same split as main model (first 80% train)."""
    random.shuffle(oasis_data)
    split = int(0.8 * len(oasis_data))
    train_data = oasis_data[:split]
    test_data  = oasis_data[split:]

    X_train, y_train = [], []
    for _, vol, nwbv in train_data:
        X_train.append(vol); y_train.append(nwbv)
        for _ in range(2):
            X_train.append(augment(vol)); y_train.append(nwbv)

    X = torch.tensor(np.array(X_train)).unsqueeze(1).float()
    y = torch.tensor(y_train).float().unsqueeze(1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS, eta_min=1e-7)
    criterion = nn.MSELoss()

    best_loss = float('inf'); best_state = None
    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        idx = torch.randperm(len(X))
        X_s, y_s = X[idx], y[idx]
        el = 0; nb = 0
        for i in range(0, len(X_s), BATCH_SIZE):
            xb = X_s[i:i+BATCH_SIZE].to(DEVICE)
            yb = y_s[i:i+BATCH_SIZE].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            el += loss.item(); nb += 1
        scheduler.step()
        avg = el / nb
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch+1) % 10 == 0:
            print(f"    [{label}] finetune epoch {epoch+1}/{FINETUNE_EPOCHS}  loss={avg:.5f}")

    model.load_state_dict(best_state)

    # Evaluate on test split
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for _, vol, nwbv in test_data:
            x = torch.tensor(vol).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
            p = float(model(x).cpu().item())
            preds.append(p); trues.append(nwbv)
    preds = np.array(preds); trues = np.array(trues)
    r, p_val = stats.pearsonr(preds, trues)
    mae = float(np.mean(np.abs(preds - trues)))
    return r, p_val, mae, preds, trues, test_data


# ── Figure ─────────────────────────────────────────────────────────────────

def make_figure(results):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Ablation: Physics-Constrained Simulation vs Gaussian Blur Pre-training\n'
                 'ViT3D fine-tuned on OASIS-1 (n=375)',
                 fontsize=12, fontweight='bold', color='#1B4F72')

    methods = ['Physics Sim\n(Ours)', 'Gaussian Blur\n(Baseline)']
    colors  = ['#1A7A8A', '#E67E22']

    # Panel A: Pearson r
    ax = axes[0]
    rs = [results['physics']['r'], results['blur']['r']]
    cis = [(results['physics']['r'] - results['physics']['r_ci_lo'],
            results['physics']['r_ci_hi'] - results['physics']['r']),
           (results['blur']['r'] - results['blur']['r_ci_lo'],
            results['blur']['r_ci_hi'] - results['blur']['r'])]
    bars = ax.bar(methods, rs, color=colors, alpha=0.85, width=0.4,
                  edgecolor='white', linewidth=1.2)
    for bar, ci, r_val in zip(bars, cis, rs):
        ax.errorbar(bar.get_x() + bar.get_width()/2, r_val,
                    yerr=[[ci[0]], [ci[1]]], fmt='none',
                    color='#2C3E50', capsize=5, linewidth=1.5)
        ax.text(bar.get_x() + bar.get_width()/2, r_val + 0.01,
                f'{r_val:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Pearson r', fontsize=10)
    ax.set_title('(A) Correlation with FreeSurfer nWBV', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)

    # Panel B: MAE
    ax2 = axes[1]
    maes = [results['physics']['mae'], results['blur']['mae']]
    bars2 = ax2.bar(methods, maes, color=colors, alpha=0.85, width=0.4,
                    edgecolor='white', linewidth=1.2)
    for bar, mae in zip(bars2, maes):
        ax2.text(bar.get_x() + bar.get_width()/2, mae + 0.001,
                 f'{mae:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax2.set_ylabel('MAE (nWBV units)', fontsize=10)
    ax2.set_title('(B) Mean Absolute Error', fontsize=10, fontweight='bold')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, alpha=0.3)

    # Panel C: scatter both
    ax3 = axes[2]
    t_p = results['physics']['trues']; p_p = results['physics']['preds']
    t_b = results['blur']['trues'];    p_b = results['blur']['preds']
    ax3.scatter(t_p, p_p, c='#1A7A8A', s=45, alpha=0.8,
                edgecolors='white', lw=0.5, label=f'Physics sim (r={results["physics"]["r"]:.3f})', zorder=3)
    ax3.scatter(t_b, p_b, c='#E67E22', s=45, alpha=0.8, marker='^',
                edgecolors='white', lw=0.5, label=f'Gaussian blur (r={results["blur"]["r"]:.3f})', zorder=3)
    all_vals = np.concatenate([t_p, p_p, t_b, p_b])
    lims = [all_vals.min()-0.01, all_vals.max()+0.01]
    ax3.plot(lims, lims, 'k--', lw=1.2, alpha=0.4, label='Perfect')
    ax3.set_xlabel('True nWBV (FreeSurfer)', fontsize=10)
    ax3.set_ylabel('Predicted nWBV', fontsize=10)
    ax3.set_title('(C) Predicted vs True nWBV', fontsize=10, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.set_xlim(lims); ax3.set_ylim(lims)
    ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
    ax3.yaxis.grid(True, alpha=0.3)

    plt.tight_layout(pad=1.5)
    for ext in ['pdf', 'png']:
        fig.savefig(FIGURES_DIR / f"fig_ablation_gaussblur.{ext}",
                    dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Figure saved.")


# ── Main ───────────────────────────────────────────────────────────────────

def make_model():
    m = BaselineViT3D(img_size=TARGET_SHAPE, patch_size=16, num_classes=4,
                      embed_dim=256, num_layers=4, num_heads=8)
    m.head = nn.Linear(m.head.in_features, 1)
    m.to(DEVICE)
    return m


def fisher_ci(r, n):
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96*se)), float(np.tanh(z + 1.96*se))


if __name__ == '__main__':
    print("="*60)
    print("Ablation: Physics Sim vs Gaussian Blur Pre-training")
    print(f"Device: {DEVICE}")
    print("="*60)

    converter = FieldConverter(config={'method': 'hyperfine'})

    print("\nLoading IXI volumes...")
    ixi_vols = load_ixi(N_IXI)

    print("\nLoading OASIS training data...")
    oasis_data = load_oasis_data()
    print(f"  Loaded {len(oasis_data)} OASIS subjects")

    results = {}

    # ── Physics-sim model ──
    print("\n[1/2] Physics-Constrained Pre-training...")
    model_phys = make_model()
    model_phys = pretrain(model_phys, ixi_vols,
                          lambda v: physics_sim_degradation(v, converter),
                          label='physics')
    r_p, pval_p, mae_p, preds_p, trues_p, _ = finetune_oasis(model_phys, list(oasis_data), label='physics')
    ci_lo_p, ci_hi_p = fisher_ci(r_p, len(preds_p))
    print(f"  Physics: r={r_p:.4f} [{ci_lo_p:.3f}–{ci_hi_p:.3f}]  MAE={mae_p:.4f}")
    results['physics'] = {
        'r': round(r_p, 4), 'r_ci_lo': round(ci_lo_p, 4), 'r_ci_hi': round(ci_hi_p, 4),
        'pearson_p': round(pval_p, 4), 'mae': round(mae_p, 4),
        'n_test': len(preds_p),
        'preds': preds_p.tolist(), 'trues': trues_p.tolist()
    }

    # ── Gaussian blur model ──
    print("\n[2/2] Gaussian Blur Pre-training...")
    model_blur = make_model()
    model_blur = pretrain(model_blur, ixi_vols,
                          lambda v: gaussian_blur_degradation(v, sigma=2.0),
                          label='blur')
    r_b, pval_b, mae_b, preds_b, trues_b, _ = finetune_oasis(model_blur, list(oasis_data), label='blur')
    ci_lo_b, ci_hi_b = fisher_ci(r_b, len(preds_b))
    print(f"  Gaussian blur: r={r_b:.4f} [{ci_lo_b:.3f}–{ci_hi_b:.3f}]  MAE={mae_b:.4f}")
    results['blur'] = {
        'r': round(r_b, 4), 'r_ci_lo': round(ci_lo_b, 4), 'r_ci_hi': round(ci_hi_b, 4),
        'pearson_p': round(pval_b, 4), 'mae': round(mae_b, 4),
        'n_test': len(preds_b),
        'preds': preds_b.tolist(), 'trues': trues_b.tolist()
    }

    # Convert np arrays for JSON serialization
    results['physics']['preds'] = [round(float(x), 4) for x in preds_p]
    results['physics']['trues'] = [round(float(x), 4) for x in trues_p]
    results['blur']['preds']    = [round(float(x), 4) for x in preds_b]
    results['blur']['trues']    = [round(float(x), 4) for x in trues_b]

    # Keep np arrays for figure
    results['physics']['preds_arr'] = preds_p
    results['physics']['trues_arr'] = trues_p
    results['blur']['preds_arr']    = preds_b
    results['blur']['trues_arr']    = trues_b

    # Summary
    delta_r   = r_p - r_b
    delta_mae = mae_b - mae_p
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Physics sim:   r={r_p:.4f}  MAE={mae_p:.4f}")
    print(f"Gaussian blur: r={r_b:.4f}  MAE={mae_b:.4f}")
    print(f"Improvement:   Δr={delta_r:+.4f}  ΔMAE={delta_mae:+.4f}")

    # Figure
    print("\nGenerating figure...")
    fig_results = {
        'physics': {**results['physics'], 'preds': preds_p, 'trues': trues_p},
        'blur':    {**results['blur'],    'preds': preds_b, 'trues': trues_b},
    }
    make_figure(fig_results)

    # Save JSON (without np arrays)
    save = {
        'ablation': 'physics_sim_vs_gaussian_blur',
        'n_ixi_pretrain': len(ixi_vols),
        'pretrain_epochs': PRETRAIN_EPOCHS,
        'finetune_epochs': FINETUNE_EPOCHS,
        'physics_sim': {k: v for k, v in results['physics'].items() if not k.endswith('_arr')},
        'gaussian_blur': {k: v for k, v in results['blur'].items() if not k.endswith('_arr')},
        'delta_r': round(delta_r, 4),
        'delta_mae': round(delta_mae, 4),
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nResults saved to {OUT_DIR}/results.json")
