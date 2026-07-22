"""
Fine-tune ViT on OASIS-1 with disease variation.

Training targets (per subject):
  BTF - from CSV nWBV (FreeSurfer normalized whole brain volume)
  VBR - from FSL segmentation (_fseg.img): CSF / (CSF+GM+WM)
  TCR - from T1w + FSL seg: WM_mean / GM_mean
  MCI - from T1w + FSL seg: mean cortical (GM) intensity

FSL seg labels: 0=background, 1=CSF, 2=GM, 3=WM
"""

import sys, glob, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.field_conversion import FieldConverter
from models.baselines import BaselineViT3D

# ── Config ────────────────────────────────────────────────────────────────────
OASIS_PROC_DIR  = project_root / "data" / "oasis_processed"
CSV_PATH        = project_root / "data" / "oasis_raw" / "oasis_cross-sectional.xlsx"
PRETRAIN_CKPT   = project_root / "checkpoints" / "best_model.pt"
SAVE_CKPT       = project_root / "checkpoints" / "oasis_finetuned.pt"
RESULTS_DIR     = project_root / "experiments" / "oasis_finetune"

TARGET_SHAPE    = (64, 64, 64)
BATCH_SIZE      = 4
LR              = 5e-5        # conservative LR for fine-tuning
EPOCHS          = 50
PATIENCE        = 10
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Scan finder ───────────────────────────────────────────────────────────────

def find_oasis_scans():
    """Find processed T1w + FSL seg pairs for all subjects."""
    scans = {}
    for img_path in Path(OASIS_PROC_DIR).rglob("*_mpr_n4_anon_111_t88_gfc.img"):
        parts = img_path.parts
        for part in parts:
            if part.startswith("OAS1_") and "_MR" in part:
                subj_id = "_".join(part.split("_")[:3])
                seg_path = img_path.parent / img_path.name.replace("_gfc.img", "_masked_gfc_fseg.img")
                # Also check FSL_SEG folder
                if not seg_path.exists():
                    seg_candidates = list(Path(OASIS_PROC_DIR).rglob(f"{subj_id}*fseg.img"))
                    seg_path = seg_candidates[0] if seg_candidates else None
                scans[subj_id] = {"t1w": img_path, "seg": seg_path}
                break
    return scans


# ── Biomarker computation ─────────────────────────────────────────────────────

def compute_biomarkers_from_seg(t1w: np.ndarray, seg: np.ndarray, nwbv: float):
    """Compute all 4 biomarkers from T1w volume + FSL segmentation."""
    csf_mask = seg == 1
    gm_mask  = seg == 2
    wm_mask  = seg == 3
    brain_mask = gm_mask | wm_mask | csf_mask

    n_csf = csf_mask.sum()
    n_gm  = gm_mask.sum()
    n_wm  = wm_mask.sum()
    n_brain = n_csf + n_gm + n_wm

    # BTF: use nWBV directly (FreeSurfer gold standard)
    btf = float(nwbv)

    # VBR: CSF fraction within brain (proxy for ventricular enlargement)
    vbr = float(n_csf / n_brain) if n_brain > 0 else 0.0

    # TCR: WM/GM intensity ratio (tissue contrast)
    wm_mean = float(t1w[wm_mask].mean()) if n_wm > 0 else 0.5
    gm_mean = float(t1w[gm_mask].mean()) if n_gm > 0 else 0.5
    tcr = float(wm_mean / (gm_mean + 1e-6))

    # MCI: mean cortical (GM) intensity normalized
    mci = float(t1w[gm_mask].mean()) if n_gm > 0 else 0.5

    return np.array([btf, vbr, tcr, mci], dtype=np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class OASISDataset(Dataset):
    def __init__(self, records, converter):
        self.records   = records
        self.converter = converter

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        # Load T1w
        t1w_img = nib.load(str(rec["t1w"]))
        t1w = t1w_img.get_fdata(dtype=np.float32)
        if t1w.ndim == 4: t1w = t1w[..., 0]
        vmin, vmax = t1w.min(), t1w.max()
        if vmax > vmin: t1w = (t1w - vmin) / (vmax - vmin)

        # Resize to model input
        if t1w.shape != TARGET_SHAPE:
            zoom_f = [t / s for t, s in zip(TARGET_SHAPE, t1w.shape)]
            t1w = zoom(t1w, zoom_f, order=1)

        # Simulate low-field
        lf = self.converter.convert(t1w, method='hyperfine')

        # Single target: nWBV (FreeSurfer ground truth for BTF)
        label = np.array([rec["nwbv"]], dtype=np.float32)

        return torch.from_numpy(lf).unsqueeze(0).float(), torch.from_numpy(label).float()


# ── Build dataset records ─────────────────────────────────────────────────────

def build_records(scans, df):
    records = []
    missing_seg = 0
    for subj_id, paths in scans.items():
        if subj_id not in df.index:
            continue
        if paths["seg"] is None or not paths["seg"].exists():
            missing_seg += 1
            continue
        row = df.loc[subj_id]
        if pd.isna(row["nWBV"]):
            continue
        records.append({
            "subj_id": subj_id,
            "t1w":     paths["t1w"],
            "seg":     paths["seg"],
            "nwbv":    float(row["nWBV"]),
            "cdr":     float(row["CDR"]) if not pd.isna(row["CDR"]) else -1.0,
            "age":     float(row["Age"]),
        })
    print(f"Records with seg: {len(records)}, missing seg: {missing_seg}")
    return records


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("OASIS Fine-tuning")
    print(f"Device: {DEVICE}")
    print("=" * 65)

    # Load CSV
    df = pd.read_excel(CSV_PATH)
    df["subj_id"] = df["ID"].str.extract(r"(OAS1_\d{4}_MR\d)")
    df = df.set_index("subj_id")

    # Find scans
    scans = find_oasis_scans()
    print(f"Found {len(scans)} subjects with T1w scans")
    records = build_records(scans, df)
    print(f"Usable records: {len(records)}")

    if len(records) < 10:
        print("Too few records, aborting.")
        return

    # Split 80/10/10
    n = len(records)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val
    np.random.seed(42)
    idx = np.random.permutation(n)
    train_recs = [records[i] for i in idx[:n_train]]
    val_recs   = [records[i] for i in idx[n_train:n_train+n_val]]
    test_recs  = [records[i] for i in idx[n_train+n_val:]]
    print(f"Split: train={len(train_recs)}, val={len(val_recs)}, test={len(test_recs)}")

    converter = FieldConverter({})
    train_ds  = OASISDataset(train_recs, converter)
    val_ds    = OASISDataset(val_recs,   converter)
    test_ds   = OASISDataset(test_recs,  converter)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Load pretrained model (4-output), freeze encoder, swap head to 1 output (nWBV)
    model = BaselineViT3D(
        img_size=TARGET_SHAPE, patch_size=16, num_classes=4,
        embed_dim=256, num_layers=4, num_heads=8
    )
    ckpt = torch.load(PRETRAIN_CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Swap head: 4 → 1 output
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, 1)

    # Full fine-tuning: all params trainable
    for param in model.parameters():
        param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    print(f"Loaded pretrained weights from epoch {ckpt.get('epoch','?')}")
    print(f"Full fine-tuning: {total:,} params, uniform LR={LR}")

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    criterion = nn.MSELoss()

    best_val_r = -float("inf")   # now tracking best r, not loss
    patience_count = 0
    history = []

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                val_loss += criterion(pred, y).item()
        val_loss /= len(val_loader)
        scheduler.step()

        # Diagnostic: check prediction variance on val set
        val_preds_all, val_labels_all = [], []
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                val_preds_all.append(model(x.to(DEVICE)).cpu().numpy())
                val_labels_all.append(y.numpy())
        import numpy as _np
        vp = _np.concatenate(val_preds_all).flatten()
        vl = _np.concatenate(val_labels_all).flatten()
        pred_std = float(_np.std(vp))
        label_std = float(_np.std(vl))
        from scipy.stats import pearsonr as _pr
        try:
            r_val = float(_pr(vl, vp)[0])
        except Exception:
            r_val = 0.0

        history.append({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss,
                        "val_pearson_r": round(float(r_val), 4)})
        print(f"Epoch {epoch+1:3d} | train={train_loss:.5f} | val={val_loss:.5f} | "
              f"r={r_val:.3f}  pred_std={pred_std:.4f} label_std={label_std:.4f}")

        # Early stop on best val Pearson r (not MSE)
        if r_val > best_val_r:
            best_val_r = r_val
            patience_count = 0
            torch.save({
                "epoch": epoch+1,
                "model_name": "vit_oasis_finetuned",
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
            }, SAVE_CKPT)
            print(f"  ✓ Saved best model (r={r_val:.3f})")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"Early stop at epoch {epoch+1}")
                break

    # Test evaluation
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(SAVE_CKPT, weights_only=False)["model_state_dict"])
    model.eval()

    all_preds, all_labels, all_cdr = [], [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            pred = model(x.to(DEVICE)).cpu().numpy()
            all_preds.append(pred)
            all_labels.append(y.numpy())
            # Grab CDR for these subjects
            batch_start = i * BATCH_SIZE
            for j in range(len(x)):
                si = idx[n_train + n_val + batch_start + j] if batch_start + j < len(test_recs) else 0
                all_cdr.append(test_recs[min(batch_start+j, len(test_recs)-1)]["cdr"])

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_cdr    = np.array(all_cdr[:len(all_preds)])

    from scipy.stats import pearsonr
    btf_r, btf_p = pearsonr(all_labels[:, 0], all_preds[:, 0])

    # CDR group analysis on test set
    cdr_groups = {}
    for i, cdr in enumerate(all_cdr):
        if cdr >= 0:
            g = str(cdr)
            cdr_groups.setdefault(g, []).append(all_preds[i, 0])
    cdr_mean_btf = {g: round(float(np.mean(v)), 4) for g, v in cdr_groups.items()}

    print("\n" + "=" * 65)
    print("TEST RESULTS")
    print("=" * 65)
    print(f"BTF vs nWBV: Pearson r={btf_r:.3f}, p={btf_p:.4f}  (n={len(all_preds)})")
    print(f"\nMean predicted BTF by CDR (test set):")
    for g in sorted(cdr_mean_btf.keys(), key=float):
        print(f"  CDR {g}: {cdr_mean_btf[g]}  (n={len(cdr_groups[g])})")
    print(f"\nExpected: BTF decreases as CDR increases")

    # Per-subject predictions for figure generation (Fig 2 scatter plot)
    per_subject_test = []
    for i in range(len(all_preds)):
        per_subject_test.append({
            "true_nwbv": round(float(all_labels[i, 0]), 4),
            "pred_nwbv": round(float(all_preds[i, 0]),  4),
            "cdr":       float(all_cdr[i]) if all_cdr[i] >= 0 else None,
        })

    # Fisher z CI for Pearson r
    import math
    n_test = len(all_preds)
    z = math.atanh(btf_r)
    se = 1.0 / math.sqrt(n_test - 3)
    ci_lo = math.tanh(z - 1.96 * se)
    ci_hi = math.tanh(z + 1.96 * se)

    results = {
        "best_val_pearson_r":  round(float(best_val_r), 3),
        "test_btf_pearson_r":  round(float(btf_r), 3),
        "test_btf_pearson_p":  round(float(btf_p), 4),
        "fisher_ci_95_lo":     round(ci_lo, 4),
        "fisher_ci_95_hi":     round(ci_hi, 4),
        "cdr_mean_btf":        cdr_mean_btf,
        "per_subject_test":    per_subject_test,
        "history":             history,
    }
    with open(RESULTS_DIR / "finetune_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    train()
