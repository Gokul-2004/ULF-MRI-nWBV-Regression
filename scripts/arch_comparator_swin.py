"""
Architecture Comparator: Swin-UNETR (Reviewer R2.2)
====================================================
Trains a Swin-UNETR-based regressor on OASIS-1 under the IDENTICAL protocol
used for ViT3D and CNN3D (same seed-42 80/10/10 split, same data loading,
same optimizer/LR/epochs/early-stopping), so the resulting MAE / Pearson r
are directly comparable to Table (main) in the manuscript.

Swin-UNETR is a segmentation network; for scalar nWBV regression we use it as
a feature extractor (out_channels=8 feature volume) -> global average pool ->
linear head -> scalar. This mirrors how a hierarchical medical ViT would be
adapted to a global-scalar task.

Model: ~62M params (vs 4.23M ViT3D, 4.1M CNN3D).
Rationale for the comparison: test whether a much larger, hierarchical
transformer improves nWBV regression at n<=375 labelled volumes, or overfits.

Output: experiments/arch_comparator_swin/results.json
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

import finetune_oasis as fo               # reuse EXACT data pipeline
from monai.networks.nets import SwinUNETR

import warnings
warnings.filterwarnings("ignore")

# ── Protocol (mirror finetune_oasis.py exactly) ─────────────────────────────
DEVICE       = torch.device("cpu")
TARGET_SHAPE = (64, 64, 64)
BATCH_SIZE   = 4
LR           = 5e-5
EPOCHS       = 50
PATIENCE     = 10
SEED         = 42
OUT_DIR      = project_root / "experiments" / "arch_comparator_swin"


class SwinRegressor(nn.Module):
    """Swin-UNETR feature extractor -> GAP -> scalar nWBV."""
    def __init__(self, feature_size: int = 48):
        super().__init__()
        self.backbone = SwinUNETR(in_channels=1, out_channels=8,
                                  feature_size=feature_size, spatial_dims=3)
        self.gap  = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(8, 1)

    def forward(self, x):
        v = self.backbone(x)             # (B, 8, 64, 64, 64)
        z = self.gap(v).flatten(1)       # (B, 8)
        return self.head(z)              # (B, 1)


def build_test_split():
    """Reconstruct the identical seed-42 OASIS 80/10/10 split."""
    scans = fo.find_oasis_scans()
    import pandas as pd
    df = pd.read_excel(fo.CSV_PATH)
    df["subj_id"] = df["ID"].str.extract(r"(OAS1_\d{4}_MR\d)")
    df = df.set_index("subj_id")
    records = fo.build_records(scans, df)
    n = len(records)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    np.random.seed(42)
    idx = np.random.permutation(n)
    train = [records[i] for i in idx[:n_train]]
    val   = [records[i] for i in idx[n_train:n_train + n_val]]
    test  = [records[i] for i in idx[n_train + n_val:]]
    return train, val, test


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 62)
    print("Swin-UNETR Architecture Comparator (OASIS, R2.2)")
    print("=" * 62, flush=True)

    train_recs, val_recs, test_recs = build_test_split()
    print(f"Split: train={len(train_recs)} val={len(val_recs)} test={len(test_recs)}",
          flush=True)

    # Same dataset class + converter as ViT3D
    converter = fo.FieldConverter({})
    from torch.utils.data import DataLoader
    train_ds = fo.OASISDataset(train_recs, converter)
    val_ds   = fo.OASISDataset(val_recs,   converter)
    test_ds  = fo.OASISDataset(test_recs,  converter)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = SwinRegressor().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Swin-UNETR regressor: {n_params:,} params", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_val_r = -np.inf
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    patience_count = 0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        tl = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            tl += loss.item()
        tl /= len(train_loader)

        model.eval()
        vp, vy = [], []
        with torch.no_grad():
            for x, y in val_loader:
                vp.append(model(x.to(DEVICE)).cpu().numpy())
                vy.append(y.numpy())
        scheduler.step()
        vp = np.concatenate(vp).flatten()
        vy = np.concatenate(vy).flatten()
        try:
            r_val = float(pearsonr(vy, vp)[0])
        except Exception:
            r_val = 0.0
        history.append({"epoch": epoch + 1, "train_loss": round(tl, 6),
                        "val_pearson_r": round(r_val, 4)})
        print(f"Epoch {epoch+1:3d} | train={tl:.5f} | val_r={r_val:.3f}", flush=True)

        if r_val > best_val_r:
            best_val_r = r_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
            print(f"  ✓ best (r={r_val:.3f})", flush=True)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"Early stop at epoch {epoch+1}", flush=True)
                break

    # Test with best model
    model.load_state_dict(best_state)
    # Save checkpoint for the 64mT frozen-feature transfer probe
    ckpt_path = project_root / "checkpoints" / "swin_oasis.pt"
    torch.save({"model_state_dict": best_state}, ckpt_path)
    print(f"Saved Swin OASIS checkpoint -> {ckpt_path}", flush=True)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x.to(DEVICE)).cpu().numpy())
            trues.append(y.numpy())
    preds = np.concatenate(preds).flatten()
    trues = np.concatenate(trues).flatten()
    mae   = float(np.mean(np.abs(preds - trues)))
    r_test = float(pearsonr(trues, preds)[0]) if len(trues) > 2 else 0.0

    print("\n" + "=" * 62)
    print(f"Swin-UNETR OASIS TEST:  MAE={mae:.4f}  Pearson r={r_test:.3f}  "
          f"(n_test={len(trues)}, params={n_params:,})")
    print("=" * 62, flush=True)

    out = {
        "experiment":  "arch_comparator_swin_unetr",
        "addresses":   ["R2.2"],
        "model":       "SwinUNETR (feature_size=48) + GAP + linear head",
        "params":      n_params,
        "protocol":    "OASIS-1 seed-42 80/10/10, LR=5e-5, Adam, 50ep, early-stop val-r",
        "comparison":  {"vit3d_params": 4230000, "cnn3d_params": 4100000},
        "n_test":      len(trues),
        "mae":         round(mae, 4),
        "pearson_r":   round(r_test, 4),
        "best_val_r":  round(best_val_r, 4),
        "history":     history,
        "per_test": [{"true": round(float(t), 4), "pred": round(float(p), 4)}
                     for t, p in zip(trues, preds)],
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {OUT_DIR / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
