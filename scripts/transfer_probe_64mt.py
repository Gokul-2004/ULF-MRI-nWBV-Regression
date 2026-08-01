"""
64 mT Frozen-Feature Transfer Probe (Reviewer R2.2 — the decisive test)
========================================================================
Question: do the OASIS-learned representations of a large hierarchical
transformer (Swin-UNETR) transfer to the REAL 64 mT target better than those
of the compact ViT3D? This is the test that decides whether Swin's high-field
accuracy advantage survives the domain gap — R2's core doubt.

Method (feasible without per-fold training of a 62 M model):
  1. Freeze each OASIS-trained encoder (ViT3D, Swin-UNETR).
  2. Extract a fixed feature vector per real 64 mT scan (HFC session, n=23).
  3. Leave-one-subject-out ridge regression on those frozen features -> nWBV.
  4. Compare LOOCV MAE / r between ViT3D features and Swin features.

This is a *representation-transfer* probe (frozen backbone + linear head), not
the full adapted pipeline, so absolute MAE differs from the headline 0.0134.
The comparison between backbones is the point.

Output: experiments/transfer_probe_64mt/results.json
"""

import sys, json, csv, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from scipy.ndimage import zoom
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from models.baselines import BaselineViT3D
from monai.networks.nets import SwinUNETR

DS_DIR   = project_root / "data" / "ds006557_data"
GT_CSV   = project_root / "experiments" / "fastsurfer_output" / "nwbv_ground_truth.csv"
VIT_CKPT = project_root / "checkpoints" / "oasis_finetuned.pt"
SWIN_CKPT= project_root / "checkpoints" / "swin_oasis.pt"
TARGET   = (64, 64, 64)
DEVICE   = torch.device("cpu")
OUT_DIR  = project_root / "experiments" / "transfer_probe_64mt"


# ── data (identical preprocessing to loocv_cross_session.py) ────────────────
def load_nifti_norm(path):
    vol = nib.load(str(path)).get_fdata(dtype=np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
    if hi > lo:
        vol = np.clip(vol, lo, hi); vol = (vol - lo) / (hi - lo)
    return vol.astype(np.float32)

def resize64(vol):
    if vol.shape == TARGET:
        return vol
    f = [t / s for t, s in zip(TARGET, vol.shape)]
    return zoom(vol, f, order=1).astype(np.float32)

def load_scan(subject, session="HFC"):
    p = DS_DIR / subject / f"ses-{session}" / "anat" / f"{subject}_ses-{session}_acq-axi_T2w.nii.gz"
    if not p.exists():
        return None
    return resize64(load_nifti_norm(p))

def load_gt():
    gt = {}
    with open(GT_CSV) as f:
        for row in csv.DictReader(f):
            gt[row["subject"]] = float(row["nWBV_freesurfer"])
    return gt


# ── feature extractors (frozen) ─────────────────────────────────────────────
def vit_features(subjects):
    m = BaselineViT3D(img_size=TARGET, patch_size=16, num_classes=4,
                      embed_dim=256, num_layers=4, num_heads=8)
    m.head = nn.Linear(m.head.in_features, 1)
    m.load_state_dict(torch.load(VIT_CKPT, map_location=DEVICE, weights_only=False)["model_state_dict"])
    m.eval()
    feats, ys, ids = [], [], []
    gt = load_gt()
    for s in subjects:
        v = load_scan(s)
        if v is None:
            continue
        x = torch.tensor(v).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            pe = m.patch_embed(x)                      # (1,256,4,4,4)
            tok = pe.flatten(2).transpose(1, 2)        # (1,64,256)
            h = tok
            for blk in m.blocks:
                h = blk(h)
            h = m.norm(h)                              # (1,64,256)
            feat = h.mean(dim=1).flatten().numpy()     # (256,) mean-pooled token features
        feats.append(feat); ys.append(gt[s]); ids.append(s)
    return np.array(feats), np.array(ys), ids

def swin_features(subjects):
    class SwinReg(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = SwinUNETR(in_channels=1, out_channels=8, feature_size=48, spatial_dims=3)
            self.gap = nn.AdaptiveAvgPool3d(1); self.head = nn.Linear(8, 1)
        def forward(self, x):
            return self.head(self.gap(self.backbone(x)).flatten(1))
    m = SwinReg()
    m.load_state_dict(torch.load(SWIN_CKPT, map_location=DEVICE, weights_only=False)["model_state_dict"])
    m.eval()
    feats, ys, ids = [], [], []
    gt = load_gt()
    for s in subjects:
        v = load_scan(s)
        if v is None:
            continue
        x = torch.tensor(v).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            vol = m.backbone(x)                        # (1,8,64,64,64)
            feat = m.gap(vol).flatten().numpy()        # (8,) — GAP feature the head sees
        feats.append(feat); ys.append(gt[s]); ids.append(s)
    return np.array(feats), np.array(ys), ids


# ── leave-one-out ridge on frozen features ─────────────────────────────────
def loo_ridge(X, y, alpha=1.0):
    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    for tr, te in loo.split(X):
        r = Ridge(alpha=alpha).fit(X[tr], y[tr])
        preds[te] = r.predict(X[te])
    mae = float(np.mean(np.abs(preds - y)))
    r_val = float(pearsonr(y, preds)[0]) if len(y) > 2 else 0.0
    return mae, r_val, preds


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = load_gt()
    subjects = sorted(s for s in gt if (DS_DIR / s).exists())
    print(f"Subjects: {len(subjects)}", flush=True)

    out = {"experiment": "transfer_probe_64mt", "addresses": ["R2.2"],
           "method": "frozen OASIS encoder features -> LOO ridge on real 64mT (HFC)",
           "backbones": {}}

    # ViT3D
    print("Extracting ViT3D features...", flush=True)
    Xv, yv, _ = vit_features(subjects)
    # ridge alpha sweep (report best, note all)
    best = None
    for a in (0.1, 1.0, 10.0):
        mae, r, _ = loo_ridge(Xv, yv, a)
        if best is None or mae < best[0]:
            best = (mae, r, a)
    print(f"  ViT3D: MAE={best[0]:.4f} r={best[1]:.3f} (alpha={best[2]}, feat_dim={Xv.shape[1]})", flush=True)
    out["backbones"]["vit3d"] = {"mae": round(best[0],4), "r": round(best[1],4),
                                 "alpha": best[2], "feat_dim": int(Xv.shape[1])}

    # Swin (only if checkpoint exists)
    if SWIN_CKPT.exists():
        print("Extracting Swin-UNETR features...", flush=True)
        Xs, ys, _ = swin_features(subjects)
        best = None
        for a in (0.1, 1.0, 10.0):
            mae, r, _ = loo_ridge(Xs, ys, a)
            if best is None or mae < best[0]:
                best = (mae, r, a)
        print(f"  Swin:  MAE={best[0]:.4f} r={best[1]:.3f} (alpha={best[2]}, feat_dim={Xs.shape[1]})", flush=True)
        out["backbones"]["swin_unetr"] = {"mae": round(best[0],4), "r": round(best[1],4),
                                          "alpha": best[2], "feat_dim": int(Xs.shape[1])}
    else:
        print(f"  Swin checkpoint not found at {SWIN_CKPT} — run arch_comparator_swin.py first.", flush=True)
        out["backbones"]["swin_unetr"] = "PENDING — checkpoint not ready"

    # interpretation
    if "swin_unetr" in out["backbones"] and isinstance(out["backbones"]["swin_unetr"], dict):
        vm = out["backbones"]["vit3d"]["mae"]; sm = out["backbones"]["swin_unetr"]["mae"]
        if sm < vm:
            verdict = "Swin transfers BETTER to 64mT (higher-capacity features help on target)"
        elif abs(sm - vm) < 0.001:
            verdict = "Swin transfers COMPARABLY (capacity does not help on 64mT)"
        else:
            verdict = "Swin transfers WORSE to 64mT (OASIS advantage does NOT survive domain gap)"
        out["verdict"] = verdict
        print(f"\nVERDICT: {verdict}", flush=True)

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {OUT_DIR / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
