"""
Ablation: Arnold et al. 2021 Gaussian Blur Baseline vs Physics-Informed Simulation
====================================================================================
Trains the same ViT architecture using Gaussian-blur-only low-field simulation
(Arnold et al. 2021 approach) and compares against our physics-informed pipeline.

This ablation answers the question:
  "Does the physics-informed simulation improve biomarker inference, or is
   a simple Gaussian blur just as good?"

Expected result: Physics-informed ViT > Arnold-blur ViT, showing that
T1/T2 relaxation modeling and Rician noise matter for downstream inference.

Outputs:
  - experiments/ablation_arnold/results.json    — per-biomarker metrics
  - experiments/ablation_arnold/report.txt      — comparison table for paper

Usage:
    python3 scripts/ablation_arnold.py
"""

import sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter, zoom
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.field_conversion import FieldConverter
from utils.biomarker_extraction import extract_biomarkers_for_dataset
from models.baselines import BaselineViT3D

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR     = project_root / "experiments" / "ablation_arnold"
CKPT_DIR    = project_root / "checkpoints"
HF_DIR      = project_root / "data" / "high_field"
CFG_PATH    = project_root / "configs" / "config.yaml"

TARGET_SHAPE  = (64, 64, 64)
BATCH_SIZE    = 4
LR            = 5e-4
EPOCHS        = 30
PATIENCE      = 8
AUGMENT_N     = 5       # augmented copies per volume
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Arnold simulation ─────────────────────────────────────────────────────────

def arnold_lowfield(vol: np.ndarray) -> np.ndarray:
    """
    Arnold et al. 2021 Gaussian blur + noise simulation.
    No physics: no T1/T2 relaxation, no Rician noise model.
    Simulates reduced resolution via Gaussian blur then noise injection.
    """
    # In-plane degradation: 1mm → 1.6mm → sigma ≈ 0.68 voxels
    # Through-plane: 1mm → 5mm → sigma ≈ 2.12 voxels (after resize to 64^3, proportional)
    blurred = gaussian_filter(vol, sigma=(0.68, 0.68, 2.12))
    # Contrast reduction (Hyperfine has ~30% lower SNR-equivalent contrast)
    blurred = blurred * 0.7
    # Additive Gaussian noise (NOT Rician — this is the key difference)
    noise = np.random.normal(0, 0.04, blurred.shape).astype(np.float32)
    return np.clip(blurred + noise, 0, 1).astype(np.float32)


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment(vol: np.ndarray) -> np.ndarray:
    """Random flip + mild intensity jitter."""
    for axis in range(3):
        if np.random.rand() > 0.5:
            vol = np.flip(vol, axis=axis).copy()
    vol = vol * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05)
    return np.clip(vol, 0, 1).astype(np.float32)


# ── Dataset (loads high-field, applies simulation on-the-fly) ─────────────────

class AblationDataset(Dataset):
    def __init__(self, pairs: list, method: str, is_train: bool = True):
        """
        pairs: list of {"name": str, "hf_path": Path, "label": np.ndarray}
        method: "physics" | "arnold"
        """
        self.samples = []
        converter = FieldConverter({})

        for pair in pairs:
            hf = np.load(pair["hf_path"]).astype(np.float32)
            vmax = hf.max()
            if vmax > 0:
                hf = hf / vmax

            # Resize to model input size
            if hf.shape != TARGET_SHAPE:
                factors = [t / s for t, s in zip(TARGET_SHAPE, hf.shape)]
                hf = zoom(hf, factors, order=1).astype(np.float32)

            for aug_i in range(AUGMENT_N if is_train else 1):
                vol = augment(hf) if (is_train and aug_i > 0) else hf
                if method == "physics":
                    lf = converter.convert(vol, method='hyperfine')
                else:
                    lf = arnold_lowfield(vol)
                self.samples.append({
                    "lf":    torch.from_numpy(lf).unsqueeze(0).float(),
                    "label": torch.from_numpy(pair["label"]).float(),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["lf"], s["label"]


# ── Training ──────────────────────────────────────────────────────────────────

def train_vit(pairs_train, pairs_val, method: str, tag: str):
    print(f"\n{'='*60}")
    print(f"Training ViT — {tag} ({method} simulation)")
    print(f"{'='*60}")

    train_ds = AblationDataset(pairs_train, method=method, is_train=True)
    val_ds   = AblationDataset(pairs_val,   method=method, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = BaselineViT3D(
        img_size=TARGET_SHAPE, patch_size=16, num_classes=4,
        embed_dim=256, num_layers=4, num_heads=8
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_count = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                val_loss += criterion(model(x.to(DEVICE)), y.to(DEVICE)).item()

        train_loss /= max(len(train_loader), 1)
        val_loss   /= max(len(val_loader), 1)
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}/{EPOCHS} | train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    print(f"  Best val_loss: {best_val_loss:.4f}")
    return model, best_val_loss


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, pairs_test, method: str) -> dict:
    test_ds = AblationDataset(pairs_test, method=method, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            all_preds.append(model(x.to(DEVICE)).cpu().numpy())
            all_labels.append(y.numpy())

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    biomarker_names = ["BTF", "TCR", "VBR", "MCI"]

    from scipy.stats import pearsonr
    overall_mse = float(np.mean((preds - labels) ** 2))
    overall_mae = float(np.mean(np.abs(preds - labels)))

    per_bm = {}
    for i, name in enumerate(biomarker_names):
        p, l = preds[:, i], labels[:, i]
        mse = float(np.mean((p - l) ** 2))
        mae = float(np.mean(np.abs(p - l)))
        r2  = float(1 - np.sum((p - l)**2) / (np.sum((l - l.mean())**2) + 1e-8))
        try:
            r_val, _ = pearsonr(l, p)
        except Exception:
            r_val = 0.0
        per_bm[name] = {"mse": round(mse, 5), "mae": round(mae, 4),
                        "r2": round(r2, 4), "pearson_r": round(float(r_val), 4)}

    return {
        "overall_mse": round(overall_mse, 5),
        "overall_mae": round(overall_mae, 4),
        "per_biomarker": per_bm,
        "n_test": len(preds),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load biomarker labels
    print("Loading biomarker labels...")
    biomarker_labels = extract_biomarkers_for_dataset(str(HF_DIR))
    print(f"  {len(biomarker_labels)} volumes with labels")

    # Build pairs list
    pairs = []
    for name, label in biomarker_labels.items():
        hf_path = HF_DIR / f"{name}.npy"
        if hf_path.exists():
            pairs.append({
                "name": name,
                "hf_path": hf_path,
                "label": np.array(label, dtype=np.float32),
            })

    print(f"  {len(pairs)} usable pairs")
    if len(pairs) < 5:
        print("Too few pairs for ablation. Aborting.")
        return

    # Split 70/15/15
    np.random.seed(42)
    idx = np.random.permutation(len(pairs))
    n_train = int(0.70 * len(pairs))
    n_val   = int(0.15 * len(pairs))
    pairs_train = [pairs[i] for i in idx[:n_train]]
    pairs_val   = [pairs[i] for i in idx[n_train:n_train+n_val]]
    pairs_test  = [pairs[i] for i in idx[n_train+n_val:]]
    print(f"  Split: train={len(pairs_train)}, val={len(pairs_val)}, test={len(pairs_test)}")

    results = {}

    # Train physics-informed ViT
    phys_model, phys_val_loss = train_vit(pairs_train, pairs_val, "physics", "Physics-Informed")
    results["physics"] = evaluate(phys_model, pairs_test, "physics")
    results["physics"]["val_loss"] = round(phys_val_loss, 5)

    # Train Arnold ViT
    arnold_model, arnold_val_loss = train_vit(pairs_train, pairs_val, "arnold", "Arnold Gaussian")
    results["arnold"] = evaluate(arnold_model, pairs_test, "arnold")
    results["arnold"]["val_loss"] = round(arnold_val_loss, 5)

    # Save results
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    lines = []
    lines.append("=" * 75)
    lines.append("Ablation: Physics-Informed Sim vs Arnold Gaussian Blur")
    lines.append(f"n_test = {results['physics']['n_test']}  |  ↑ higher is better, ↓ lower is better")
    lines.append("=" * 75)
    lines.append(f"{'Method':<28} {'MSE↓':>8} {'MAE↓':>8} {'BTF-R2↑':>10} {'BTF-r↑':>8}")
    lines.append("-" * 65)
    for method, label in [("physics", "Ours (Physics-Informed)"),
                           ("arnold",  "Arnold et al. (Gaussian)")]:
        r = results[method]
        btf = r["per_biomarker"]["BTF"]
        lines.append(
            f"{label:<28} {r['overall_mse']:>8.5f} {r['overall_mae']:>8.4f} "
            f"{btf['r2']:>10.4f} {btf['pearson_r']:>8.4f}"
        )
    lines.append("=" * 75)
    lines.append("\nPer-Biomarker R2 (Physics vs Arnold):")
    lines.append(f"{'Biomarker':<8} {'Phys R2':>10} {'Arnold R2':>12} {'Phys r':>10} {'Arnold r':>10}")
    lines.append("-" * 55)
    for bm in ["BTF", "TCR", "VBR", "MCI"]:
        pr  = results["physics"]["per_biomarker"][bm]
        ar  = results["arnold"]["per_biomarker"][bm]
        lines.append(
            f"{bm:<8} {pr['r2']:>10.4f} {ar['r2']:>12.4f} "
            f"{pr['pearson_r']:>10.4f} {ar['pearson_r']:>10.4f}"
        )
    lines.append("=" * 75)

    report = "\n".join(lines)
    print("\n" + report)
    with open(OUT_DIR / "report.txt", "w") as f:
        f.write(report + "\n")

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
