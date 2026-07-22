"""
Pseudo-Label Ablation
======================
Compares two adapter training label sources on the fixed 15-train / 8-test split:

  (A) SynthSeg+ pseudo-labels  — realistic deployment scenario
  (B) FastSurfer GT            — upper-bound / this paper's primary protocol

Outputs:
  1. Agreement between SynthSeg+ and FastSurfer GT on all 23 subjects
     (Pearson r, MAE, Bland-Altman bias)
  2. Adapter test MAE (8 subjects) under each label source
  3. Conclusion: how much does label source affect final MAE?

This directly addresses the reviewer concern:
  "Both pseudo-labels and GT come from FreeSurfer-family pipelines."

Output: experiments/pseudolabel_ablation/results.json
"""

import sys
import json
import copy
import csv
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
SYNTHSEG_CSV = project_root / "experiments" / "synthseg_output" / "nwbv_synthseg.csv"
VIT_CKPT     = project_root / "checkpoints" / "oasis_finetuned.pt"
OUT_DIR      = project_root / "experiments" / "pseudolabel_ablation"
TARGET_SHAPE = (64, 64, 64)
DEVICE       = torch.device("cpu")

# Fixed split matching original paper (seed=42, 15 train / 8 test)
TRAIN_N      = 15
SEED         = 42
EPOCHS       = 80
LR           = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 4
N_AUGMENTS   = 5
N_BOOT       = 10_000

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Data helpers ───────────────────────────────────────────────────────────────

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


def load_scan(subject: str, session: str = "HFC") -> np.ndarray | None:
    path = (DS_DIR / subject / f"ses-{session}" / "anat" /
            f"{subject}_ses-{session}_acq-axi_T2w.nii.gz")
    if not path.exists():
        return None
    return resize64(load_nifti_norm(path))


def load_csv_labels(csv_path: Path, value_col: str) -> dict[str, float]:
    labels = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            labels[row["subject"]] = float(row[value_col])
    return labels


def load_all_subjects() -> list[str]:
    gt = load_csv_labels(GT_CSV, "nWBV_freesurfer")
    return sorted(s for s in gt if (DS_DIR / s).exists())


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model() -> BaselineViT3D:
    model = BaselineViT3D(
        img_size=TARGET_SHAPE, patch_size=16, num_classes=4,
        embed_dim=256, num_layers=4, num_heads=8,
    )
    model.head = nn.Linear(model.head.in_features, 1)
    ckpt = torch.load(VIT_CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def freeze_to_adapter(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.norm.parameters():
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True


# ── Training & eval ────────────────────────────────────────────────────────────

def train_adapter(
    base_model: nn.Module,
    train_subjects: list[str],
    labels: dict[str, float],
    label_name: str,
) -> nn.Module:
    model = copy.deepcopy(base_model)
    freeze_to_adapter(model)
    model.to(DEVICE)

    X, y = [], []
    for subj in train_subjects:
        vol = load_scan(subj, "HFC")
        if vol is None:
            print(f"    WARNING: HFC missing for {subj}")
            continue
        nwbv = labels[subj]
        X.append(vol)
        y.append(nwbv)
        for _ in range(N_AUGMENTS):
            X.append(augment(vol))
            y.append(nwbv)

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
        epoch_loss, n_b = 0.0, 0
        for i in range(0, len(X_t), BATCH_SIZE):
            xb = X_t[i:i+BATCH_SIZE].to(DEVICE)
            yb = y_t[i:i+BATCH_SIZE].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1
        avg = epoch_loss / max(n_b, 1)
        scheduler.step()
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 20 == 0:
            print(f"    [{label_name}] Epoch {epoch+1:3d}/{EPOCHS}  loss={avg:.6f}")

    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: nn.Module, test_subjects: list[str], gt: dict[str, float]) -> list[dict]:
    records = []
    for subj in test_subjects:
        vol = load_scan(subj, "HFC")
        if vol is None:
            continue
        x    = torch.tensor(vol).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        pred = float(model(x).item())
        true = gt[subj]
        records.append({
            "subject":    subj,
            "pred":       round(pred, 4),
            "true_nwbv":  round(true, 4),
            "abs_error":  round(abs(pred - true), 4),
        })
    return records


def bootstrap_mae_ci(records: list[dict], n_boot: int = N_BOOT) -> tuple[float, float]:
    errs = np.array([r["abs_error"] for r in records])
    rng  = np.random.default_rng(SEED)
    means = [np.mean(rng.choice(errs, size=len(errs), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("Pseudo-Label Ablation: SynthSeg+ vs FastSurfer GT")
    print("=" * 62)

    subjects     = load_all_subjects()
    gt_labels    = load_csv_labels(GT_CSV, "nWBV_freesurfer")
    synth_labels = load_csv_labels(SYNTHSEG_CSV, "nWBV_synthseg")

    # ── Part 1: label agreement ────────────────────────────────────────────────
    print("\n--- Label Agreement (all 23 subjects) ---")
    common = [s for s in subjects if s in synth_labels]
    gt_vals    = np.array([gt_labels[s]    for s in common])
    synth_vals = np.array([synth_labels[s] for s in common])

    r_agree, p_agree = stats.pearsonr(synth_vals, gt_vals)
    mae_agree        = float(np.mean(np.abs(synth_vals - gt_vals)))
    bias_agree       = float(np.mean(synth_vals - gt_vals))

    print(f"  n={len(common)}  Pearson r={r_agree:.4f}  "
          f"MAE={mae_agree:.4f}  Bias={bias_agree:+.4f}")
    for s in common:
        diff = synth_labels[s] - gt_labels[s]
        print(f"  {s}  SynthSeg+={synth_labels[s]:.4f}  "
              f"FastSurfer={gt_labels[s]:.4f}  diff={diff:+.4f}")

    # ── Part 2: fixed 15/8 split ───────────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(subjects))
    train_subjects = [subjects[i] for i in idx[:TRAIN_N]]
    test_subjects  = [subjects[i] for i in idx[TRAIN_N:]]

    print(f"\nTrain: {len(train_subjects)} subjects  |  Test: {len(test_subjects)} subjects")

    base_model = build_model()

    results_by_label = {}
    for label_name, labels in [("FastSurfer_GT", gt_labels),
                                ("SynthSeg_pseudo", synth_labels)]:
        print(f"\n--- Training adapter with {label_name} ---")
        # Only use training subjects that have this label source
        valid_train = [s for s in train_subjects if s in labels]
        model       = train_adapter(base_model, valid_train, labels, label_name)
        records     = evaluate(model, test_subjects, gt_labels)  # always eval vs GT

        mae       = float(np.mean([r["abs_error"] for r in records]))
        ci_lo, ci_hi = bootstrap_mae_ci(records)
        bias      = float(np.mean([r["pred"] - r["true_nwbv"] for r in records]))

        print(f"  Test MAE = {mae:.4f} [95% CI: {ci_lo:.4f}–{ci_hi:.4f}]  "
              f"Bias = {bias:+.4f}")

        results_by_label[label_name] = {
            "n_train": len(valid_train),
            "n_test":  len(records),
            "mae":     round(mae, 4),
            "mae_ci":  [round(ci_lo, 4), round(ci_hi, 4)],
            "bias":    round(bias, 4),
            "per_subject": records,
        }

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("PSEUDO-LABEL ABLATION SUMMARY")
    print(f"{'='*62}")
    print(f"Label agreement (SynthSeg+ vs FastSurfer GT):")
    print(f"  Pearson r = {r_agree:.4f}  MAE = {mae_agree:.4f}  Bias = {bias_agree:+.4f}")
    print(f"\nAdapter test MAE (n={len(test_subjects)}):")
    for k, v in results_by_label.items():
        print(f"  {k:20s}  MAE={v['mae']:.4f}  CI=[{v['mae_ci'][0]:.4f}–{v['mae_ci'][1]:.4f}]")

    mae_penalty = (results_by_label["SynthSeg_pseudo"]["mae"] -
                   results_by_label["FastSurfer_GT"]["mae"])
    print(f"\nPseudo-label MAE penalty: {mae_penalty:+.4f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    results = {
        "experiment":    "pseudolabel_ablation",
        "description":   "SynthSeg+ pseudo-labels vs FastSurfer GT for adapter training",
        "split":         {"train_n": TRAIN_N, "test_n": len(test_subjects), "seed": SEED},
        "label_agreement": {
            "n":          len(common),
            "pearson_r":  round(r_agree, 4),
            "pearson_p":  round(p_agree, 4),
            "mae":        round(mae_agree, 4),
            "bias":       round(bias_agree, 4),
            "per_subject": [
                {"subject": s,
                 "synthseg": round(synth_labels[s], 4),
                 "fastsurfer": round(gt_labels[s], 4),
                 "diff": round(synth_labels[s] - gt_labels[s], 4)}
                for s in common
            ],
        },
        "adapter_results":   results_by_label,
        "mae_penalty":       round(mae_penalty, 4),
        "conclusion": (
            "Pseudo-label penalty is negligible" if abs(mae_penalty) < 0.005
            else "Pseudo-label degrades performance — use GT where available"
        ),
    }
    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
