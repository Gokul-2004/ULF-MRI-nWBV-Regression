"""
LoRA Adapter Arm for Cross-Session LOOCV (Reviewer R2.4)
========================================================
Reviewer 2 asked for parameter-efficient adaptation methods beyond
head-only / LayerNorm+head / full fine-tune (LoRA, adapters, prompt tuning).
This adds a **LoRA** arm under the *identical* HFC->HFE cross-session LOOCV
protocol used in scripts/ablation_adapter_strategy.py, so it slots straight
into the existing adapter-strategy table as a fourth row.

LoRA placement note
-------------------
BaselineViT3D uses torch.nn.MultiheadAttention, whose forward reads
out_proj.weight/bias directly (it does NOT call out_proj as a submodule), so
wrapping the attention projections would have no effect. LoRA is therefore
injected into the two nn.Linear layers of each TransformerBlock MLP
(mlp[0]: 256->1024 and mlp[3]: 1024->256) — the standard, correct target for
this architecture. The regression head is also trained (as in every arm).

Everything else — data, augmentation (N=5), 80 epochs, AdamW, cosine schedule,
seed 42, bootstrap CIs — is identical to the published adapter ablation.

Requires: torch + nibabel + scipy + the ds006557 NIfTI data (data/ds006557_data)
and checkpoints/oasis_finetuned.pt. Run on the machine that has the dataset:

    python3 scripts/ablation_lora.py

Output: experiments/ablation_lora/results.json  (LoRA row + combined 4-way table)
"""

import sys
import json
import copy
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from scipy.ndimage import zoom as _zoom

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.baselines import BaselineViT3D

# ── Config (mirrors ablation_adapter_strategy.py) ───────────────────────────────
DS_DIR       = project_root / "data" / "ds006557_data"
GT_CSV       = project_root / "experiments" / "fastsurfer_output" / "nwbv_ground_truth.csv"
VIT_CKPT     = project_root / "checkpoints" / "oasis_finetuned.pt"
OUT_DIR      = project_root / "experiments" / "ablation_lora"
ADAPTER_JSON = project_root / "experiments" / "ablation_adapter" / "results.json"
TARGET_SHAPE = (64, 64, 64)
DEVICE       = torch.device("cpu")

EPOCHS       = 80
LR           = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 4
N_AUGMENTS   = 5
N_BOOT       = 10_000
SEED         = 42

LORA_R       = 4       # low-rank dimension
LORA_ALPHA   = 8       # scaling = alpha / r

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Data (identical to adapter ablation) ────────────────────────────────────────

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
    return _zoom(vol, factors, order=1).astype(np.float32)


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
    if not path.exists():
        return None
    return resize64(load_nifti_norm(path))


def load_subjects():
    import csv
    gt = {}
    with open(GT_CSV) as f:
        for row in csv.DictReader(f):
            gt[row["subject"]] = float(row["nWBV_freesurfer"])
    return sorted(s for s in gt if (DS_DIR / s).exists()), gt


# ── LoRA ────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank update:
       y = W0 x (+b0)  +  (alpha/r) * B(A(x)),  A,B trainable, W0 frozen."""

    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.scaling = alpha / r
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)          # start as identity (output == base)

    def forward(self, x):
        return self.base(x) + self.scaling * self.B(self.A(x))


def inject_lora(model, r, alpha):
    """Replace the two MLP Linears in every transformer block with LoRALinear."""
    for block in model.blocks:
        for idx in (0, 3):                     # mlp = [Linear, GELU, Dropout, Linear, Dropout]
            layer = block.mlp[idx]
            if isinstance(layer, nn.Linear):
                block.mlp[idx] = LoRALinear(layer, r, alpha)
    return model


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model():
    model = BaselineViT3D(img_size=TARGET_SHAPE, patch_size=16, num_classes=4,
                          embed_dim=256, num_layers=4, num_heads=8)
    model.head = nn.Linear(model.head.in_features, 1)
    ckpt = torch.load(VIT_CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def configure_lora(model):
    """Inject LoRA, freeze base, unfreeze LoRA A/B + regression head."""
    inject_lora(model, LORA_R, LORA_ALPHA)
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALinear):
            for p in m.A.parameters():
                p.requires_grad = True
            for p in m.B.parameters():
                p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Train / predict (identical schedule to adapter ablation) ────────────────────

def train_fold(base_model, train_subjects, gt):
    model = copy.deepcopy(base_model)
    n_trainable = configure_lora(model)
    model.to(DEVICE)

    X, y = [], []
    for subj in train_subjects:
        vol = load_scan(subj, "HFC")
        if vol is None:
            continue
        X.append(vol); y.append(gt[subj])
        for _ in range(N_AUGMENTS):
            X.append(augment(vol)); y.append(gt[subj])

    X_t = torch.tensor(np.array(X)).unsqueeze(1).float()
    y_t = torch.tensor(y).float().unsqueeze(1)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_loss  = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.train()
    for _ in range(EPOCHS):
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
        scheduler.step()
        avg = epoch_loss / max(nb, 1)
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, n_trainable


@torch.no_grad()
def predict(model, subject, session):
    vol = load_scan(subject, session)
    if vol is None:
        return None
    x = torch.tensor(vol).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    return float(model(x).item())


def bootstrap_mae_ci(abs_errors):
    rng = np.random.default_rng(SEED)
    means = [np.mean(rng.choice(abs_errors, size=len(abs_errors), replace=True))
             for _ in range(N_BOOT)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 62)
    print(f"LoRA Adapter Arm (cross-session LOOCV)  r={LORA_R} alpha={LORA_ALPHA}")
    print("=" * 62)

    subjects, gt = load_subjects()
    base_model   = build_model()

    records, n_trainable = [], None
    for i, held_out in enumerate(subjects):
        train_subjs = [s for s in subjects if s != held_out]
        model, n_trainable = train_fold(base_model, train_subjs, gt)
        pred = predict(model, held_out, "HFE")
        true = gt[held_out]
        if pred is None:
            continue
        err = abs(pred - true)
        records.append({"subject": held_out, "pred": round(pred, 4),
                        "true": round(true, 4), "abs_error": round(err, 4)})
        print(f"  [{i+1:2d}/23] {held_out}  err={err:.4f}")

    abs_errs = np.array([r["abs_error"] for r in records])
    mae   = float(np.mean(abs_errs))
    mstd  = float(np.std(abs_errs))
    ci_lo, ci_hi = bootstrap_mae_ci(abs_errs)
    bias  = float(np.mean([r["pred"] - r["true"] for r in records]))

    lora_row = {
        "strategy":         "lora",
        "lora_rank":        LORA_R,
        "lora_alpha":       LORA_ALPHA,
        "lora_targets":     "TransformerBlock MLP linears (mlp[0], mlp[3]) + head",
        "trainable_params": n_trainable,
        "mae":              round(mae, 4),
        "mae_std":          round(mstd, 4),
        "mae_ci_95":        [round(ci_lo, 4), round(ci_hi, 4)],
        "bias":             round(bias, 4),
        "n_subjects":       len(records),
        "per_subject":      records,
    }

    # Combine with the existing 3-strategy ablation for a 4-row table
    combined = {"lora": lora_row}
    if ADAPTER_JSON.exists():
        prev = json.load(open(ADAPTER_JSON)).get("results", {})
        for k, v in prev.items():
            combined[k] = {kk: v[kk] for kk in
                           ("strategy", "trainable_params", "mae", "mae_std",
                            "mae_ci_95", "bias") if kk in v}

    print("\n" + "=" * 62)
    print("ADAPTER ABLATION incl. LoRA")
    print(f"{'Strategy':<12}{'Params':>12}{'MAE':>9}{'95% CI':>20}{'Bias':>9}")
    print("-" * 62)
    order = ["head_only", "ln_head", "lora", "full_ft"]
    for k in order:
        if k in combined:
            r = combined[k]
            ci = f"[{r['mae_ci_95'][0]:.4f}-{r['mae_ci_95'][1]:.4f}]"
            print(f"  {k:<10}{r['trainable_params']:>12,}{r['mae']:>9.4f}"
                  f"{ci:>20}{r['bias']:>+9.4f}")

    out = {
        "experiment":  "ablation_lora",
        "addresses":   ["R2.4"],
        "protocol":    "cross_session_loocv",
        "description": "LoRA arm added to head_only/ln_head/full_ft adapter ablation",
        "lora":        lora_row,
        "combined_table": combined,
    }
    json.dump(out, open(OUT_DIR / "results.json", "w"), indent=2)
    print(f"\nSaved -> {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
