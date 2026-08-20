"""
Zenodo External Validation (Reviewer R1.3) — Steps 3 & 4 of Branch B
====================================================================
Apples-to-apples external validation on the van den Broek Zenodo cohort:
  - Ground truth: FastSurfer nWBV = BrainSegVol / MaskVol, the SAME definition
    and pipeline as the paper's OASIS/ds006557 labels (run_fastsurfer_zenodo.sh).
  - Prediction: run the trained ViT3D on the paired 64 mT T2w scans, exactly the
    preprocessing used for ds006557 (percentile-norm -> resize 64^3).

Two checkpoints evaluated (no external adapter exists, so honestly report both):
  - real64mt_finetuned.pt : the ds006557-adapted model (full pipeline)
  - oasis_finetuned.pt     : unadapted reference

    python scripts/zenodo_external_validation.py

Output: experiments/zenodo_external_validation/results.json
"""

import sys, json, csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from scipy.ndimage import zoom
from scipy.stats import pearsonr, spearmanr

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import warnings; warnings.filterwarnings("ignore")
from models.baselines import BaselineViT3D

DEVICE = torch.device("cpu")
TARGET = (64, 64, 64)
ZEN64 = project_root / "Website_dat" / "extracted" / \
    "Paired 64mT and 3T Brain MRI Scans of Healthy Subjects for Neuroimaging Research v2" / \
    "Data" / "64mT data"
FS_DIR = project_root / "experiments" / "fastsurfer_zenodo"
OUT_DIR = project_root / "experiments" / "zenodo_external_validation"
CKPTS = {"real64mt_adapted": project_root / "checkpoints" / "real64mt_finetuned.pt",
         "oasis_unadapted":  project_root / "checkpoints" / "oasis_finetuned.pt"}


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


def measure(content, name):
    import re
    m = re.search(rf'# Measure {name},\s*\w+,\s*[^,]+,\s*([\d.]+)', content)
    return float(m.group(1)) if m else None


def fastsurfer_nwbv(subj):
    f = FS_DIR / subj / subj / "stats" / "aseg+DKT.stats"
    if not f.exists():
        return None
    c = f.read_text()
    mask, brain = measure(c, "Mask"), measure(c, "BrainSeg")
    return brain / mask if (mask and brain and mask > 0) else None


def load_64mt_t2w(subj):
    """The 64mT T2w scan (matches ds006557 acq-axi_T2w modality)."""
    anat = ZEN64 / subj / "ses-01" / "anat"
    for f in sorted(anat.glob(f"{subj}_ses-01_run-*_T2w.nii.gz")):
        if "localizer" not in f.name:
            return resize64(load_nifti_norm(f))
    return None


def build_model(ckpt):
    m = BaselineViT3D(img_size=TARGET, patch_size=16, num_classes=4,
                      embed_dim=256, num_layers=4, num_heads=8)
    m.head = nn.Linear(m.head.in_features, 1)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False)["model_state_dict"])
    m.eval()
    return m


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # gather subjects that have BOTH FastSurfer GT and a 64mT T2w
    subjects = []
    for d in sorted(FS_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith("sub-"):
            continue
        gt = fastsurfer_nwbv(d.name)
        vol = load_64mt_t2w(d.name)
        if gt is not None and vol is not None:
            subjects.append((d.name, gt, vol))
    print(f"Paired subjects with GT + 64mT: {len(subjects)}")
    if len(subjects) < 3:
        sys.exit("Too few paired subjects — segmentation may still be running.")

    trues = np.array([s[1] for s in subjects])
    X = torch.tensor(np.stack([s[2] for s in subjects])).unsqueeze(1).float()

    results = {"experiment": "zenodo_external_validation", "addresses": ["R1.3"],
               "dataset": "van den Broek et al. (2025) Zenodo, paired 64mT/3T",
               "ground_truth": "FastSurfer nWBV = BrainSeg/Mask (same as OASIS/ds006557)",
               "n_subjects": len(subjects),
               "subjects": [s[0] for s in subjects],
               "gt_nwbv": [round(float(t), 4) for t in trues],
               "models": {}}

    for name, ckpt in CKPTS.items():
        if not ckpt.exists():
            print(f"  {name}: checkpoint missing, skip"); continue
        m = build_model(ckpt)
        with torch.no_grad():
            preds = m(X).cpu().numpy().flatten()
        mae = float(np.mean(np.abs(preds - trues)))
        bias = float(np.mean(preds - trues))
        r, pr = pearsonr(trues, preds) if len(trues) > 2 else (0, 1)
        rho, prho = spearmanr(trues, preds) if len(trues) > 2 else (0, 1)
        results["models"][name] = {
            "checkpoint": ckpt.name, "mae": round(mae, 4), "bias": round(bias, 4),
            "pearson_r": round(float(r), 4), "pearson_p": round(float(pr), 4),
            "spearman_rho": round(float(rho), 4), "spearman_p": round(float(prho), 4),
            "preds": [round(float(p), 4) for p in preds]}
        print(f"  {name:20s}: MAE={mae:.4f}  r={r:.3f} (p={pr:.3f})  rho={rho:.3f}  bias={bias:+.4f}")

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {OUT_DIR / 'results.json'}")
    print("\n=== R1.3 EXTERNAL VALIDATION (FastSurfer GT, apples-to-apples) ===")
    print(f"  n={len(subjects)} paired 64mT/3T subjects, external cohort")
    for name, r in results["models"].items():
        print(f"  {name}: MAE={r['mae']}  Pearson r={r['pearson_r']} (p={r['pearson_p']})")


if __name__ == "__main__":
    main()
