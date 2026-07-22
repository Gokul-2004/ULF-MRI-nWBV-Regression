"""
Zenodo Real Low-Field MRI Validation Script
============================================
Validates trained model on real 64mT Hyperfine scanner data.

Pipeline:
1. Load 3T T1w scans for 11 paired subjects → compute ground truth biomarkers
2. Load 64mT T1w scans for same subjects → run through trained model
3. Compare predicted vs ground truth → real-world validation metrics

Dataset: van den Broek et al. (2025), Zenodo DOI: 10.5281/zenodo.15374449
"""

import os
import sys
import json
import numpy as np
import nibabel as nib
import torch
from pathlib import Path
from scipy import ndimage
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Paths ────────────────────────────────────────────────────────────────────

ZENODO_BASE = PROJECT_ROOT / "Website_dat" / "extracted" / \
    "Paired 64mT and 3T Brain MRI Scans of Healthy Subjects for Neuroimaging Research v2" / "Data"

DATA_3T   = ZENODO_BASE / "3T data"
DATA_64MT = ZENODO_BASE / "64mT data"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "zenodo_validation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 11 paired subjects
PAIRED_SUBJECTS = [
    "sub-0011", "sub-0015", "sub-0023", "sub-0025", "sub-0027",
    "sub-0035", "sub-0046", "sub-0047", "sub-0048", "sub-0064", "sub-0066"
]

# ─── Biomarker extraction (intensity-based, no FreeSurfer needed) ──────────────

def load_nifti(path):
    """Load NIfTI file and return normalized numpy array."""
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    # Normalize to [0, 1]
    p1, p99 = np.percentile(data[data > 0], [1, 99]) if data.max() > 0 else (0, 1)
    data = np.clip(data, p1, p99)
    data = (data - p1) / (p99 - p1 + 1e-8)
    return data

def skull_strip_simple(volume):
    """Simple intensity-based skull stripping."""
    threshold = 0.1
    brain_mask = volume > threshold
    # Remove small isolated regions
    brain_mask = ndimage.binary_fill_holes(brain_mask)
    labeled, num = ndimage.label(brain_mask)
    if num > 0:
        sizes = ndimage.sum(brain_mask, labeled, range(1, num + 1))
        largest = np.argmax(sizes) + 1
        brain_mask = labeled == largest
    return brain_mask

def segment_tissues(volume, brain_mask):
    """Simple 3-class tissue segmentation using Otsu-like thresholds."""
    brain_voxels = volume[brain_mask]

    # Compute thresholds
    t1 = np.percentile(brain_voxels, 33)
    t2 = np.percentile(brain_voxels, 66)

    csf_mask   = brain_mask & (volume < t1)
    gm_mask    = brain_mask & (volume >= t1) & (volume < t2)
    wm_mask    = brain_mask & (volume >= t2)

    return csf_mask, gm_mask, wm_mask

def compute_biomarkers(volume, brain_mask, csf_mask, gm_mask, wm_mask):
    """Compute 4 biomarkers from segmented volume."""
    total_voxels = volume.size
    brain_count  = brain_mask.sum()
    csf_count    = csf_mask.sum()
    gm_count     = gm_mask.sum()
    wm_count     = wm_mask.sum()

    # BTF: Brain Tissue Fraction
    btf = brain_count / total_voxels if total_voxels > 0 else 0.0

    # VBR: Ventricle-to-Brain Ratio (using CSF as proxy for ventricles)
    vbr = csf_count / brain_count if brain_count > 0 else 0.0

    # TCR: Tissue Contrast Ratio (gray/white matter ratio)
    tcr = gm_count / wm_count if wm_count > 0 else 1.0

    # MCI: Mean Cortical Intensity
    mci = float(volume[gm_mask].mean()) if gm_mask.sum() > 0 else 0.0

    return {
        "btf": float(np.clip(btf, 0, 1)),
        "vbr": float(np.clip(vbr, 0, 1)),
        "tcr": float(np.clip(tcr / 3.0, 0, 1)),  # normalize same as training
        "mci": float(np.clip(mci, 0, 1))
    }

def extract_biomarkers_from_scan(nifti_path):
    """Full pipeline: load scan → segment → compute biomarkers."""
    volume     = load_nifti(nifti_path)
    brain_mask = skull_strip_simple(volume)
    csf, gm, wm = segment_tissues(volume, brain_mask)
    biomarkers = compute_biomarkers(volume, brain_mask, csf, gm, wm)
    return biomarkers, volume

# ─── Model inference ──────────────────────────────────────────────────────────

def load_best_model(config_path=None):
    """Load the best saved model checkpoint."""
    from models.baselines import BaselineViT3D, BaselineCNN3D, BaselineUNet3D
    import yaml

    config_path = config_path or PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    best_path = checkpoint_dir / "best_model.pt"

    if not best_path.exists():
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        if not checkpoints:
            print("No checkpoint found — will run intensity-based validation only.")
            return None, config
        best_path = sorted(checkpoints)[-1]

    print(f"Loading checkpoint: {best_path}")
    checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
    model_name = checkpoint.get('model_name', 'vit')
    print(f"Model type: {model_name}")

    model_map = {
        'vit': BaselineViT3D,
        'cnn': BaselineCNN3D,
        'unet': BaselineUNet3D
    }
    img_size = tuple(config.get('data', {}).get('image_size', [64, 64, 64]))
    num_classes = config.get('biomarkers', {}).get('num_classes', 4)
    ModelClass = model_map.get(model_name, BaselineViT3D)
    if ModelClass == BaselineViT3D:
        model = BaselineViT3D(img_size=img_size, patch_size=16,
                              num_classes=num_classes, embed_dim=256,
                              num_layers=4, num_heads=8)
    else:
        model = ModelClass(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config

def preprocess_for_model(volume, target_size=(64, 64, 64)):
    """Resize and prepare volume for model inference."""
    from scipy.ndimage import zoom
    current_shape = volume.shape
    zoom_factors = [target_size[i] / current_shape[i] for i in range(3)]
    resized = zoom(volume, zoom_factors, order=1)
    # Normalize
    resized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
    tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    return tensor

def predict_biomarkers(model, volume):
    """Run model inference on a volume."""
    tensor = preprocess_for_model(volume)
    with torch.no_grad():
        output = model(tensor)
    predictions = output.squeeze().numpy()
    biomarker_names = ["btf", "tcr", "vbr", "mci"]
    return {name: float(np.clip(predictions[i], 0, 1))
            for i, name in enumerate(biomarker_names)}

# ─── Main validation ──────────────────────────────────────────────────────────

def find_t1w_3t(subject_id):
    """Find high-resolution T1w scan for a 3T subject."""
    subject_dir = DATA_3T / subject_id / "anat"
    if not subject_dir.exists():
        return None
    # Prefer high-res T1w
    for pattern in [f"{subject_id}_acq-highres_T1w.nii.gz",
                    f"{subject_id}_T1w.nii.gz"]:
        path = subject_dir / pattern
        if path.exists():
            return path
    return None

def find_t1w_64mt(subject_id):
    """Find T1w scan for a 64mT subject."""
    subject_dir = DATA_64MT / subject_id
    if not subject_dir.exists():
        return None
    # Search in sessions
    for session in ["ses-01", "ses-02"]:
        anat_dir = subject_dir / session / "anat"
        if anat_dir.exists():
            for f in anat_dir.glob("*_T1w.nii.gz"):
                if "localizer" not in f.name:
                    return f
    return None

def run_validation():
    print("=" * 70)
    print("Zenodo Real Low-Field MRI Validation")
    print("=" * 70)

    # Step 1: Try to load trained model
    print("\n[1/4] Loading trained model...")
    model, config = load_best_model()
    use_model = model is not None

    # Step 2: Process all paired subjects
    print(f"\n[2/4] Processing {len(PAIRED_SUBJECTS)} paired subjects...")
    results = []

    for subject_id in PAIRED_SUBJECTS:
        print(f"\n  Processing {subject_id}...")

        # Find scans
        t1w_3t   = find_t1w_3t(subject_id)
        t1w_64mt = find_t1w_64mt(subject_id)

        if t1w_3t is None:
            print(f"    3T scan not found for {subject_id}")
            continue
        if t1w_64mt is None:
            print(f"    64mT scan not found for {subject_id}")
            continue

        print(f"    3T scan:   {t1w_3t.name}")
        print(f"    64mT scan: {t1w_64mt.name}")

        # Ground truth from 3T
        gt_biomarkers, vol_3t = extract_biomarkers_from_scan(t1w_3t)
        print(f"    Ground truth (3T): BTF={gt_biomarkers['btf']:.3f}, "
              f"VBR={gt_biomarkers['vbr']:.3f}, "
              f"TCR={gt_biomarkers['tcr']:.3f}, "
              f"MCI={gt_biomarkers['mci']:.3f}")

        # Intensity-based prediction from 64mT (baseline)
        intensity_biomarkers, vol_64mt = extract_biomarkers_from_scan(t1w_64mt)
        print(f"    Intensity pred (64mT): BTF={intensity_biomarkers['btf']:.3f}, "
              f"VBR={intensity_biomarkers['vbr']:.3f}, "
              f"TCR={intensity_biomarkers['tcr']:.3f}, "
              f"MCI={intensity_biomarkers['mci']:.3f}")

        result = {
            "subject": subject_id,
            "ground_truth": gt_biomarkers,
            "intensity_prediction": intensity_biomarkers,
        }

        # Model prediction from 64mT (if model available)
        if use_model:
            model_biomarkers = predict_biomarkers(model, vol_64mt)
            print(f"    Model pred (64mT):     BTF={model_biomarkers['btf']:.3f}, "
                  f"VBR={model_biomarkers['vbr']:.3f}, "
                  f"TCR={model_biomarkers['tcr']:.3f}, "
                  f"MCI={model_biomarkers['mci']:.3f}")
            result["model_prediction"] = model_biomarkers

        results.append(result)

    # Step 3: Compute metrics
    print(f"\n[3/4] Computing validation metrics for {len(results)} subjects...")

    biomarker_names = ["btf", "vbr", "tcr", "mci"]
    metrics = {"intensity_baseline": {}, "model": {}} if use_model else {"intensity_baseline": {}}

    for bm in biomarker_names:
        gt_vals = [r["ground_truth"][bm] for r in results]

        # Intensity baseline metrics
        int_vals = [r["intensity_prediction"][bm] for r in results]
        if len(gt_vals) > 1:
            corr, pval = pearsonr(gt_vals, int_vals)
            mae  = np.mean(np.abs(np.array(gt_vals) - np.array(int_vals)))
            rmse = np.sqrt(np.mean((np.array(gt_vals) - np.array(int_vals))**2))
            metrics["intensity_baseline"][bm] = {
                "correlation": round(corr, 4),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "p_value": round(pval, 4)
            }

        # Model metrics
        if use_model and "model_prediction" in results[0]:
            mod_vals = [r["model_prediction"][bm] for r in results]
            if len(gt_vals) > 1:
                corr, pval = pearsonr(gt_vals, mod_vals)
                mae  = np.mean(np.abs(np.array(gt_vals) - np.array(mod_vals)))
                rmse = np.sqrt(np.mean((np.array(gt_vals) - np.array(mod_vals))**2))
                metrics["model"][bm] = {
                    "correlation": round(corr, 4),
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                    "p_value": round(pval, 4)
                }

    # Step 4: Print and save report
    print(f"\n[4/4] Results Summary")
    print("=" * 70)
    print(f"\n{'Biomarker':<10} {'GT Mean':>10} {'Intensity Corr':>15} {'Intensity MAE':>14}")
    print("-" * 55)
    for bm in biomarker_names:
        gt_mean = np.mean([r["ground_truth"][bm] for r in results])
        int_corr = metrics["intensity_baseline"].get(bm, {}).get("correlation", "N/A")
        int_mae  = metrics["intensity_baseline"].get(bm, {}).get("mae", "N/A")
        print(f"{bm.upper():<10} {gt_mean:>10.4f} {str(int_corr):>15} {str(int_mae):>14}")

    if use_model and metrics.get("model"):
        print(f"\n{'Biomarker':<10} {'Model Corr':>12} {'Model MAE':>12} {'p-value':>10}")
        print("-" * 48)
        for bm in biomarker_names:
            mod_corr = metrics["model"].get(bm, {}).get("correlation", "N/A")
            mod_mae  = metrics["model"].get(bm, {}).get("mae", "N/A")
            pval     = metrics["model"].get(bm, {}).get("p_value", "N/A")
            print(f"{bm.upper():<10} {str(mod_corr):>12} {str(mod_mae):>12} {str(pval):>10}")

    # Save results
    output = {
        "dataset": "Zenodo Paired 64mT/3T (van den Broek et al., 2025)",
        "num_subjects": len(results),
        "paired_subjects": PAIRED_SUBJECTS,
        "metrics": metrics,
        "per_subject_results": results
    }

    results_path = RESULTS_DIR / "zenodo_validation_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Plain text report
    report_path = RESULTS_DIR / "zenodo_validation_report.txt"
    with open(report_path, "w") as f:
        f.write("Zenodo Real Low-Field MRI Validation Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Dataset: van den Broek et al. (2025), Zenodo\n")
        f.write(f"Subjects: {len(results)} paired 64mT/3T\n\n")
        f.write("Intensity Baseline (64mT direct segmentation):\n")
        for bm in biomarker_names:
            m = metrics["intensity_baseline"].get(bm, {})
            f.write(f"  {bm.upper()}: corr={m.get('correlation','N/A')}, "
                    f"MAE={m.get('mae','N/A')}\n")
        if use_model and metrics.get("model"):
            f.write("\nTrained Model (ViT, low-field input):\n")
            for bm in biomarker_names:
                m = metrics["model"].get(bm, {})
                f.write(f"  {bm.upper()}: corr={m.get('correlation','N/A')}, "
                        f"MAE={m.get('mae','N/A')}, p={m.get('p_value','N/A')}\n")

    print(f"Report saved to: {report_path}")
    return output

if __name__ == "__main__":
    run_validation()
