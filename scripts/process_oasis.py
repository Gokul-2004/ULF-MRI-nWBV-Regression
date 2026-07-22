"""
OASIS-1 Processing and Validation Pipeline

Steps:
  1. Extract downloaded tar.gz discs
  2. Load T1w DICOM/NIfTI scans per subject
  3. Simulate low-field (Hyperfine 64mT) using improved physics model
  4. Run ViT model to predict biomarkers
  5. Compare against CSV ground truth (nWBV=BTF, CDR, age)
  6. Generate validation report

Ground truth from CSV:
  - nWBV: normalized whole brain volume (= our BTF)
  - eTIV: estimated total intracranial volume
  - CDR:  Clinical Dementia Rating (0=normal, 0.5=very mild, 1=mild, 2=moderate)
  - Age:  subject age
"""

import sys, os, json, tarfile, glob, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import zoom
from scipy.stats import pearsonr, spearmanr

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.field_conversion import FieldConverter
from models.baselines import BaselineViT3D

# ── Paths ──────────────────────────────────────────────────────────────────────
OASIS_RAW_DIR   = project_root / "data" / "oasis_raw"
OASIS_WORK_DIR  = project_root / "data" / "oasis_processed"
CHECKPOINT_PATH = project_root / "checkpoints" / "best_model.pt"
RESULTS_DIR     = project_root / "experiments" / "oasis_validation"
CSV_PATH        = OASIS_RAW_DIR / "oasis_cross-sectional.xlsx"

TARGET_SHAPE    = (64, 64, 64)   # model input size
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Step 1: Extract disc archives ──────────────────────────────────────────────

def extract_discs():
    """Extract all downloaded tar.gz discs."""
    OASIS_WORK_DIR.mkdir(parents=True, exist_ok=True)
    discs = sorted(OASIS_RAW_DIR.glob("oasis_cross-sectional_disc*.tar.gz"))
    if not discs:
        print("No disc files found. Check downloads in:", OASIS_RAW_DIR)
        return 0

    n_extracted = 0
    for disc in discs:
        print(f"Extracting {disc.name} ...")
        try:
            with tarfile.open(disc, "r:gz") as tf:
                tf.extractall(OASIS_WORK_DIR)
            n_extracted += 1
        except Exception as e:
            print(f"  Warning: {e}")
    print(f"Extracted {n_extracted}/{len(discs)} discs to {OASIS_WORK_DIR}")
    return n_extracted


# ── Step 2: Find T1w scans ─────────────────────────────────────────────────────

def find_t1w_scans():
    """
    Find all T1w NIfTI files in extracted OASIS data.
    OASIS directory structure: disc_X/OAS1_XXXX_MR1/PROCESSED/MPRAGE/T88_111/
    Returns dict: {subject_id: nifti_path}
    """
    scans = {}

    # OASIS NIfTI files are typically named: OAS1_XXXX_MR1_mpr_n4_anon_111_t88_gfc.img
    # or similar. They may also be .nii or .nii.gz
    patterns = [
        str(OASIS_WORK_DIR / "**" / "*.nii"),
        str(OASIS_WORK_DIR / "**" / "*.nii.gz"),
        str(OASIS_WORK_DIR / "**" / "*.img"),
    ]

    for pat in patterns:
        for fp in glob.glob(pat, recursive=True):
            p = Path(fp)
            # Extract subject ID from path (OAS1_XXXX_MR1)
            parts = p.parts
            for part in parts:
                if part.startswith("OAS1_") and "_MR" in part:
                    subj_id = part.split("_")[0] + "_" + part.split("_")[1] + "_" + part.split("_")[2]
                    # Prefer the averaged/processed file (contains 'avg' or 'gfc')
                    if subj_id not in scans:
                        scans[subj_id] = p
                    elif "gfc" in p.name.lower() or "avg" in p.name.lower():
                        scans[subj_id] = p   # prefer processed
                    break

    print(f"Found {len(scans)} subjects with T1w scans")
    return scans


# ── Step 3: Load and preprocess a scan ────────────────────────────────────────

def load_and_preprocess(nifti_path: Path, target_shape=TARGET_SHAPE) -> np.ndarray:
    """Load NIfTI, normalize, resize to target shape."""
    img  = nib.load(str(nifti_path))
    data = img.get_fdata(dtype=np.float32)

    # Take the first volume if 4D
    if data.ndim == 4:
        data = data[..., 0]

    # Normalize to [0, 1]
    vmin, vmax = data.min(), data.max()
    if vmax > vmin:
        data = (data - vmin) / (vmax - vmin)

    # Resize to target shape
    if data.shape != target_shape:
        zoom_f = [t / s for t, s in zip(target_shape, data.shape)]
        data   = zoom(data, zoom_f, order=1)

    return data.astype(np.float32)


# ── Step 4: Simulate low-field ─────────────────────────────────────────────────

def simulate_low_field(volume: np.ndarray, converter: FieldConverter) -> np.ndarray:
    return converter.convert(volume, method='hyperfine')


# ── Step 5: Run model inference ────────────────────────────────────────────────

def load_model():
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model = BaselineViT3D(
        img_size=TARGET_SHAPE, patch_size=16, num_classes=4,
        embed_dim=256, num_layers=4, num_heads=8
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE).eval()
    print(f"Model loaded from {CHECKPOINT_PATH} (epoch {ckpt.get('epoch','?')})")
    return model


def predict(model, volume: np.ndarray) -> np.ndarray:
    """Run model on a single volume, return [BTF, VBR, TCR, MCI]."""
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,D,H,W)
    with torch.no_grad():
        pred = model(tensor).cpu().numpy().squeeze()
    return pred


# ── Step 6: Validation metrics ────────────────────────────────────────────────

def compute_metrics(gt: np.ndarray, pred: np.ndarray, name: str) -> dict:
    """Pearson r, Spearman r, MAE, RMSE between arrays."""
    mask = ~(np.isnan(gt) | np.isnan(pred))
    gt, pred = gt[mask], pred[mask]
    if len(gt) < 3:
        return {}
    r_p, p_p   = pearsonr(gt, pred)
    r_s, p_s   = spearmanr(gt, pred)
    mae        = float(np.mean(np.abs(gt - pred)))
    rmse       = float(np.sqrt(np.mean((gt - pred)**2)))
    return {
        "n": int(len(gt)),
        "pearson_r": round(float(r_p), 3),
        "pearson_p": round(float(p_p), 4),
        "spearman_r": round(float(r_s), 3),
        "spearman_p": round(float(p_s), 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("OASIS-1 Validation Pipeline")
    print("=" * 70)

    # Load clinical labels
    df = pd.read_excel(CSV_PATH)
    df['subj_id'] = df['ID'].str.extract(r'(OAS1_\d{4}_MR\d)')
    df = df.dropna(subset=['nWBV']).set_index('subj_id')
    print(f"Clinical CSV: {len(df)} subjects with nWBV labels")
    print(f"  CDR distribution: {df['CDR'].value_counts().sort_index().to_dict()}")

    # Extract discs if not done yet
    already_extracted = list(OASIS_WORK_DIR.glob("OAS1_*")) if OASIS_WORK_DIR.exists() else []
    if not already_extracted:
        n = extract_discs()
        if n == 0:
            print("\nNo discs extracted. Waiting for downloads...")
            print(f"Check: tail -f {OASIS_RAW_DIR}/download_log.txt")
            return
    else:
        print(f"Using already extracted data: {len(already_extracted)} subject folders")

    # Find T1w scans
    scans = find_t1w_scans()
    if not scans:
        print("No T1w scans found after extraction. Check disc format.")
        return

    # Match scans to CSV subjects
    common = sorted(set(scans.keys()) & set(df.index))
    print(f"\nMatched {len(common)} subjects (scan + CSV labels)")

    # Load model
    model     = load_model()
    converter = FieldConverter({})

    # Per-subject inference
    results = []
    gt_nwbv, pred_btf = [], []
    gt_cdr, subj_ages = [], []

    for subj_id in tqdm(common, desc="Processing subjects"):
        try:
            # Load and preprocess T1w
            volume_hf  = load_and_preprocess(scans[subj_id])
            # Simulate low-field
            volume_lf  = simulate_low_field(volume_hf, converter)
            # Predict
            preds      = predict(model, volume_lf)
            btf, vbr, tcr, mci = float(preds[0]), float(preds[1]), float(preds[2]), float(preds[3])

            row = df.loc[subj_id]
            nwbv = float(row['nWBV'])
            cdr  = float(row['CDR']) if not pd.isna(row['CDR']) else np.nan
            age  = float(row['Age'])

            results.append({
                "subject_id": subj_id,
                "age": age,
                "cdr": cdr,
                "gt_nwbv": nwbv,
                "pred_btf": btf,
                "pred_vbr": vbr,
                "pred_tcr": tcr,
                "pred_mci": mci,
            })

            gt_nwbv.append(nwbv)
            pred_btf.append(btf)
            if not np.isnan(cdr):
                gt_cdr.append(cdr)
                subj_ages.append(age)

        except Exception as e:
            print(f"  Skipped {subj_id}: {e}")

    print(f"\nProcessed {len(results)} subjects")

    # Compute validation metrics
    gt_nwbv  = np.array(gt_nwbv)
    pred_btf = np.array(pred_btf)
    gt_cdr   = np.array(gt_cdr)
    subj_ages= np.array(subj_ages)

    metrics_btf_nwbv = compute_metrics(gt_nwbv, pred_btf, "BTF vs nWBV")
    metrics_btf_age  = compute_metrics(subj_ages, pred_btf[:len(subj_ages)], "BTF vs Age")

    # CDR group analysis
    cdr_groups = {}
    for r in results:
        if not np.isnan(r['cdr']):
            g = str(r['cdr'])
            cdr_groups.setdefault(g, []).append(r['pred_btf'])
    cdr_mean_btf = {g: round(float(np.mean(v)), 4) for g, v in cdr_groups.items()}

    summary = {
        "n_subjects": len(results),
        "btf_vs_nwbv": metrics_btf_nwbv,
        "btf_vs_age": metrics_btf_age,
        "cdr_mean_btf": cdr_mean_btf,
        "overall_pred_btf": {
            "mean": round(float(np.mean(pred_btf)), 4),
            "std":  round(float(np.std(pred_btf)), 4),
            "min":  round(float(np.min(pred_btf)), 4),
            "max":  round(float(np.max(pred_btf)), 4),
        }
    }

    # Save results
    results_path = RESULTS_DIR / "oasis_validation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    summary_path = RESULTS_DIR / "oasis_validation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print report
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"\nBTF (predicted) vs nWBV (FreeSurfer ground truth):")
    print(f"  Pearson r  = {metrics_btf_nwbv.get('pearson_r', 'N/A')}  (p={metrics_btf_nwbv.get('pearson_p', 'N/A')})")
    print(f"  Spearman r = {metrics_btf_nwbv.get('spearman_r', 'N/A')}  (p={metrics_btf_nwbv.get('spearman_p', 'N/A')})")
    print(f"  MAE = {metrics_btf_nwbv.get('mae', 'N/A')},  RMSE = {metrics_btf_nwbv.get('rmse', 'N/A')}")
    print(f"\nMean BTF by CDR group:")
    for cdr_val in sorted(cdr_mean_btf.keys(), key=float):
        n = len(cdr_groups.get(cdr_val, []))
        print(f"  CDR {cdr_val}: mean BTF = {cdr_mean_btf[cdr_val]}  (n={n})")
    print(f"\nExpected: BTF should DECREASE as CDR increases (brain atrophy)")
    print(f"\nResults saved to: {RESULTS_DIR}")

    # Write text report
    report_path = RESULTS_DIR / "oasis_validation_report.txt"
    with open(report_path, 'w') as f:
        f.write("OASIS-1 VALIDATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Subjects processed: {len(results)}\n")
        f.write(f"Device: {DEVICE}\n\n")
        f.write("BTF (predicted) vs nWBV (FreeSurfer GT):\n")
        for k, v in metrics_btf_nwbv.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nMean Predicted BTF by CDR:\n")
        for g in sorted(cdr_mean_btf.keys(), key=float):
            f.write(f"  CDR {g}: {cdr_mean_btf[g]}\n")
        f.write(json.dumps(summary, indent=2))

    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
