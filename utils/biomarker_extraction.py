"""
Biomarker extraction from high-field MRI volumes.

Two extraction modes:
1. FreeSurfer-based (preferred): Uses subcortical segmentation labels from
   FreeSurfer to compute anatomically validated biomarkers.
2. Intensity-based (fallback): Uses Otsu thresholding and morphological
   operations when segmentation labels are not available.

Biomarkers extracted:
- BTF (Brain Tissue Fraction): ratio of brain tissue to total volume
- TCR (Tissue Contrast Ratio): gray matter / white matter volume ratio
- VBR (Ventricle-to-Brain Ratio): ventricle volume / brain volume
- MCI (Mean Cortical Intensity): average intensity in cortical gray matter
"""

import numpy as np
from pathlib import Path
from typing import Dict
import json

from scipy.ndimage import binary_erosion, generate_binary_structure


BIOMARKER_NAMES = ["btf", "tcr", "vbr", "mci"]

# FreeSurfer subcortical segmentation label IDs
# Reference: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
FS_LABELS = {
    "left_wm": 2,
    "left_cortex": 3,
    "left_lat_vent": 4,
    "left_inf_lat_vent": 5,
    "left_cerebellum_wm": 7,
    "left_cerebellum_cortex": 8,
    "left_thalamus": 10,
    "left_caudate": 11,
    "left_putamen": 12,
    "left_pallidum": 13,
    "third_ventricle": 14,
    "fourth_ventricle": 15,
    "brain_stem": 16,
    "left_hippocampus": 17,
    "left_amygdala": 18,
    "csf": 24,
    "left_accumbens": 26,
    "left_ventral_dc": 28,
    "right_wm": 41,
    "right_cortex": 42,
    "right_lat_vent": 43,
    "right_inf_lat_vent": 44,
    "right_cerebellum_wm": 46,
    "right_cerebellum_cortex": 47,
    "right_thalamus": 49,
    "right_caudate": 50,
    "right_putamen": 51,
    "right_pallidum": 52,
    "right_hippocampus": 53,
    "right_amygdala": 54,
    "right_accumbens": 58,
    "right_ventral_dc": 60,
}

# Grouped label sets for biomarker computation
VENTRICLE_LABELS = {
    FS_LABELS["left_lat_vent"],
    FS_LABELS["left_inf_lat_vent"],
    FS_LABELS["right_lat_vent"],
    FS_LABELS["right_inf_lat_vent"],
    FS_LABELS["third_ventricle"],
    FS_LABELS["fourth_ventricle"],
}

CORTEX_LABELS = {
    FS_LABELS["left_cortex"],
    FS_LABELS["right_cortex"],
}

WHITE_MATTER_LABELS = {
    FS_LABELS["left_wm"],
    FS_LABELS["right_wm"],
}

# All brain tissue (everything non-zero excluding CSF and ventricles)
BRAIN_TISSUE_LABELS = (
    CORTEX_LABELS | WHITE_MATTER_LABELS |
    {FS_LABELS[k] for k in [
        "left_cerebellum_wm", "left_cerebellum_cortex",
        "right_cerebellum_wm", "right_cerebellum_cortex",
        "left_thalamus", "left_caudate", "left_putamen", "left_pallidum",
        "right_thalamus", "right_caudate", "right_putamen", "right_pallidum",
        "left_hippocampus", "left_amygdala",
        "right_hippocampus", "right_amygdala",
        "left_accumbens", "right_accumbens",
        "left_ventral_dc", "right_ventral_dc",
        "brain_stem",
    ]}
)


# ---------------------------------------------------------------------------
# FreeSurfer-based extraction (preferred)
# ---------------------------------------------------------------------------

def compute_biomarkers_from_segmentation(
    volume: np.ndarray,
    seg: np.ndarray,
) -> Dict[str, float]:
    """
    Extract 4 biomarkers using FreeSurfer segmentation labels.

    Args:
        volume: 3D MRI volume (D, H, W), values in [0, 1]
        seg: 3D integer segmentation map (same shape), FreeSurfer label IDs

    Returns:
        dict with keys: 'btf', 'tcr', 'vbr', 'mci'
    """
    total_voxels = volume.size

    # Build masks from segmentation labels
    brain_mask = np.isin(seg, list(BRAIN_TISSUE_LABELS | VENTRICLE_LABELS))
    cortex_mask = np.isin(seg, list(CORTEX_LABELS))
    wm_mask = np.isin(seg, list(WHITE_MATTER_LABELS))
    ventricle_mask = np.isin(seg, list(VENTRICLE_LABELS))

    brain_count = int(np.sum(brain_mask))
    cortex_count = int(np.sum(cortex_mask))
    wm_count = int(np.sum(wm_mask))
    ventricle_count = int(np.sum(ventricle_mask))

    # BTF: Brain Tissue Fraction
    btf = brain_count / total_voxels if total_voxels > 0 else 0.0

    # TCR: Tissue Contrast Ratio (gray matter / white matter volume ratio)
    # Normalized: ratio typically 1.0 - 3.0, map to [0, 1]
    if wm_count > 0:
        gw_ratio = cortex_count / wm_count
        tcr = float(np.clip(gw_ratio / 3.0, 0.0, 1.0))
    else:
        tcr = 0.5

    # VBR: Ventricle-to-Brain Ratio
    # Normalized: ratio typically 0.01 - 0.15, map to [0, 1]
    if brain_count > 0:
        vbr_raw = ventricle_count / brain_count
        vbr = float(np.clip(vbr_raw / 0.15, 0.0, 1.0))
    else:
        vbr = 0.0

    # MCI: Mean Cortical Intensity
    if cortex_count > 0:
        mci = float(np.mean(volume[cortex_mask]))
    else:
        mci = 0.5

    return {"btf": btf, "tcr": tcr, "vbr": vbr, "mci": mci}


# ---------------------------------------------------------------------------
# Intensity-based extraction (fallback for volumes without segmentations)
# ---------------------------------------------------------------------------

def _otsu_threshold(data: np.ndarray) -> float:
    """Compute Otsu's threshold for a 1D array of values."""
    hist, bin_edges = np.histogram(data, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
    if total == 0:
        return 0.5

    sum_total = (hist * bin_centers).sum()
    sum_bg = 0.0
    weight_bg = 0
    max_variance = 0.0
    threshold = bin_centers[0]

    for i in range(len(hist)):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += hist[i] * bin_centers[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > max_variance:
            max_variance = variance
            threshold = bin_centers[i]

    return float(threshold)


def compute_biomarkers_intensity(volume: np.ndarray) -> Dict[str, float]:
    """
    Extract 4 biomarkers using intensity-based heuristics (fallback).

    Args:
        volume: 3D numpy array (D, H, W), values in [0, 1]

    Returns:
        dict with keys: 'btf', 'tcr', 'vbr', 'mci'
    """
    volume = volume.astype(np.float32)
    v_min, v_max = volume.min(), volume.max()
    if v_max > v_min:
        volume = (volume - v_min) / (v_max - v_min)

    threshold = _otsu_threshold(volume.flatten())

    # BTF
    brain_mask = volume > threshold
    btf = float(np.sum(brain_mask) / volume.size)

    # TCR
    brain_voxels = volume[volume > threshold * 0.5]
    if len(brain_voxels) > 0:
        median_val = np.median(brain_voxels)
        high = brain_voxels[brain_voxels >= median_val]
        low = brain_voxels[brain_voxels < median_val]
        if len(low) > 0 and np.mean(low) > 1e-8:
            ratio = np.mean(high) / np.mean(low)
            tcr = float(np.clip((ratio - 1.0) / 3.0, 0.0, 1.0))
        else:
            tcr = 0.5
    else:
        tcr = 0.5

    # VBR
    brain_mask_loose = volume > threshold * 0.3
    brain_count = np.sum(brain_mask_loose)
    if brain_count > 0:
        shape = volume.shape
        slices = tuple(slice(int(s * 0.3), int(s * 0.7)) for s in shape)
        central = volume[slices]
        central_brain = brain_mask_loose[slices]
        brain_mean = np.mean(volume[brain_mask_loose])
        vent_thresh = 0.3 * brain_mean
        vent_voxels = np.sum((central < vent_thresh) & central_brain)
        vbr = float(np.clip(vent_voxels / brain_count / 0.2, 0.0, 1.0))
    else:
        vbr = 0.0

    # MCI
    brain_mask_med = volume > threshold * 0.5
    if np.sum(brain_mask_med) > 0:
        struct = generate_binary_structure(3, 1)
        eroded = binary_erosion(brain_mask_med, structure=struct, iterations=3)
        cortex_mask = brain_mask_med & ~eroded
        if np.sum(cortex_mask) > 0:
            mci = float(np.mean(volume[cortex_mask]))
        else:
            mci = float(np.mean(volume[brain_mask_med]))
    else:
        mci = 0.5

    return {"btf": btf, "tcr": tcr, "vbr": vbr, "mci": mci}


# ---------------------------------------------------------------------------
# Unified extraction entry point
# ---------------------------------------------------------------------------

def compute_biomarkers(
    volume: np.ndarray,
    seg: np.ndarray = None,
) -> Dict[str, float]:
    """
    Extract 4 biomarkers from a high-field MRI volume.

    Uses FreeSurfer segmentation when available (more accurate),
    falls back to intensity-based heuristics otherwise.

    Args:
        volume: 3D numpy array (D, H, W), values in [0, 1]
        seg: optional 3D integer segmentation map (FreeSurfer labels)

    Returns:
        dict with keys: 'btf', 'tcr', 'vbr', 'mci'
    """
    if seg is not None and seg.size > 0:
        return compute_biomarkers_from_segmentation(volume, seg)
    return compute_biomarkers_intensity(volume)


def extract_biomarkers_for_dataset(high_field_dir: str) -> Dict[str, np.ndarray]:
    """
    Process all .npy files in high_field_dir and extract biomarkers.

    For IXI volumes, looks for FreeSurfer segmentations in
    high_field_dir/segmentations/{stem}_seg.npy. Falls back to
    intensity-based extraction when segmentations are not found.

    Args:
        high_field_dir: path to directory containing high-field .npy MRI volumes

    Returns:
        dict mapping filename stem -> np.array([btf, tcr, vbr, mci])
    """
    high_field_path = Path(high_field_dir)
    cache_path = high_field_path / "biomarker_labels.json"
    seg_dir = high_field_path / "segmentations"

    # Check cache
    if cache_path.exists():
        with open(cache_path, "r") as f:
            cached = json.load(f)
        return {k: np.array(v, dtype=np.float32) for k, v in cached.items()}

    # Find all volume files
    volume_files = sorted(high_field_path.glob("*.npy"))
    if len(volume_files) == 0:
        print(f"Warning: No .npy files found in {high_field_dir}")
        return {}

    biomarker_labels = {}
    seg_count = 0
    intensity_count = 0

    for vol_path in volume_files:
        stem = vol_path.stem
        volume = np.load(vol_path)

        if volume.ndim == 4:
            volume = volume[0]

        # Normalize
        volume = volume.astype(np.float32)
        v_min, v_max = volume.min(), volume.max()
        if v_max > v_min:
            volume = (volume - v_min) / (v_max - v_min)

        # Try to load FreeSurfer segmentation
        seg_path = seg_dir / f"{stem}_seg.npy"
        if seg_path.exists():
            seg = np.load(seg_path)
            method = "FreeSurfer"
            seg_count += 1
        else:
            seg = None
            method = "intensity"
            intensity_count += 1

        biomarkers = compute_biomarkers(volume, seg)
        biomarker_labels[stem] = np.array(
            [biomarkers["btf"], biomarkers["tcr"], biomarkers["vbr"], biomarkers["mci"]],
            dtype=np.float32,
        )
        print(
            f"  Extracting biomarkers from {stem}... ({method})"
        )
        print(
            f"    BTF={biomarkers['btf']:.4f}, TCR={biomarkers['tcr']:.4f}, "
            f"VBR={biomarkers['vbr']:.4f}, MCI={biomarkers['mci']:.4f}"
        )

    # Cache results
    cache_data = {k: v.tolist() for k, v in biomarker_labels.items()}
    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=2)
    print(f"  Cached biomarker labels to {cache_path}")
    print(f"  Methods used: {seg_count} FreeSurfer, {intensity_count} intensity-based")

    return biomarker_labels
