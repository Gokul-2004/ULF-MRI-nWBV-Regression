"""
Download preprocessed IXI brain MRI dataset and convert to project format.

Downloads skull-stripped, affine-aligned T1 volumes from the preprocessed
IXI dataset (Chen et al.), converts .pkl to .npy, and generates simulated
low-field versions using FieldConverter.

Source: https://github.com/junyuchen245/Preprocessed_IXI_Dataset
License: CC BY-SA 3.0

Usage:
    python3 scripts/download_ixi_data.py --num-subjects 50
    python3 scripts/download_ixi_data.py --num-subjects 100 --skip-download
"""

import sys
import argparse
import pickle
import gc
import random
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


GDRIVE_FILE_ID = "1-VQewCVNj5eTtc3eQGhTM2yXBQmgm8Ol"
DATASET_FILENAME = "IXI_data.zip"


def pkload(fname):
    """Load a pickle file (matches IXI dataset convention)."""
    with open(fname, "rb") as f:
        return pickle.load(f)


def download_ixi(download_dir: Path) -> Path:
    """Download the preprocessed IXI dataset from Google Drive."""
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / DATASET_FILENAME
    extracted_dir = download_dir / "IXI_data"

    if extracted_dir.exists() and any(extracted_dir.rglob("*.pkl")):
        print(f"  Dataset already extracted at {extracted_dir}")
        return extracted_dir

    if not zip_path.exists():
        print("  Installing gdown...")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q",
             "--break-system-packages", "gdown"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print(f"  Downloading IXI dataset (~1.44 GB)...")
        import gdown
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, str(zip_path), quiet=False)

        if not zip_path.exists():
            print("ERROR: Download failed. Please download manually from:")
            print(f"  https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}")
            print(f"  Save as: {zip_path}")
            sys.exit(1)
    else:
        print(f"  Zip already exists: {zip_path}")

    # Extract
    print("  Extracting...")
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(download_dir)

    print(f"  Extracted to {extracted_dir}")
    return extracted_dir


def discover_pkl_files(ixi_dir: Path) -> list:
    """Find all .pkl files in the IXI dataset directory."""
    pkl_files = []
    for split_dir in ["Train", "Val", "Test"]:
        split_path = ixi_dir / split_dir
        if split_path.exists():
            pkl_files.extend(sorted(split_path.glob("*.pkl")))

    # Also check root level
    pkl_files.extend(
        f for f in sorted(ixi_dir.glob("*.pkl"))
        if f.name != "atlas.pkl"
    )

    # Deduplicate
    seen = set()
    unique = []
    for f in pkl_files:
        if f.name not in seen:
            seen.add(f.name)
            unique.append(f)

    return unique


def convert_pkl_to_npy(
    pkl_files: list,
    high_field_dir: Path,
    num_subjects: int,
) -> list:
    """Convert .pkl volumes to individual .npy files.

    Saves both the MRI volume and the FreeSurfer segmentation label map.
    """
    high_field_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = high_field_dir / "segmentations"
    seg_dir.mkdir(parents=True, exist_ok=True)

    # Shuffle deterministically for variety, then take num_subjects
    rng = random.Random(42)
    selected = list(pkl_files)
    rng.shuffle(selected)
    selected = selected[:num_subjects]

    saved_paths = []
    for i, pkl_path in enumerate(selected):
        stem = f"IXI_{i + 1:03d}"
        out_path = high_field_dir / f"{stem}.npy"
        seg_path = seg_dir / f"{stem}_seg.npy"

        if out_path.exists() and seg_path.exists():
            saved_paths.append(out_path)
            continue

        try:
            data = pkload(str(pkl_path))
            # data is a tuple: (image, segmentation_label)
            if isinstance(data, (tuple, list)) and len(data) >= 2:
                volume = np.array(data[0], dtype=np.float32)
                seg_label = np.array(data[1], dtype=np.int16)
            else:
                volume = np.array(data, dtype=np.float32)
                seg_label = None

            # Handle 4D
            if volume.ndim == 4:
                volume = volume[0]

            # Normalize to [0, 1]
            v_min, v_max = volume.min(), volume.max()
            if v_max > v_min:
                volume = (volume - v_min) / (v_max - v_min)

            np.save(out_path, volume)
            if seg_label is not None:
                np.save(seg_path, seg_label)

            saved_paths.append(out_path)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"    [{i + 1}/{num_subjects}] {pkl_path.name} -> {stem}.npy  "
                      f"shape={volume.shape}"
                      + (f"  seg={seg_label.shape}" if seg_label is not None else ""))

            del volume, seg_label, data
            gc.collect()

        except Exception as e:
            print(f"    WARNING: Failed to convert {pkl_path.name}: {e}")
            continue

    return saved_paths


def generate_low_field(
    high_field_dir: Path,
    low_field_dir: Path,
    field_config: dict,
) -> list:
    """Generate low-field versions using FieldConverter."""
    from utils.field_conversion import FieldConverter

    low_field_dir.mkdir(parents=True, exist_ok=True)
    converter = FieldConverter(field_config)

    ixi_files = sorted(high_field_dir.glob("IXI_*.npy"))
    saved = []

    for i, hf_path in enumerate(ixi_files):
        lf_name = f"{hf_path.stem}_low_field.npy"
        lf_path = low_field_dir / lf_name

        if lf_path.exists():
            saved.append(lf_path)
            continue

        volume = np.load(hf_path)
        low_field = converter.convert(volume, method="combined")
        np.save(lf_path, low_field.astype(np.float32))
        saved.append(lf_path)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"    [{i + 1}/{len(ixi_files)}] {hf_path.name} -> {lf_name}")

        del volume, low_field
        gc.collect()

    return saved


def invalidate_cache(high_field_dir: Path):
    """Delete biomarker_labels.json so it regenerates with all volumes."""
    cache_path = high_field_dir / "biomarker_labels.json"
    if cache_path.exists():
        cache_path.unlink()
        print(f"  Deleted cache: {cache_path}")
    else:
        print("  No cache to delete")


def verify(high_field_dir: Path, low_field_dir: Path, num_spot_checks: int = 3):
    """Verify converted data meets pipeline requirements."""
    hf_files = sorted(high_field_dir.glob("IXI_*.npy"))
    lf_files = sorted(low_field_dir.glob("IXI_*_low_field.npy"))

    print(f"  High-field IXI files: {len(hf_files)}")
    print(f"  Low-field IXI files:  {len(lf_files)}")

    # Check naming convention
    missing = []
    for hf in hf_files:
        expected_lf = low_field_dir / f"{hf.stem}_low_field.npy"
        if not expected_lf.exists():
            missing.append(hf.stem)

    if missing:
        print(f"  WARNING: {len(missing)} high-field files missing low-field pair")
        return False

    # Spot-check data quality
    rng = random.Random(123)
    check_files = rng.sample(list(hf_files), min(num_spot_checks, len(hf_files)))

    for hf in check_files:
        vol = np.load(hf)
        ok = True
        if vol.ndim != 3:
            print(f"  FAIL: {hf.name} is {vol.ndim}D, expected 3D")
            ok = False
        if vol.dtype != np.float32:
            print(f"  FAIL: {hf.name} dtype={vol.dtype}, expected float32")
            ok = False
        if vol.min() < -0.01 or vol.max() > 1.01:
            print(f"  FAIL: {hf.name} range=[{vol.min():.3f}, {vol.max():.3f}]")
            ok = False
        if ok:
            print(f"  OK: {hf.name} shape={vol.shape} range=[{vol.min():.3f}, {vol.max():.3f}]")

    # Check total volume count (existing + IXI)
    all_hf = list(high_field_dir.glob("*.npy"))
    all_lf = list(low_field_dir.glob("*_low_field.npy"))
    print(f"\n  Total high-field volumes: {len(all_hf)}")
    print(f"  Total low-field volumes:  {len(all_lf)}")

    cache = high_field_dir / "biomarker_labels.json"
    if cache.exists():
        print("  WARNING: biomarker_labels.json cache still exists — should be deleted")
        return False

    print("\n  All checks passed.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download IXI dataset and prepare for biomarker inference"
    )
    parser.add_argument(
        "--num-subjects", type=int, default=50,
        help="Number of IXI subjects to process (default: 50)"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download if data already exists"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config file for FieldConverter settings"
    )
    parser.add_argument(
        "--download-dir", type=str, default="data/ixi_raw",
        help="Directory to store downloaded IXI data"
    )
    args = parser.parse_args()

    import yaml
    with open(project_root / args.config, "r") as f:
        config = yaml.safe_load(f)

    high_field_dir = project_root / config["data"]["high_field_dir"]
    low_field_dir = project_root / config["data"]["low_field_dir"]
    download_dir = project_root / args.download_dir
    field_config = config.get("field_conversion", {})

    print("=" * 70)
    print("IXI Dataset Integration")
    print("=" * 70)

    # Step 1: Download
    if not args.skip_download:
        print("\n[Step 1] Downloading preprocessed IXI dataset...")
        ixi_dir = download_ixi(download_dir)
    else:
        ixi_dir = download_dir / "IXI_data"
        if not ixi_dir.exists():
            print(f"ERROR: --skip-download set but {ixi_dir} not found")
            sys.exit(1)
        print("\n[Step 1] Skipping download (--skip-download)")

    # Step 2: Discover pkl files
    print("\n[Step 2] Discovering .pkl files...")
    pkl_files = discover_pkl_files(ixi_dir)
    print(f"  Found {len(pkl_files)} .pkl files")

    if len(pkl_files) == 0:
        print("ERROR: No .pkl files found in the dataset")
        sys.exit(1)

    num_to_process = min(args.num_subjects, len(pkl_files))
    print(f"  Will process {num_to_process} subjects")

    # Step 3: Convert to .npy
    print(f"\n[Step 3] Converting {num_to_process} volumes to .npy...")
    saved_hf = convert_pkl_to_npy(pkl_files, high_field_dir, num_to_process)
    print(f"  Saved {len(saved_hf)} high-field volumes")

    # Step 4: Generate low-field versions
    print(f"\n[Step 4] Generating low-field versions...")
    saved_lf = generate_low_field(high_field_dir, low_field_dir, field_config)
    print(f"  Saved {len(saved_lf)} low-field volumes")

    # Step 5: Invalidate cache
    print("\n[Step 5] Invalidating biomarker cache...")
    invalidate_cache(high_field_dir)

    # Step 6: Verify
    print("\n[Step 6] Verifying...")
    ok = verify(high_field_dir, low_field_dir)

    print("\n" + "=" * 70)
    if ok:
        print("Done! Run the training pipeline with:")
        print("  python3 experiments/stage1/run_stage1.py --config configs/config.yaml")
    else:
        print("Verification found issues. Please check the output above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
