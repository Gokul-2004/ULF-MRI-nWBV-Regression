"""
Download real high-field MRI brain images from public datasets
Supports: fastMRI, nilearn datasets, and direct URL downloads
"""

import sys
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
import json
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def install_package(package_name):
    """Install a package using pip in user mode"""
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "--user", package_name
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False


def download_fastmri_sample(output_dir, num_samples=3):
    """
    Download sample data from fastMRI dataset
    fastMRI provides real MRI data from NYU Langone Health
    """
    print("Downloading from fastMRI dataset...")
    print("Note: fastMRI requires manual download from:")
    print("  https://fastmri.med.nyu.edu/")
    print("  Or use: python3 -m fastmri.data.download")
    
    try:
        # Try to use fastmri package if available
        try:
            from fastmri.data import download
            print("fastmri package found, attempting download...")
            # This would download the dataset
            # download.main()  # Uncomment if you want to use this
        except ImportError:
            print("fastmri package not found.")
            print("To install: pip install --user fastmri")
            print("Or download manually from: https://fastmri.med.nyu.edu/")
    except Exception as e:
        print(f"Error: {e}")
    
    return []


def download_nilearn_data(output_dir, num_samples=3):
    """
    Download sample MRI data using nilearn
    This provides real brain MRI data from various public datasets
    """
    print("Downloading sample MRI data using nilearn...")
    
    # Try to import nilearn
    try:
        from nilearn import datasets
        from nilearn import image
    except ImportError:
        print("nilearn not found. Attempting to install...")
        if not install_package("nilearn"):
            print("Failed to install nilearn automatically.")
            print("Please install manually: pip install --user nilearn")
            return []
        from nilearn import datasets
        from nilearn import image
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    volumes = []
    
    # Try Haxby dataset (small, fast)
    try:
        print("\n1. Downloading Haxby 2001 dataset (small, fast)...")
        haxby_dataset = datasets.fetch_haxby()
        
        img = image.load_img(haxby_dataset.anat)
        data = img.get_fdata()
        
        # Normalize
        data = data.astype(np.float32)
        v_min, v_max = data.min(), data.max()
        if v_max > v_min:
            data = (data - v_min) / (v_max - v_min)
        
        output_path = output_dir / "haxby_high_field.npy"
        np.save(output_path, data)
        volumes.append(output_path)
        print(f"   ✓ Saved: {output_path.name} (shape: {data.shape})")
    except Exception as e:
        print(f"   ✗ Error with Haxby: {e}")
    
    # Try OASIS (if requested)
    if num_samples > 1:
        try:
            print(f"\n2. Downloading OASIS dataset ({num_samples} samples)...")
            print("   (This may take a while, downloading real brain data...)")
            oasis_dataset = datasets.fetch_oasis_vbm(n_subjects=min(num_samples, 10))
            
            for i, img_path in enumerate(tqdm(
                oasis_dataset.gray_matter_maps[:num_samples-1], 
                desc="   Processing OASIS"
            )):
                try:
                    img = image.load_img(img_path)
                    data = img.get_fdata()
                    
                    # Normalize
                    data = data.astype(np.float32)
                    v_min, v_max = data.min(), data.max()
                    if v_max > v_min:
                        data = (data - v_min) / (v_max - v_min)
                    
                    output_path = output_dir / f"oasis_high_field_{i+1:03d}.npy"
                    np.save(output_path, data)
                    volumes.append(output_path)
                    print(f"   ✓ Saved: {output_path.name} (shape: {data.shape})")
                except Exception as e:
                    print(f"   ✗ Error processing sample {i+1}: {e}")
        except Exception as e:
            print(f"   ✗ Error with OASIS: {e}")
            print("   (OASIS download may require more time/space)")
    
    return volumes


def download_from_github_release(repo, output_dir, pattern="*.nii.gz"):
    """
    Download MRI data from a GitHub release
    """
    print(f"Downloading from GitHub: {repo}")
    print("Note: This requires manual download or using GitHub API")
    print(f"Repository: {repo}")
    return []


def create_download_instructions(output_dir):
    """Create a file with instructions for manual downloads"""
    instructions = """
# Instructions for Downloading Real High-Field MRI Data

## Option 1: Using nilearn (Recommended - Easiest)

```bash
# Install nilearn
pip install --user nilearn

# Run the download script
python3 scripts/download_real_mri.py --source nilearn --num-samples 5
```

## Option 2: fastMRI Dataset

1. Visit: https://fastmri.med.nyu.edu/
2. Register and download the brain dataset
3. Extract to: data/high_field/

Or use:
```bash
pip install --user fastmri
python3 -m fastmri.data.download
```

## Option 3: OASIS Dataset

1. Visit: https://www.oasis-brains.org/
2. Download the OASIS-1 dataset
3. Extract to: data/high_field/

## Option 4: Other Public Datasets

- **ADNI**: https://adni.loni.usc.edu/ (requires registration)
- **IXI**: https://brain-development.org/ixi-dataset/
- **Human Connectome Project**: https://www.humanconnectome.org/

## Option 5: Use Synthetic Data (For Testing)

If you just need data for testing the conversion pipeline:

```bash
python3 scripts/generate_high_field.py --num-volumes 10
```

This generates realistic synthetic brain MRI volumes that mimic real data.
"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "DOWNLOAD_INSTRUCTIONS.md", 'w') as f:
        f.write(instructions)
    
    print(f"\n✓ Download instructions saved to: {output_dir}/DOWNLOAD_INSTRUCTIONS.md")


def main():
    parser = argparse.ArgumentParser(
        description='Download real high-field MRI brain images from public datasets'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/high_field',
        help='Output directory for downloaded volumes'
    )
    parser.add_argument(
        '--source',
        type=str,
        choices=['nilearn', 'fastmri', 'instructions'],
        default='nilearn',
        help='Data source to download from'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=3,
        help='Number of samples to download'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Downloading Real High-Field MRI Data")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print(f"Source: {args.source}")
    print("="*80)
    
    volumes = []
    
    if args.source == 'instructions':
        create_download_instructions(args.output_dir)
        return
    
    elif args.source == 'nilearn':
        volumes = download_nilearn_data(args.output_dir, args.num_samples)
    
    elif args.source == 'fastmri':
        volumes = download_fastmri_sample(args.output_dir, args.num_samples)
    
    print("\n" + "="*80)
    if volumes:
        print(f"✓ Successfully downloaded {len(volumes)} high-field MRI volumes")
        print(f"  Location: {args.output_dir}")
        print(f"\nFiles downloaded:")
        for vol in volumes:
            print(f"  - {vol.name}")
        print(f"\nTo convert to low-field, run:")
        print(f"  python3 scripts/convert_high_to_low.py --input {args.output_dir} --output data/low_field")
    else:
        print("✗ No volumes downloaded automatically.")
        print("\nOptions:")
        print("1. Install nilearn: pip install --user nilearn")
        print("2. Then run: python3 scripts/download_real_mri.py --source nilearn")
        print("3. Or use synthetic data: python3 scripts/generate_high_field.py")
        print("4. Or see instructions: python3 scripts/download_real_mri.py --source instructions")
    print("="*80)


if __name__ == "__main__":
    main()




