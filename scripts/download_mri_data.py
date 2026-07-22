"""
Download real high-field MRI brain images from public datasets
Supports multiple sources: nilearn sample data, OASIS, and other public datasets
"""

import sys
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
import requests
import zipfile
import tarfile
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def download_nilearn_sample_data(output_dir, num_samples=5):
    """
    Download sample MRI data from nilearn (requires nilearn package)
    This provides real brain MRI data for testing
    """
    print("Downloading sample MRI data from nilearn...")
    
    try:
        from nilearn import datasets
        from nilearn import image
    except ImportError:
        print("nilearn not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nilearn"])
        from nilearn import datasets
        from nilearn import image
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Fetching OASIS dataset sample...")
    # Download OASIS dataset (small sample)
    oasis_dataset = datasets.fetch_oasis_vbm(n_subjects=min(num_samples, 20))
    
    volumes = []
    for i, img_path in enumerate(tqdm(oasis_dataset.gray_matter_maps[:num_samples], desc="Processing")):
        try:
            # Load the image
            img = image.load_img(img_path)
            data = img.get_fdata()
            
            # Normalize to [0, 1]
            data = data.astype(np.float32)
            v_min, v_max = data.min(), data.max()
            if v_max > v_min:
                data = (data - v_min) / (v_max - v_min)
            
            # Save as numpy array
            output_path = output_dir / f"oasis_high_field_{i+1:03d}.npy"
            np.save(output_path, data)
            volumes.append(output_path)
            
            print(f"  Saved: {output_path.name} (shape: {data.shape})")
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")
            continue
    
    print(f"\n✓ Downloaded {len(volumes)} volumes from OASIS dataset")
    return volumes


def download_abide_sample_data(output_dir, num_samples=3):
    """
    Download sample data from ABIDE dataset
    """
    print("Downloading sample MRI data from ABIDE...")
    
    try:
        from nilearn import datasets
        from nilearn import image
    except ImportError:
        print("nilearn not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nilearn"])
        from nilearn import datasets
        from nilearn import image
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Fetching ABIDE dataset sample...")
    # Download ABIDE dataset (small sample)
    abide_dataset = datasets.fetch_abide_pcp(n_subjects=min(num_samples, 10))
    
    volumes = []
    for i, img_path in enumerate(tqdm(abide_dataset.func_preproc[:num_samples], desc="Processing")):
        try:
            # Load the image
            img = image.load_img(img_path)
            data = img.get_fdata()
            
            # If 4D, take first volume
            if len(data.shape) == 4:
                data = data[:, :, :, 0]
            
            # Normalize to [0, 1]
            data = data.astype(np.float32)
            v_min, v_max = data.min(), data.max()
            if v_max > v_min:
                data = (data - v_min) / (v_max - v_min)
            
            # Save as numpy array
            output_path = output_dir / f"abide_high_field_{i+1:03d}.npy"
            np.save(output_path, data)
            volumes.append(output_path)
            
            print(f"  Saved: {output_path.name} (shape: {data.shape})")
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")
            continue
    
    print(f"\n✓ Downloaded {len(volumes)} volumes from ABIDE dataset")
    return volumes


def download_haxby_sample_data(output_dir):
    """
    Download Haxby 2001 dataset (small, fast download)
    """
    print("Downloading Haxby 2001 dataset...")
    
    try:
        from nilearn import datasets
        from nilearn import image
    except ImportError:
        print("nilearn not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nilearn"])
        from nilearn import datasets
        from nilearn import image
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Fetching Haxby dataset...")
    haxby_dataset = datasets.fetch_haxby()
    
    volumes = []
    try:
        # Load the main image
        img = image.load_img(haxby_dataset.anat)
        data = img.get_fdata()
        
        # Normalize to [0, 1]
        data = data.astype(np.float32)
        v_min, v_max = data.min(), data.max()
        if v_max > v_min:
            data = (data - v_min) / (v_max - v_min)
        
        # Save as numpy array
        output_path = output_dir / "haxby_high_field.npy"
        np.save(output_path, data)
        volumes.append(output_path)
        
        print(f"  Saved: {output_path.name} (shape: {data.shape})")
    except Exception as e:
        print(f"  Error processing Haxby data: {e}")
    
    print(f"\n✓ Downloaded {len(volumes)} volume from Haxby dataset")
    return volumes


def download_from_url(url, output_dir, extract=True):
    """
    Download MRI data from a direct URL
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading from URL: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        filename = url.split('/')[-1]
        filepath = output_dir / filename
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        
        print(f"  Downloaded: {filepath}")
        
        # Extract if needed
        if extract:
            if filename.endswith('.zip'):
                print("  Extracting ZIP file...")
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
            elif filename.endswith(('.tar', '.tar.gz', '.tgz')):
                print("  Extracting TAR file...")
                with tarfile.open(filepath, 'r:*') as tar_ref:
                    tar_ref.extractall(output_dir)
        
        return filepath
    except Exception as e:
        print(f"  Error downloading: {e}")
        return None


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
        choices=['nilearn', 'oasis', 'abide', 'haxby', 'all'],
        default='nilearn',
        help='Data source to download from'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of samples to download (for nilearn sources)'
    )
    parser.add_argument(
        '--url',
        type=str,
        default=None,
        help='Direct URL to download MRI data from'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Downloading Real High-Field MRI Data")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print(f"Source: {args.source}")
    print("="*80)
    
    volumes = []
    
    if args.url:
        # Download from direct URL
        download_from_url(args.url, args.output_dir)
    elif args.source == 'nilearn' or args.source == 'all':
        try:
            vols = download_nilearn_sample_data(args.output_dir, args.num_samples)
            volumes.extend(vols)
        except Exception as e:
            print(f"Error with nilearn: {e}")
    
    if args.source == 'oasis' or args.source == 'all':
        try:
            vols = download_nilearn_sample_data(args.output_dir, args.num_samples)
            volumes.extend(vols)
        except Exception as e:
            print(f"Error with OASIS: {e}")
    
    if args.source == 'abide' or args.source == 'all':
        try:
            vols = download_abide_sample_data(args.output_dir, min(args.num_samples, 3))
            volumes.extend(vols)
        except Exception as e:
            print(f"Error with ABIDE: {e}")
    
    if args.source == 'haxby' or args.source == 'all':
        try:
            vols = download_haxby_sample_data(args.output_dir)
            volumes.extend(vols)
        except Exception as e:
            print(f"Error with Haxby: {e}")
    
    print("\n" + "="*80)
    if volumes:
        print(f"✓ Successfully downloaded {len(volumes)} high-field MRI volumes")
        print(f"  Location: {args.output_dir}")
        print(f"\nTo convert to low-field, run:")
        print(f"  python3 scripts/convert_high_to_low.py --input {args.output_dir} --output data/low_field")
    else:
        print("✗ No volumes downloaded. Check errors above.")
        print("\nAlternative: Use synthetic data generation:")
        print("  python3 scripts/generate_high_field.py --num-volumes 5")
    print("="*80)


if __name__ == "__main__":
    main()




