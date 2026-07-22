"""
All-in-one script to get high-field MRI data and convert to low-field
Supports both real data download and synthetic data generation
"""

import sys
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.field_conversion import FieldConverter
import yaml


def generate_realistic_high_field(size=(256, 256, 256), save_path=None):
    """
    Generate a realistic high-field MRI volume that mimics real brain structure
    This creates synthetic data that's good enough for testing the conversion pipeline
    """
    print(f"Generating realistic high-field MRI volume ({size})...")
    
    volume = np.zeros(size, dtype=np.float32)
    center = np.array(size) // 2
    
    # Create realistic brain structures
    for i in tqdm(range(size[0]), desc="Generating volume", leave=False):
        for j in range(size[1]):
            for k in range(size[2]):
                # Distance from center
                dist = np.sqrt(
                    (i - center[0])**2 + 
                    (j - center[1])**2 + 
                    (k - center[2])**2
                )
                
                # Create brain tissue layers
                max_dist = max(size) * 0.4
                
                if dist < max_dist * 0.3:
                    # White matter (bright, 0.7-0.9)
                    intensity = 0.75 + 0.15 * np.exp(-dist / (max_dist * 0.2))
                elif dist < max_dist * 0.6:
                    # Gray matter (medium, 0.4-0.6)
                    intensity = 0.5 + 0.1 * np.exp(-dist / (max_dist * 0.4))
                elif dist < max_dist:
                    # CSF/background (dark, 0.1-0.3)
                    intensity = 0.2 + 0.1 * np.exp(-dist / (max_dist * 0.7))
                else:
                    intensity = 0.05
                
                # Add anatomical detail (cortical folds)
                intensity += 0.08 * (
                    np.sin(i * 0.08) * np.cos(j * 0.08) * 
                    np.sin(k * 0.08)
                )
                
                # Add fine structures
                intensity += 0.04 * (
                    np.sin(i * 0.15 + j * 0.12) * 
                    np.cos(k * 0.13)
                )
                
                volume[i, j, k] = np.clip(intensity, 0, 1)
    
    # Add realistic high-field noise (very low, high SNR)
    noise = np.random.normal(0, 0.015, size)
    volume = volume + noise
    volume = np.clip(volume, 0, 1)
    
    if save_path:
        np.save(save_path, volume)
        print(f"  Saved to: {save_path}")
    
    return volume


def convert_to_low_field(high_field_path, output_path, config):
    """Convert high-field image to low-field"""
    print(f"\nConverting to low-field: {high_field_path.name}")
    
    # Load high-field
    high_field = np.load(high_field_path)
    
    # Convert
    converter = FieldConverter(config)
    low_field = converter.convert(high_field, method='combined')
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, low_field)
    
    # Metrics
    mse = np.mean((high_field - low_field) ** 2)
    psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
    
    print(f"  High-field: shape={high_field.shape}, range=[{high_field.min():.3f}, {high_field.max():.3f}]")
    print(f"  Low-field:  shape={low_field.shape}, range=[{low_field.min():.3f}, {low_field.max():.3f}]")
    print(f"  PSNR: {psnr:.2f} dB, MSE: {mse:.6f}")
    
    return low_field


def try_download_real_data(output_dir, num_samples=3):
    """Try to download real MRI data using nilearn"""
    print("\nAttempting to download real MRI data...")
    
    # Check if nilearn is available
    try:
        from nilearn import datasets, image
        print("✓ nilearn found, downloading real data...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        volumes = []
        
        # Download Haxby (small, fast)
        try:
            print("  Downloading Haxby dataset...")
            haxby = datasets.fetch_haxby()
            img = image.load_img(haxby.anat)
            data = img.get_fdata().astype(np.float32)
            
            # Normalize
            v_min, v_max = data.min(), data.max()
            if v_max > v_min:
                data = (data - v_min) / (v_max - v_min)
            
            output_path = output_dir / "haxby_high_field.npy"
            np.save(output_path, data)
            volumes.append(output_path)
            print(f"  ✓ Saved: {output_path.name}")
        except Exception as e:
            print(f"  ✗ Haxby failed: {e}")
        
        # Try OASIS if more samples needed
        if len(volumes) < num_samples:
            try:
                print(f"  Downloading OASIS dataset ({num_samples - len(volumes)} samples)...")
                oasis = datasets.fetch_oasis_vbm(n_subjects=min(num_samples, 5))
                
                for i, img_path in enumerate(oasis.gray_matter_maps[:num_samples - len(volumes)]):
                    try:
                        img = image.load_img(img_path)
                        data = img.get_fdata().astype(np.float32)
                        
                        v_min, v_max = data.min(), data.max()
                        if v_max > v_min:
                            data = (data - v_min) / (v_max - v_min)
                        
                        output_path = output_dir / f"oasis_high_field_{i+1:03d}.npy"
                        np.save(output_path, data)
                        volumes.append(output_path)
                        print(f"  ✓ Saved: {output_path.name}")
                    except Exception as e:
                        print(f"  ✗ Error with sample {i+1}: {e}")
            except Exception as e:
                print(f"  ✗ OASIS failed: {e}")
        
        return volumes
        
    except ImportError:
        print("✗ nilearn not available")
        print("  To install: pip install --user nilearn")
        print("  Or use synthetic data (option below)")
        return []


def main():
    parser = argparse.ArgumentParser(
        description='Get high-field MRI data and convert to low-field'
    )
    parser.add_argument(
        '--high-field-dir',
        type=str,
        default='data/high_field',
        help='Directory for high-field images'
    )
    parser.add_argument(
        '--low-field-dir',
        type=str,
        default='data/low_field',
        help='Directory for low-field images'
    )
    parser.add_argument(
        '--source',
        type=str,
        choices=['real', 'synthetic', 'auto'],
        default='auto',
        help='Data source: real (download), synthetic (generate), or auto (try real first)'
    )
    parser.add_argument(
        '--num-volumes',
        type=int,
        default=5,
        help='Number of volumes to generate/download'
    )
    parser.add_argument(
        '--size',
        type=int,
        nargs=3,
        default=[256, 256, 256],
        help='Volume size for synthetic data (depth height width)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Config file path'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("High-Field to Low-Field MRI Conversion")
    print("="*80)
    
    high_field_dir = Path(args.high_field_dir)
    low_field_dir = Path(args.low_field_dir)
    high_field_dir.mkdir(parents=True, exist_ok=True)
    
    # Get high-field data
    high_field_files = []
    
    if args.source == 'real' or args.source == 'auto':
        if args.source == 'auto':
            print("\n[Step 1] Attempting to download real MRI data...")
            files = try_download_real_data(high_field_dir, args.num_volumes)
            if files:
                high_field_files = files
                print(f"✓ Downloaded {len(files)} real MRI volumes")
            else:
                print("  Falling back to synthetic data...")
                args.source = 'synthetic'
        else:
            files = try_download_real_data(high_field_dir, args.num_volumes)
            high_field_files = files
    
    if args.source == 'synthetic' or (args.source == 'auto' and not high_field_files):
        print(f"\n[Step 1] Generating {args.num_volumes} synthetic high-field volumes...")
        size = tuple(args.size)
        
        for i in range(args.num_volumes):
            output_path = high_field_dir / f"high_field_volume_{i+1:03d}.npy"
            generate_realistic_high_field(size, output_path)
            high_field_files.append(output_path)
        
        print(f"✓ Generated {len(high_field_files)} synthetic volumes")
    
    if not high_field_files:
        print("\n✗ No high-field data available!")
        print("\nOptions:")
        print("1. Install nilearn: pip install --user nilearn")
        print("2. Then run: python3 scripts/get_and_convert_mri.py --source real")
        print("3. Or use synthetic: python3 scripts/get_and_convert_mri.py --source synthetic")
        return
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    field_config = config.get('field_conversion', {
        'noise_level': 0.1,
        'resolution_factor': 0.5,
        'contrast_reduction': 0.3,
        'method': 'combined'
    })
    
    # Convert to low-field
    print(f"\n[Step 2] Converting {len(high_field_files)} volumes to low-field...")
    print("="*80)
    
    low_field_files = []
    for high_field_file in tqdm(high_field_files, desc="Converting"):
        low_field_name = f"{high_field_file.stem}_low_field.npy"
        low_field_path = low_field_dir / low_field_name
        
        convert_to_low_field(high_field_file, low_field_path, field_config)
        low_field_files.append(low_field_path)
    
    print("\n" + "="*80)
    print("✓ Conversion Complete!")
    print("="*80)
    print(f"\nHigh-field volumes: {len(high_field_files)}")
    print(f"  Location: {high_field_dir}")
    print(f"\nLow-field volumes: {len(low_field_files)}")
    print(f"  Location: {low_field_dir}")
    print("\nYou can now use these for training Stage 1!")


if __name__ == "__main__":
    main()




