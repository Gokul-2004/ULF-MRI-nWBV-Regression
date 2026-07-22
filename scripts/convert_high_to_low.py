"""
Script to convert high-field MRI images to low-field
This script processes actual high-field images and converts them to simulated low-field
"""

import sys
from pathlib import Path
import numpy as np
import torch
import yaml
from tqdm import tqdm
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.field_conversion import FieldConverter
from utils.data_utils import MRIDataset


def generate_sample_high_field_volume(size=(256, 256, 256), save_path=None):
    """
    Generate a realistic sample high-field MRI volume for testing
    Creates a brain-like structure with realistic intensity patterns
    """
    print(f"Generating sample high-field MRI volume ({size})...")
    
    volume = np.zeros(size, dtype=np.float32)
    center = np.array(size) // 2
    
    # Create brain-like structures
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                # Distance from center
                dist = np.sqrt(
                    (i - center[0])**2 + 
                    (j - center[1])**2 + 
                    (k - center[2])**2
                )
                
                # Create brain-like intensity patterns
                # White matter (higher intensity)
                if dist < max(size) * 0.3:
                    intensity = 0.7 + 0.2 * np.exp(-dist / (max(size) * 0.15))
                # Gray matter (medium intensity)
                elif dist < max(size) * 0.4:
                    intensity = 0.4 + 0.2 * np.exp(-dist / (max(size) * 0.2))
                # CSF (lower intensity)
                else:
                    intensity = 0.1 + 0.1 * np.exp(-dist / (max(size) * 0.25))
                
                # Add anatomical structures
                intensity += 0.1 * np.sin(i * 0.05) * np.cos(j * 0.05) * np.sin(k * 0.05)
                
                # Add some realistic noise (high-field has good SNR)
                intensity += np.random.normal(0, 0.02)
                
                volume[i, j, k] = np.clip(intensity, 0, 1)
    
    if save_path:
        np.save(save_path, volume)
        print(f"Saved high-field volume to {save_path}")
    
    return volume


def load_high_field_image(image_path):
    """Load a high-field MRI image from file"""
    path = Path(image_path)
    
    if path.suffix == '.npy':
        volume = np.load(path)
    elif path.suffix == '.npz':
        data = np.load(path)
        volume = data['volume'] if 'volume' in data else data[list(data.keys())[0]]
    else:
        # Try to load as NIfTI
        try:
            import nibabel as nib
            img = nib.load(path)
            volume = img.get_fdata()
        except ImportError:
            raise ImportError("nibabel required for NIfTI files. Install with: pip install nibabel")
        except Exception as e:
            raise ValueError(f"Could not load image from {path}: {e}")
    
    # Normalize to [0, 1]
    volume = volume.astype(np.float32)
    v_min, v_max = volume.min(), volume.max()
    if v_max > v_min:
        volume = (volume - v_min) / (v_max - v_min)
    
    return volume


def convert_single_image(
    high_field_path,
    output_path,
    converter,
    method='combined'
):
    """Convert a single high-field image to low-field"""
    print(f"\nConverting: {high_field_path.name}")
    
    # Load high-field image
    if high_field_path.suffix == '.npy' or high_field_path.exists():
        high_field = load_high_field_image(high_field_path)
    else:
        print(f"  Generating sample volume (file not found)")
        high_field = generate_sample_high_field_volume()
    
    print(f"  High-field shape: {high_field.shape}")
    print(f"  High-field range: [{high_field.min():.3f}, {high_field.max():.3f}]")
    
    # Convert to low-field
    low_field = converter.convert(high_field, method=method)
    
    print(f"  Low-field shape: {low_field.shape}")
    print(f"  Low-field range: [{low_field.min():.3f}, {low_field.max():.3f}]")
    
    # Save low-field image
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, low_field)
    print(f"  Saved to: {output_path}")
    
    # Calculate quality metrics
    mse = np.mean((high_field - low_field) ** 2)
    psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
    
    print(f"  MSE: {mse:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")
    
    return {
        'high_field_path': str(high_field_path),
        'low_field_path': str(output_path),
        'method': method,
        'mse': float(mse),
        'psnr': float(psnr),
        'high_field_shape': list(high_field.shape),
        'low_field_shape': list(low_field.shape)
    }


def convert_directory(
    input_dir,
    output_dir,
    config,
    methods=None
):
    """Convert all high-field images in a directory"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    converter = FieldConverter(config)
    
    # Find all image files
    image_extensions = ['.nii', '.nii.gz', '.npy', '.npz']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_dir.glob(f'*{ext}')))
    
    if len(image_files) == 0:
        print(f"No image files found in {input_dir}")
        print("Generating a sample high-field image for conversion...")
        
        # Generate a sample
        sample_path = input_dir / 'sample_high_field.npy'
        high_field = generate_sample_high_field_volume()
        np.save(sample_path, high_field)
        image_files = [sample_path]
    
    methods = methods or ['combined']
    results = []
    
    for image_file in tqdm(image_files, desc="Converting images"):
        for method in methods:
            output_name = f"{image_file.stem}_low_field_{method}.npy"
            output_path = output_dir / output_name
            
            result = convert_single_image(
                image_file,
                output_path,
                converter,
                method=method
            )
            results.append(result)
    
    # Save conversion log
    log_path = output_dir / 'conversion_log.json'
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nConversion log saved to: {log_path}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert high-field MRI images to low-field'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/high_field',
        help='Input directory or file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/low_field',
        help='Output directory'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Config file path'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='combined',
        choices=['simulation', 'degradation', 'combined'],
        help='Conversion method'
    )
    parser.add_argument(
        '--generate-sample',
        action='store_true',
        help='Generate a sample high-field image if none found'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    field_config = config.get('field_conversion', {
        'noise_level': 0.1,
        'resolution_factor': 0.5,
        'contrast_reduction': 0.3,
        'method': 'simulation'
    })
    
    print("="*80)
    print("High-Field to Low-Field Conversion")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Method: {args.method}")
    print(f"Config: {field_config}")
    print("="*80)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file conversion
        output_path = Path(args.output)
        converter = FieldConverter(field_config)
        result = convert_single_image(
            input_path,
            output_path,
            converter,
            method=args.method
        )
        print("\n✓ Conversion complete!")
    else:
        # Directory conversion
        methods = [args.method] if args.method else ['combined']
        results = convert_directory(
            args.input,
            args.output,
            field_config,
            methods=methods
        )
        print(f"\n✓ Converted {len(results)} image(s)")
        print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()




