"""
Generate realistic synthetic high-field MRI volumes
Creates brain-like structures with realistic intensity patterns
"""

import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


def generate_brain_like_volume(size=(256, 256, 256), complexity='high'):
    """
    Generate a realistic brain-like MRI volume
    
    Args:
        size: Tuple of (depth, height, width)
        complexity: 'low', 'medium', or 'high' - controls detail level
    
    Returns:
        numpy array of shape size with values in [0, 1]
    """
    print(f"Generating {complexity} complexity brain-like volume {size}...")
    
    volume = np.zeros(size, dtype=np.float32)
    center = np.array(size) // 2
    
    # Parameters based on complexity
    if complexity == 'high':
        num_structures = 8
        noise_level = 0.01
        detail_level = 3
    elif complexity == 'medium':
        num_structures = 5
        noise_level = 0.02
        detail_level = 2
    else:  # low
        num_structures = 3
        noise_level = 0.03
        detail_level = 1
    
    # Create multiple brain-like structures
    for struct_idx in range(num_structures):
        # Random offset for each structure
        offset = np.random.uniform(-0.3, 0.3, 3) * np.array(size)
        struct_center = center + offset
        
        # Random size for each structure
        radius = np.random.uniform(0.15, 0.35) * max(size)
        
        # Create structure
        for i in tqdm(range(size[0]), desc=f"Structure {struct_idx+1}/{num_structures}", leave=False):
            for j in range(size[1]):
                for k in range(size[2]):
                    # Distance from structure center
                    dist = np.sqrt(
                        (i - struct_center[0])**2 + 
                        (j - struct_center[1])**2 + 
                        (k - struct_center[2])**2
                    )
                    
                    if dist < radius:
                        # White matter (higher intensity, 0.6-0.9)
                        if dist < radius * 0.4:
                            intensity = 0.7 + 0.2 * np.exp(-dist / (radius * 0.3))
                        # Gray matter (medium intensity, 0.3-0.6)
                        elif dist < radius * 0.7:
                            intensity = 0.4 + 0.2 * np.exp(-dist / (radius * 0.5))
                        # CSF/background (lower intensity, 0.1-0.3)
                        else:
                            intensity = 0.15 + 0.15 * np.exp(-dist / (radius * 0.8))
                        
                        # Add anatomical detail
                        if detail_level >= 2:
                            # Add cortical folds
                            intensity += 0.05 * (
                                np.sin(i * 0.1) * np.cos(j * 0.1) * 
                                np.sin(k * 0.1)
                            )
                        
                        if detail_level >= 3:
                            # Add fine structures
                            intensity += 0.03 * (
                                np.sin(i * 0.2) * np.cos(j * 0.15) * 
                                np.sin(k * 0.18)
                            )
                        
                        # Combine with existing (take max for overlapping structures)
                        volume[i, j, k] = np.maximum(volume[i, j, k], intensity)
    
    # Add realistic high-field noise (very low, high SNR)
    noise = np.random.normal(0, noise_level, size)
    volume = volume + noise
    
    # Normalize to [0, 1]
    volume = np.clip(volume, 0, 1)
    
    # Add some intensity variations (simulating different tissue types)
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                if volume[i, j, k] > 0.1:  # Only for tissue regions
                    # Add subtle variations
                    variation = 0.05 * np.sin(i * 0.05 + j * 0.07 + k * 0.06)
                    volume[i, j, k] = np.clip(volume[i, j, k] + variation, 0, 1)
    
    return volume


def generate_multiple_volumes(output_dir, num_volumes=5, size=(256, 256, 256), complexity='high'):
    """Generate multiple high-field MRI volumes"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_volumes} high-field MRI volumes...")
    print(f"Output directory: {output_dir}")
    print(f"Size: {size}")
    print(f"Complexity: {complexity}")
    print("="*80)
    
    volumes = []
    for i in range(num_volumes):
        print(f"\nGenerating volume {i+1}/{num_volumes}...")
        volume = generate_brain_like_volume(size, complexity)
        
        # Save volume
        output_path = output_dir / f"high_field_volume_{i+1:03d}.npy"
        np.save(output_path, volume)
        volumes.append(output_path)
        
        print(f"  Saved: {output_path}")
        print(f"  Shape: {volume.shape}")
        print(f"  Range: [{volume.min():.3f}, {volume.max():.3f}]")
        print(f"  Mean: {volume.mean():.3f}, Std: {volume.std():.3f}")
    
    print(f"\n✓ Generated {num_volumes} high-field volumes")
    print(f"  Location: {output_dir}")
    
    return volumes


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic high-field MRI volumes'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/high_field',
        help='Output directory for high-field volumes'
    )
    parser.add_argument(
        '--num-volumes',
        type=int,
        default=10,
        help='Number of volumes to generate'
    )
    parser.add_argument(
        '--size',
        type=int,
        nargs=3,
        default=[256, 256, 256],
        help='Volume size (depth height width)'
    )
    parser.add_argument(
        '--complexity',
        type=str,
        choices=['low', 'medium', 'high'],
        default='high',
        help='Complexity level of generated volumes'
    )
    
    args = parser.parse_args()
    
    size = tuple(args.size)
    
    volumes = generate_multiple_volumes(
        args.output_dir,
        num_volumes=args.num_volumes,
        size=size,
        complexity=args.complexity
    )
    
    print("\n" + "="*80)
    print("High-field volumes ready for conversion!")
    print("="*80)
    print(f"\nTo convert to low-field, run:")
    print(f"  python3 scripts/convert_high_to_low.py --input {args.output_dir} --output data/low_field")


if __name__ == "__main__":
    main()




