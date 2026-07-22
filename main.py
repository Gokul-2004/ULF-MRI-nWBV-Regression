"""
Main entry point for the Biomarker Inference from Low-Field MRI System
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description='Biomarker Inference from Low-Field MRI'
    )
    parser.add_argument(
        '--stage', type=str, choices=['stage1', 'stage2'], default='stage1',
        help='Which stage to run'
    )
    parser.add_argument(
        '--config', type=str, default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--use-synthetic', action='store_true',
        help='Use synthetic data instead of real MRI'
    )
    parser.add_argument(
        '--skip-hybrid', action='store_true',
        help='Skip hybrid model training'
    )
    parser.add_argument(
        '--extract-biomarkers', action='store_true',
        help='Only extract biomarkers from high-field data (no training)'
    )

    args = parser.parse_args()

    if args.extract_biomarkers:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        from utils.biomarker_extraction import extract_biomarkers_for_dataset
        high_field_dir = config['data']['high_field_dir']
        labels = extract_biomarkers_for_dataset(high_field_dir)
        print(f"\nExtracted biomarkers for {len(labels)} volumes:")
        for name, values in labels.items():
            print(f"  {name}: BTF={values[0]:.4f}, TCR={values[1]:.4f}, "
                  f"VBR={values[2]:.4f}, MCI={values[3]:.4f}")
        return

    if args.stage == 'stage1':
        # Build stage1 argv
        stage1_args = ['run_stage1.py', '--config', args.config]
        if args.use_synthetic:
            stage1_args.append('--use-synthetic')
        if args.skip_hybrid:
            stage1_args.append('--skip-hybrid')

        original_argv = sys.argv
        sys.argv = stage1_args
        try:
            from experiments.stage1.run_stage1 import main as stage1_main
            stage1_main()
        finally:
            sys.argv = original_argv

    elif args.stage == 'stage2':
        print("Stage 2 not yet implemented")


if __name__ == "__main__":
    main()
