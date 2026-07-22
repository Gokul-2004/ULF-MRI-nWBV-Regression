"""
Setup verification script to check if all dependencies and imports work correctly
"""

import sys
from pathlib import Path

def check_imports():
    """Check if all required modules can be imported"""
    print("Checking imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch not found: {e}")
        return False
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"✗ NumPy not found: {e}")
        return False
    
    try:
        import yaml
        print("✓ PyYAML")
    except ImportError as e:
        print(f"✗ PyYAML not found: {e}")
        return False
    
    # Check project modules
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    try:
        from models.transformer_gnn.hybrid_model import HybridTransformerGNN
        print("✓ HybridTransformerGNN model")
    except ImportError as e:
        print(f"✗ HybridTransformerGNN import failed: {e}")
        return False
    
    try:
        from utils.field_conversion import FieldConverter
        print("✓ FieldConverter")
    except ImportError as e:
        print(f"✗ FieldConverter import failed: {e}")
        return False
    
    try:
        from training.trainer import Trainer
        print("✓ Trainer")
    except ImportError as e:
        print(f"✗ Trainer import failed: {e}")
        return False
    
    try:
        from inference.inferencer import CostAwareInferencer
        print("✓ CostAwareInferencer")
    except ImportError as e:
        print(f"✗ CostAwareInferencer import failed: {e}")
        return False
    
    return True


def check_directories():
    """Check if required directories exist"""
    print("\nChecking directory structure...")
    
    required_dirs = [
        'data/raw',
        'data/processed',
        'data/high_field',
        'data/low_field',
        'checkpoints',
        'logs',
        'experiments/stage1',
        'configs'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} missing")
            all_exist = False
    
    return all_exist


def check_config_files():
    """Check if config files exist"""
    print("\nChecking configuration files...")
    
    config_files = [
        'configs/config.yaml',
        'configs/stage1_config.yaml'
    ]
    
    all_exist = True
    for config_file in config_files:
        path = Path(config_file)
        if path.exists():
            print(f"✓ {config_file}")
            
            # Try to load it
            try:
                import yaml
                with open(path, 'r') as f:
                    yaml.safe_load(f)
                print(f"  ✓ Valid YAML")
            except Exception as e:
                print(f"  ✗ Invalid YAML: {e}")
                all_exist = False
        else:
            print(f"✗ {config_file} missing")
            all_exist = False
    
    return all_exist


def main():
    print("=" * 60)
    print("Transformer-GNN Biomarker Estimation - Setup Check")
    print("=" * 60)
    
    checks = [
        ("Imports", check_imports),
        ("Directories", check_directories),
        ("Config Files", check_config_files)
    ]
    
    results = []
    for name, check_func in checks:
        result = check_func()
        results.append((name, result))
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n✓ All checks passed! Project is ready to use.")
        return 0
    else:
        print("\n✗ Some checks failed. Please install missing dependencies.")
        print("Run: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())

