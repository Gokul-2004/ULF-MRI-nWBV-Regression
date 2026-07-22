"""
Quick test script to verify the pipeline works
"""

import sys
from pathlib import Path
import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    try:
        from models.transformer_gnn.hybrid_model import HybridTransformerGNN
        from models.baselines import BaselineCNN3D, BaselineViT3D, BaselineUNet3D
        from utils.data_utils import create_data_loaders, MRIDataset
        from utils.field_conversion import FieldConverter
        from training.trainer import Trainer
        from inference.inferencer import CostAwareInferencer
        from evaluation.comparison import Evaluator
        print("✓ All core imports successful")
        
        # Check optional dependencies
        missing = []
        try:
            import nibabel
        except ImportError:
            missing.append("nibabel (optional, for NIfTI files)")
        
        try:
            import scipy
        except ImportError:
            missing.append("scipy (optional, for advanced field conversion)")
        
        if missing:
            print(f"  Note: Optional dependencies missing: {', '.join(missing)}")
            print("  Install with: pip install -r requirements.txt")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading with synthetic data"""
    print("\nTesting data loading...")
    try:
        from utils.data_utils import create_data_loaders
        import yaml
        
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Use smaller size for testing
        config['data']['image_size'] = [64, 64, 64]
        
        train_loader, val_loader, test_loader = create_data_loaders(
            config,
            'data/processed',
            use_synthetic=True
        )
        
        # Test a batch
        batch = next(iter(train_loader))
        print(f"✓ Data loading successful")
        print(f"  Batch shape: {batch['volume'].shape}")
        print(f"  Label shape: {batch['label'].shape}")
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    try:
        import yaml
        from models.transformer_gnn.hybrid_model import HybridTransformerGNN
        from models.baselines import BaselineCNN3D, BaselineViT3D, BaselineUNet3D
        
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Test hybrid model
        hybrid_model = HybridTransformerGNN(config)
        print(f"✓ Hybrid model created: {sum(p.numel() for p in hybrid_model.parameters())} parameters")
        
        # Test baseline models
        image_size = tuple(config['data']['image_size'])
        cnn = BaselineCNN3D(image_size, num_classes=1)
        print(f"✓ CNN model created: {sum(p.numel() for p in cnn.parameters())} parameters")
        
        vit = BaselineViT3D(image_size, num_classes=1, num_layers=6)
        print(f"✓ ViT model created: {sum(p.numel() for p in vit.parameters())} parameters")
        
        unet = BaselineUNet3D(num_classes=1, encoder_depth=3, decoder_depth=3)
        print(f"✓ U-Net model created: {sum(p.numel() for p in unet.parameters())} parameters")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test forward pass through models"""
    print("\nTesting forward pass...")
    try:
        import yaml
        from models.transformer_gnn.hybrid_model import HybridTransformerGNN
        from models.baselines import BaselineCNN3D
        
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = 1  # Reduced for memory
        # Use smaller image size for testing to avoid memory issues
        image_size = (64, 64, 64)  # Smaller than config for testing
        
        # Temporarily modify config for testing
        original_size = config['data']['image_size']
        config['data']['image_size'] = list(image_size)
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, 1, *image_size).to(device)
        
        # Test hybrid model
        hybrid_model = HybridTransformerGNN(config).to(device)
        hybrid_model.eval()
        with torch.no_grad():
            output = hybrid_model(dummy_input)
        print(f"✓ Hybrid model forward pass: input {dummy_input.shape} -> output {output.shape}")
        
        # Test CNN
        cnn = BaselineCNN3D(image_size, num_classes=1).to(device)
        cnn.eval()
        with torch.no_grad():
            output = cnn(dummy_input)
        print(f"✓ CNN forward pass: input {dummy_input.shape} -> output {output.shape}")
        
        # Restore config
        config['data']['image_size'] = original_size
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_field_conversion():
    """Test field conversion"""
    print("\nTesting field conversion...")
    try:
        from utils.field_conversion import FieldConverter
        import numpy as np
        
        config = {
            'noise_level': 0.1,
            'resolution_factor': 0.5,
            'contrast_reduction': 0.3,
            'method': 'simulation'
        }
        
        converter = FieldConverter(config)
        
        # Create dummy high-field volume (smaller for testing)
        high_field = np.random.rand(32, 32, 32).astype(np.float32)
        
        # Convert to low-field
        low_field = converter.convert(high_field)
        
        print(f"✓ Field conversion successful")
        print(f"  High-field shape: {high_field.shape}")
        print(f"  Low-field shape: {low_field.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Field conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*80)
    print("Pipeline Test Suite")
    print("="*80)
    
    tests = [
        ("Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Field Conversion", test_field_conversion),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} test crashed: {e}")
            results.append((name, False))
    
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Pipeline is ready to use.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
