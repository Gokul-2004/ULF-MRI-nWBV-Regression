"""
Stage 1: Biomarker Inference from Low-Field MRI

Pipeline:
1. Extract biomarker labels from high-field MRI volumes
2. Create paired data loaders (low-field input -> biomarker labels)
3. Train baseline models (CNN, ViT, U-Net) and hybrid Transformer-GNN
4. Evaluate all models with per-biomarker metrics
5. Generate comparison report
"""

import torch
import torch.nn as nn
import yaml
import argparse
import traceback
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.biomarker_extraction import extract_biomarkers_for_dataset
from utils.data_utils import create_data_loaders
from models.transformer_gnn.hybrid_model import HybridTransformerGNN
from models.baselines import BaselineCNN3D, BaselineViT3D, BaselineUNet3D
from evaluation.comparison import Evaluator


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_baseline_model(config, model, model_name, train_loader, val_loader, device):
    """Train a single baseline model."""
    print(f"  Training {model_name}...")

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs']
    )

    best_val_loss = float('inf')
    patience = config['training']['early_stopping']['patience']
    patience_counter = 0
    num_epochs = min(config['training']['epochs'], 30)

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            volumes = batch['volume'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            predictions = model(volumes)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                volumes = batch['volume'].to(device)
                labels = batch['label'].to(device)
                predictions = model(volumes)
                loss = criterion(predictions, labels)
                val_loss += loss.item()

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{num_epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    model.eval()
    return model


def train_hybrid_model(config, train_loader, val_loader, device):
    """Train the hybrid Transformer-GNN model."""
    print("  Training Hybrid Transformer-GNN...")

    model = HybridTransformerGNN(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"    Model parameters: {params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs']
    )

    best_val_loss = float('inf')
    patience = config['training']['early_stopping']['patience']
    patience_counter = 0
    num_epochs = min(config['training']['epochs'], 30)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            volumes = batch['volume'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            predictions = model(volumes)
            loss = criterion(predictions, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                volumes = batch['volume'].to(device)
                labels = batch['label'].to(device)
                predictions = model(volumes)
                loss = criterion(predictions, labels)
                val_loss += loss.item()

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{num_epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    model.eval()
    return model


def evaluate_model(model, test_loader, device, model_name):
    """Evaluate a single model and return predictions."""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            volumes = batch['volume'].to(device)
            labels = batch['label'].to(device)
            predictions = model(volumes)
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

    return {
        'predictions': torch.cat(all_predictions, dim=0),
        'labels': torch.cat(all_labels, dim=0),
    }


def main():
    parser = argparse.ArgumentParser(description='Stage 1: Biomarker Inference from Low-Field MRI')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--use-synthetic', action='store_true',
                        help='Use synthetic data for testing')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and only evaluate')
    parser.add_argument('--skip-hybrid', action='store_true',
                        help='Skip hybrid model training')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    device = torch.device(
        config['inference']['device'] if torch.cuda.is_available() else 'cpu'
    )

    print("=" * 80)
    print("Stage 1: Biomarker Inference from Low-Field MRI")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Image size: {config['data']['image_size']}")
    print(f"Biomarkers: {config['biomarkers']['target_biomarkers']}")
    print("=" * 80)

    # Step 1: Extract biomarkers
    print("\n[Step 1] Extracting biomarkers from high-field MRI data...")
    high_field_dir = config['data']['high_field_dir']
    if Path(high_field_dir).exists() and not args.use_synthetic:
        biomarker_labels = extract_biomarkers_for_dataset(high_field_dir)
        print(f"  Extracted biomarkers for {len(biomarker_labels)} volumes")
        for name, values in biomarker_labels.items():
            print(f"    {name}: BTF={values[0]:.4f}, TCR={values[1]:.4f}, "
                  f"VBR={values[2]:.4f}, MCI={values[3]:.4f}")
    else:
        print("  Using synthetic data (no high-field directory found)")
        biomarker_labels = {}

    # Step 2: Create data loaders
    print("\n[Step 2] Creating paired data loaders...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            config, use_synthetic=args.use_synthetic
        )
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Val samples:   {len(val_loader.dataset)}")
        print(f"  Test samples:  {len(test_loader.dataset)}")

        # Verify shapes
        sample = next(iter(train_loader))
        print(f"  Volume shape:  {list(sample['volume'].shape)}")
        print(f"  Label shape:   {list(sample['label'].shape)}")
    except Exception as e:
        print(f"  Error creating data loaders: {e}")
        traceback.print_exc()
        return

    models = {}
    image_size = tuple(config['data']['image_size'])
    num_classes = config['biomarkers']['num_classes']

    if not args.skip_training:
        # Step 3: Train baseline models
        print("\n[Step 3] Training baseline models...")

        # CNN
        try:
            cnn = BaselineCNN3D(image_size, num_classes=num_classes)
            models['CNN'] = train_baseline_model(
                config, cnn, 'CNN', train_loader, val_loader, device
            )
            print("  CNN trained successfully")
        except Exception as e:
            print(f"  CNN failed: {e}")
            traceback.print_exc()

        # ViT
        try:
            vit = BaselineViT3D(
                image_size, patch_size=16, num_classes=num_classes,
                embed_dim=256, num_layers=4, num_heads=8
            )
            models['ViT'] = train_baseline_model(
                config, vit, 'ViT', train_loader, val_loader, device
            )
            print("  ViT trained successfully")
        except Exception as e:
            print(f"  ViT failed: {e}")
            traceback.print_exc()

        # U-Net
        try:
            unet = BaselineUNet3D(
                in_channels=1, num_classes=num_classes,
                encoder_depth=3, decoder_depth=3
            )
            models['UNet'] = train_baseline_model(
                config, unet, 'UNet', train_loader, val_loader, device
            )
            print("  U-Net trained successfully")
        except Exception as e:
            print(f"  U-Net failed: {e}")
            traceback.print_exc()

        # Step 4: Train hybrid model
        if not args.skip_hybrid:
            print("\n[Step 4] Training Hybrid Transformer-GNN model...")
            try:
                hybrid = train_hybrid_model(config, train_loader, val_loader, device)
                models['Hybrid-TF-GNN'] = hybrid
                print("  Hybrid model trained successfully")
            except Exception as e:
                print(f"  Hybrid model failed: {e}")
                traceback.print_exc()
        else:
            print("\n[Step 4] Skipping hybrid model (--skip-hybrid)")

    # Step 5: Evaluate and compare
    if len(models) == 0:
        print("\nNo models available for evaluation.")
        return

    print(f"\n[Step 5] Evaluating {len(models)} models...")
    evaluator = Evaluator(config)
    model_results = {}

    for model_name, model in models.items():
        try:
            results = evaluate_model(model, test_loader, device, model_name)
            model_results[model_name] = results

            mse = nn.MSELoss()(results['predictions'], results['labels']).item()
            mae = nn.L1Loss()(results['predictions'], results['labels']).item()
            print(f"  {model_name:20s} - MSE: {mse:.4f}, MAE: {mae:.4f}")
        except Exception as e:
            print(f"  {model_name} evaluation failed: {e}")
            traceback.print_exc()

    if len(model_results) == 0:
        print("No models evaluated successfully.")
        return

    # Save ViT as best_model.pt (used by fine-tuning pipeline)
    if 'ViT' in models:
        ckpt_dir = Path(config['training']['checkpoint_dir'])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        vit_path = ckpt_dir / 'best_model.pt'
        torch.save({
            'epoch': config['training']['epochs'],
            'model_name': 'BaselineViT3D',
            'model_state_dict': models['ViT'].state_dict(),
        }, vit_path)
        print(f"  Saved ViT checkpoint → {vit_path}")

    # Generate comparison
    results_dir = Path(config.get('evaluation', {}).get('results_dir', 'experiments/stage1/results'))
    results_dir.mkdir(parents=True, exist_ok=True)

    comparison_df = evaluator.compare_models(
        model_results,
        save_path=str(results_dir / 'comparison.csv')
    )

    report = evaluator.generate_report(
        comparison_df,
        save_path=str(results_dir / 'report.txt')
    )

    print("\n" + report)
    print(f"\nResults saved to {results_dir}/")
    print("\nStage 1 completed successfully!")


if __name__ == "__main__":
    main()
