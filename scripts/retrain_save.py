"""
Quick retrain script — trains ViT model and saves checkpoint.
"""
import sys
import yaml
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_utils import create_data_loaders
from utils.biomarker_extraction import extract_biomarkers_for_dataset
from models.baselines import BaselineViT3D
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn

# Load config
with open(PROJECT_ROOT / "configs" / "config.yaml") as f:
    config = yaml.safe_load(f)

config['training']['epochs'] = 50
config['training']['batch_size'] = 4
HIGH_FIELD_DIR = str(PROJECT_ROOT / "data" / "high_field")
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

print("Extracting biomarkers...")
biomarker_labels = extract_biomarkers_for_dataset(HIGH_FIELD_DIR)
print(f"Got labels for {len(biomarker_labels)} volumes")
# Make sure config has correct data paths
config['data']['high_field_dir'] = HIGH_FIELD_DIR
config['data']['low_field_dir'] = str(PROJECT_ROOT / "data" / "low_field")

print("Creating data loaders...")
train_loader, val_loader, test_loader = create_data_loaders(config)
print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

print("Building ViT model...")
img_size = tuple(config.get('data', {}).get('image_size', [64, 64, 64]))
num_classes = config.get('biomarkers', {}).get('num_classes', 4)
model = BaselineViT3D(
    img_size=img_size,
    patch_size=16,
    num_classes=num_classes,
    embed_dim=256,
    num_layers=4,
    num_heads=8
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Device: {device}, Params: {sum(p.numel() for p in model.parameters()):,}")

optimizer = Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=50)
criterion = nn.MSELoss()

best_val_loss = float('inf')
patience = 10
patience_counter = 0

print("\nTraining...")
for epoch in range(50):
    # Train
    model.train()
    train_losses = []
    for batch in train_loader:
        volumes = batch['volume'].to(device)
        labels  = batch['label'].to(device)
        optimizer.zero_grad()
        preds = model(volumes)
        loss  = criterion(preds, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())
    scheduler.step()

    # Validate
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            volumes = batch['volume'].to(device)
            labels  = batch['label'].to(device)
            preds   = model(volumes)
            loss    = criterion(preds, labels)
            val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss   = np.mean(val_losses)
    print(f"Epoch {epoch+1:3d}/50 | Train: {train_loss:.4f} | Val: {val_loss:.4f}", end="")

    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss
        patience_counter = 0
        ckpt = {
            'epoch': epoch + 1,
            'model_name': 'vit',
            'model_state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            'config': config
        }
        torch.save(ckpt, CHECKPOINT_DIR / "best_model.pt")
        print(" ← best", end="")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    print()

print(f"\nDone. Best val loss: {best_val_loss:.4f}")
print(f"Checkpoint saved to: {CHECKPOINT_DIR / 'best_model.pt'}")
