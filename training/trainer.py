"""
Training pipeline for biomarker estimation models.
Supports both the Hybrid Transformer-GNN and baseline models.
Includes per-biomarker metrics for multi-output regression.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
from pathlib import Path
import logging
from tqdm import tqdm


class Trainer:
    """Main trainer class for model training"""

    def __init__(self, config: Dict, model: nn.Module = None, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        self._setup_logging()

        # Build default model if none provided
        if self.model is None:
            self._build_model()
        else:
            self.model = self.model.to(self.device)

        self._setup_optimizer()
        self._setup_criterion()

    def _setup_logging(self):
        log_dir = Path(self.config['training']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, self.config.get('logging', {}).get('level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _build_model(self):
        from models.transformer_gnn.hybrid_model import HybridTransformerGNN
        self.model = HybridTransformerGNN(self.config).to(self.device)
        self.logger.info(
            f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters"
        )

    def _setup_optimizer(self):
        train_cfg = self.config['training']
        lr = float(train_cfg.get('learning_rate', 5e-4))
        weight_decay = float(train_cfg.get('weight_decay', 1e-5))

        if train_cfg.get('optimizer', 'adamw') == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )

        if train_cfg.get('scheduler') == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=train_cfg['epochs']
            )
        elif train_cfg.get('scheduler') == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=train_cfg['epochs'] // 3, gamma=0.1
            )

    def _setup_criterion(self):
        loss_name = self.config.get('biomarkers', {}).get('loss_function', 'mse')
        if loss_name == "mse":
            self.criterion = nn.MSELoss()
        elif loss_name == "mae":
            self.criterion = nn.L1Loss()
        elif loss_name == "huber":
            self.criterion = nn.HuberLoss()
        else:
            self.criterion = nn.MSELoss()

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            volumes = batch['volume'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(volumes)
            loss = self.criterion(predictions, labels)
            loss.backward()

            if self.config['training'].get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / max(num_batches, 1)

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                volumes = batch['volume'].to(self.device)
                labels = batch['label'].to(self.device)

                predictions = self.model(volumes)
                loss = self.criterion(predictions, labels)

                total_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())

        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Aggregate metrics
        mse = nn.MSELoss()(all_predictions, all_labels).item()
        mae = nn.L1Loss()(all_predictions, all_labels).item()

        metrics = {
            'loss': total_loss / max(len(val_loader), 1),
            'mse': mse,
            'mae': mae,
        }

        # Per-biomarker metrics
        biomarker_names = self.config.get('biomarkers', {}).get(
            'target_biomarkers', []
        )
        num_outputs = all_predictions.shape[1] if all_predictions.ndim > 1 else 1

        for i in range(num_outputs):
            name = biomarker_names[i] if i < len(biomarker_names) else f"bio_{i}"
            pred_i = all_predictions[:, i] if all_predictions.ndim > 1 else all_predictions
            label_i = all_labels[:, i] if all_labels.ndim > 1 else all_labels

            metrics[f'{name}_mse'] = nn.MSELoss()(pred_i, label_i).item()
            metrics[f'{name}_mae'] = nn.L1Loss()(pred_i, label_i).item()

            ss_res = ((label_i - pred_i) ** 2).sum()
            ss_tot = ((label_i - label_i.mean()) ** 2).sum()
            r2 = (1 - (ss_res / ss_tot)).item() if ss_tot > 0 else 0.0
            metrics[f'{name}_r2'] = r2

        return metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(ckpt, checkpoint_path)

        if is_best:
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(ckpt, best_path)
            self.logger.info(f"Saved best model at epoch {epoch}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None
    ):
        num_epochs = num_epochs or self.config['training']['epochs']
        patience = self.config['training']['early_stopping']['patience']

        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            train_loss = self.train_epoch(train_loader)
            self.logger.info(f"Train Loss: {train_loss:.4f}")

            val_metrics = self.validate(val_loader)
            self.logger.info(f"Val Loss: {val_metrics['loss']:.4f} | "
                             f"MSE: {val_metrics['mse']:.4f} | MAE: {val_metrics['mae']:.4f}")

            if self.scheduler:
                self.scheduler.step()

            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if (epoch + 1) % self.config['training'].get('save_frequency', 5) == 0 or is_best:
                self.save_checkpoint(epoch + 1, is_best)

            if self.patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        self.logger.info("Training completed")
