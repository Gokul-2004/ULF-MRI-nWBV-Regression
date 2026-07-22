"""
Inference pipeline with cost-aware and uncertainty-aware predictions
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
import numpy as np
import time
from pathlib import Path
import logging

from models.transformer_gnn.hybrid_model import HybridTransformerGNN
from models.uncertainty.uncertainty_estimator import get_uncertainty_estimator


class CostAwareInferencer:
    """Cost-aware inference with early exit and adaptive strategies"""
    
    def __init__(self, config: Dict, model: nn.Module, device: str = "cuda"):
        self.config = config
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Setup uncertainty estimator
        self.uncertainty_estimator = get_uncertainty_estimator(config)
        
        # Cost-aware settings
        self.cost_aware_config = config['inference']['cost_aware']
        self.min_confidence = self.cost_aware_config.get('min_confidence', 0.7)
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Make prediction with uncertainty estimation
        
        Returns:
            predictions: Model predictions
            uncertainty: Uncertainty estimates (if return_uncertainty=True)
        """
        x = x.to(self.device)
        
        if return_uncertainty:
            predictions, uncertainty = self.uncertainty_estimator.estimate(
                self.model,
                x
            )
            return predictions, uncertainty
        else:
            with torch.no_grad():
                predictions = self.model(x)
            return predictions, None
    
    def early_exit_predict(
        self,
        x: torch.Tensor,
        intermediate_layers: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Early exit prediction - stop at intermediate layer if confident enough
        
        Returns:
            predictions: Final predictions
            exit_layer: Layer at which prediction was made
        """
        # This is a simplified version
        # In practice, you'd modify the model to support intermediate outputs
        with torch.no_grad():
            predictions = self.model(x)
        
        # Estimate confidence (simplified - use prediction variance)
        confidence = 1.0 / (1.0 + predictions.std().item())
        
        if confidence >= self.min_confidence:
            return predictions, 0  # Exited early (at final layer for now)
        else:
            # Would continue to deeper layers in full implementation
            return predictions, -1
    
    def adaptive_sampling_predict(
        self,
        x: torch.Tensor,
        max_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Adaptive sampling - use more samples for uncertain predictions
        
        Returns:
            predictions: Mean predictions
            uncertainty: Uncertainty estimates
            num_samples_used: Number of samples actually used
        """
        # Start with fewer samples
        initial_samples = 3
        
        predictions_list = []
        for _ in range(initial_samples):
            with torch.no_grad():
                pred = self.model(x)
                predictions_list.append(pred)
        
        predictions = torch.stack(predictions_list)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Check if uncertainty is high
        uncertainty = std_pred.mean().item()
        num_samples_used = initial_samples
        
        if uncertainty > (1.0 - self.min_confidence):
            # High uncertainty - use more samples
            additional_samples = max_samples - initial_samples
            for _ in range(additional_samples):
                with torch.no_grad():
                    pred = self.model(x)
                    predictions_list.append(pred)
            
            predictions = torch.stack(predictions_list)
            mean_pred = predictions.mean(dim=0)
            std_pred = predictions.std(dim=0)
            num_samples_used = max_samples
        
        return mean_pred, std_pred, num_samples_used
    
    def predict_batch(
        self,
        batch: Dict,
        strategy: str = "standard"
    ) -> Dict:
        """
        Predict on a batch with specified strategy
        
        Args:
            batch: Input batch with 'volume' key
            strategy: Inference strategy ('standard', 'early_exit', 'adaptive')
        
        Returns:
            Dictionary with predictions, uncertainty, and metadata
        """
        volumes = batch['volume'].to(self.device)
        
        start_time = time.time()
        
        if strategy == "early_exit" and self.cost_aware_config['enabled']:
            predictions, exit_layer = self.early_exit_predict(volumes)
            uncertainty = None  # Would compute from intermediate layers
        elif strategy == "adaptive" and self.cost_aware_config['enabled']:
            predictions, uncertainty, num_samples = self.adaptive_sampling_predict(volumes)
        else:
            predictions, uncertainty = self.predict_with_uncertainty(volumes)
            num_samples = 1
        
        inference_time = time.time() - start_time
        
        results = {
            'predictions': predictions.cpu(),
            'uncertainty': uncertainty.cpu() if uncertainty is not None else None,
            'inference_time': inference_time,
            'strategy': strategy
        }
        
        if strategy == "adaptive":
            results['num_samples'] = num_samples
        
        return results
    
    def evaluate_loader(
        self,
        data_loader: DataLoader,
        strategy: str = "standard"
    ) -> Dict:
        """Evaluate on entire data loader"""
        all_predictions = []
        all_labels = []
        all_uncertainties = []
        total_time = 0.0
        
        for batch in data_loader:
            results = self.predict_batch(batch, strategy)
            
            all_predictions.append(results['predictions'])
            all_labels.append(batch['label'])
            if results['uncertainty'] is not None:
                all_uncertainties.append(results['uncertainty'])
            total_time += results['inference_time']
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute metrics
        mse = nn.MSELoss()(all_predictions, all_labels).item()
        mae = nn.L1Loss()(all_predictions, all_labels).item()
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'avg_inference_time': total_time / len(data_loader),
            'total_inference_time': total_time
        }
        
        if len(all_uncertainties) > 0:
            all_uncertainties = torch.cat(all_uncertainties, dim=0)
            metrics['mean_uncertainty'] = all_uncertainties.mean().item()
            metrics['std_uncertainty'] = all_uncertainties.std().item()
        
        return metrics


def load_model_for_inference(
    checkpoint_path: str,
    config: Dict,
    device: str = "cuda"
) -> nn.Module:
    """Load trained model for inference"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = HybridTransformerGNN(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

