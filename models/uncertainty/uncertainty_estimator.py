"""
Uncertainty quantification methods for biomarker predictions
Supports ensemble, dropout, evidential, and Bayesian approaches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np


class UncertaintyEstimator:
    """Base class for uncertainty estimation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.method = config['uncertainty']['method']
    
    def estimate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        num_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate uncertainty for predictions
        
        Returns:
            predictions: Mean predictions
            uncertainty: Uncertainty estimates (std, entropy, etc.)
        """
        raise NotImplementedError


class EnsembleUncertainty(UncertaintyEstimator):
    """Ensemble-based uncertainty estimation"""
    
    def __init__(self, config: Dict, models: List[nn.Module]):
        super().__init__(config)
        self.models = models
    
    def estimate(
        self,
        model: Optional[nn.Module],
        x: torch.Tensor,
        num_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate uncertainty using model ensemble"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)  # [num_models, B, num_classes]
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred


class DropoutUncertainty(UncertaintyEstimator):
    """Monte Carlo Dropout uncertainty estimation"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.num_samples = config['uncertainty'].get('num_samples', 10)
        self.dropout_rate = config['uncertainty'].get('dropout_rate', 0.1)
    
    def estimate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        num_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate uncertainty using MC Dropout"""
        num_samples = num_samples or self.num_samples
        model.train()  # Enable dropout
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)  # [num_samples, B, num_classes]
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        model.eval()
        return mean_pred, std_pred


class EvidentialUncertainty(UncertaintyEstimator):
    """Evidential deep learning for uncertainty"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.temperature = config['uncertainty'].get('temperature', 1.0)
    
    def estimate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        num_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate uncertainty using evidential approach"""
        model.eval()
        with torch.no_grad():
            # Model outputs should be evidential parameters
            # (alpha, beta, nu) for regression
            evidential_params = model(x)
            
            # Extract parameters (assuming model outputs [alpha, beta, nu])
            alpha = evidential_params[:, 0:1] + 1.0
            beta = evidential_params[:, 1:2]
            nu = evidential_params[:, 2:3]
            
            # Mean prediction
            mean_pred = alpha / (alpha + beta)
            
            # Uncertainty (aleatoric + epistemic)
            aleatoric = beta / (alpha * (alpha + beta + 1))
            epistemic = beta / ((alpha + beta) * (nu + 1))
            total_uncertainty = aleatoric + epistemic
        
        return mean_pred, total_uncertainty


def get_uncertainty_estimator(config: Dict, models: Optional[List[nn.Module]] = None):
    """Factory function to get uncertainty estimator"""
    method = config['uncertainty']['method']
    
    if method == "ensemble":
        if models is None:
            raise ValueError("Ensemble method requires list of models")
        return EnsembleUncertainty(config, models)
    elif method == "dropout":
        return DropoutUncertainty(config)
    elif method == "evidential":
        return EvidentialUncertainty(config)
    elif method == "bayesian":
        # TODO: Implement Bayesian uncertainty
        raise NotImplementedError("Bayesian uncertainty not yet implemented")
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")

