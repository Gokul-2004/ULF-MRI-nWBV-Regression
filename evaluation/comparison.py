"""
Evaluation and comparison with baseline methods.
Supports per-biomarker metrics for multi-output regression.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

BIOMARKER_NAMES_DEFAULT = ["btf", "tcr", "vbr", "mci"]


class Evaluator:
    """Evaluator for comparing models"""

    def __init__(self, config: Dict):
        self.config = config
        results_dir = config.get('evaluation', {}).get('results_dir', 'experiments/stage1/results')
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.biomarker_names = config.get('biomarkers', {}).get(
            'target_biomarkers', BIOMARKER_NAMES_DEFAULT
        )

    def compute_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute evaluation metrics including per-biomarker breakdown."""
        predictions = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
        labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

        # Aggregate regression metrics
        mse = float(np.mean((predictions - labels) ** 2))
        mae = float(np.mean(np.abs(predictions - labels)))
        rmse = float(np.sqrt(mse))

        # R2 score (aggregate)
        ss_res = np.sum((labels - predictions) ** 2)
        ss_tot = np.sum((labels - np.mean(labels)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        if np.isnan(r2) or np.isinf(r2):
            r2 = 0.0

        # Correlation
        try:
            corr = np.corrcoef(predictions.flatten(), labels.flatten())[0, 1]
            correlation = float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            correlation = 0.0

        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'correlation': correlation,
        }

        # Per-biomarker metrics
        if predictions.ndim == 2 and predictions.shape[1] > 1:
            for i in range(predictions.shape[1]):
                name = self.biomarker_names[i] if i < len(self.biomarker_names) else f"bio_{i}"
                pred_i = predictions[:, i]
                label_i = labels[:, i]

                bm_mse = float(np.mean((pred_i - label_i) ** 2))
                bm_mae = float(np.mean(np.abs(pred_i - label_i)))

                ss_res_i = np.sum((label_i - pred_i) ** 2)
                ss_tot_i = np.sum((label_i - np.mean(label_i)) ** 2)
                bm_r2 = float(1 - ss_res_i / ss_tot_i) if ss_tot_i > 0 else 0.0
                if np.isnan(bm_r2) or np.isinf(bm_r2):
                    bm_r2 = 0.0

                metrics[f'{name}_mse'] = bm_mse
                metrics[f'{name}_mae'] = bm_mae
                metrics[f'{name}_r2'] = bm_r2

        # Uncertainty metrics
        if uncertainties is not None:
            uncertainties = uncertainties.cpu().numpy() if isinstance(uncertainties, torch.Tensor) else uncertainties
            metrics['mean_uncertainty'] = float(np.mean(uncertainties))
            metrics['std_uncertainty'] = float(np.std(uncertainties))
            metrics['uncertainty_calibration'] = self._compute_calibration(
                predictions, labels, uncertainties
            )

        return metrics

    def _compute_calibration(self, predictions, labels, uncertainties) -> float:
        errors = np.abs(predictions - labels).flatten()
        unc = uncertainties.flatten()
        try:
            corr = np.corrcoef(errors, unc)[0, 1]
            return float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0

    def compare_models(
        self,
        model_results: Dict[str, Dict],
        save_path: Optional[str] = None
    ):
        """Compare multiple models and return comparison data."""
        comparison_data = []

        for model_name, results in model_results.items():
            metrics = self.compute_metrics(
                results['predictions'],
                results['labels'],
                results.get('uncertainties')
            )
            metrics['model'] = model_name
            comparison_data.append(metrics)

        if HAS_PANDAS:
            df = pd.DataFrame(comparison_data)
            if save_path:
                df.to_csv(save_path, index=False)
                self._save_detailed_comparison(
                    model_results,
                    save_path.replace('.csv', '_detailed.json')
                )
            return df
        else:
            if save_path:
                json_path = save_path.replace('.csv', '.json')
                with open(json_path, 'w') as f:
                    json.dump(comparison_data, f, indent=2)
                self._save_detailed_comparison(
                    model_results,
                    save_path.replace('.csv', '_detailed.json')
                )
            return comparison_data

    def _save_detailed_comparison(self, model_results, save_path):
        detailed = {}
        for model_name, results in model_results.items():
            detailed[model_name] = {
                'num_samples': len(results['predictions']),
                'metrics': self.compute_metrics(
                    results['predictions'],
                    results['labels'],
                    results.get('uncertainties')
                )
            }
        with open(save_path, 'w') as f:
            json.dump(detailed, f, indent=2)

    def generate_report(
        self,
        comparison_df,
        save_path: Optional[str] = None
    ) -> str:
        """Generate text report from comparison."""
        lines = [
            "=" * 80,
            "Model Comparison Report — Biomarker Inference from Low-Field MRI",
            "=" * 80,
            "",
        ]

        if HAS_PANDAS and isinstance(comparison_df, pd.DataFrame):
            # Aggregate metrics
            agg_metrics = ['mse', 'mae', 'r2', 'correlation']
            for metric in agg_metrics:
                if metric in comparison_df.columns:
                    valid = comparison_df[metric].dropna()
                    if len(valid) > 0:
                        best_idx = valid.idxmin() if metric in ['mse', 'mae'] else valid.idxmax()
                        if not pd.isna(best_idx):
                            best_model = comparison_df.loc[best_idx, 'model']
                            best_val = comparison_df.loc[best_idx, metric]
                            lines.append(f"Best {metric.upper()}: {best_model} ({best_val:.4f})")

            lines.append("")

            # Per-biomarker breakdown
            lines.append("-" * 80)
            lines.append("Per-Biomarker Results:")
            lines.append("-" * 80)
            for bm in self.biomarker_names:
                mse_col = f'{bm}_mse'
                mae_col = f'{bm}_mae'
                r2_col = f'{bm}_r2'
                if mse_col in comparison_df.columns:
                    lines.append(f"\n  {bm.upper()}:")
                    for _, row in comparison_df.iterrows():
                        mse_v = row.get(mse_col, float('nan'))
                        mae_v = row.get(mae_col, float('nan'))
                        r2_v = row.get(r2_col, float('nan'))
                        lines.append(
                            f"    {row['model']:25s} MSE={mse_v:.4f}  MAE={mae_v:.4f}  R2={r2_v:.4f}"
                        )

            lines.append("")
            lines.append("=" * 80)
            lines.append("Full Metrics Table:")
            lines.append("=" * 80)
            # Show only key columns to keep it readable
            display_cols = ['model', 'mse', 'mae', 'rmse', 'r2', 'correlation']
            available_cols = [c for c in display_cols if c in comparison_df.columns]
            lines.append(comparison_df[available_cols].to_string(index=False))
        else:
            lines.append("Detailed Metrics:")
            lines.append("=" * 80)
            for item in comparison_df:
                lines.append(f"\nModel: {item.get('model', 'Unknown')}")
                for key, value in item.items():
                    if key != 'model' and isinstance(value, (int, float)):
                        lines.append(f"  {key}: {value:.4f}")

        report = "\n".join(lines)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)

        return report
