# Scalable ML Infrastructure for Hybrid Transformer-GNN Biomarker Estimation

A scalable and uncertainty-aware ML infrastructure for deploying hybrid Transformer-GNN models to perform biomarker estimation on low-field MRI data. This project focuses on building and evaluating an end-to-end system that enables efficient training, cost-aware inference, and reliable uncertainty-aware predictions.

## Project Overview

This project addresses the challenge of deploying deep learning models for biomarker estimation in resource-constrained environments with low-field MRI data. The system combines:

- **Hybrid Architecture**: Vision Transformer for spatial feature extraction + Graph Neural Networks for structural relationships
- **Uncertainty Quantification**: Multiple methods (ensemble, dropout, evidential) for reliable predictions
- **Cost-Aware Inference**: Early exit, adaptive sampling, and model cascading strategies
- **High-to-Low Field Conversion**: Simulation pipeline for converting high-field MRI to low-field characteristics

## Project Structure

```
.
├── configs/                 # Configuration files
│   ├── config.yaml         # Main configuration
│   └── stage1_config.yaml  # Stage 1 specific config
├── data/                   # Data directories
│   ├── raw/                # Raw data
│   ├── processed/          # Processed data
│   ├── high_field/         # High-field MRI images
│   └── low_field/          # Low-field MRI images (simulated)
├── models/                 # Model architectures
│   ├── transformer_gnn/   # Hybrid model
│   └── uncertainty/        # Uncertainty estimation methods
├── training/               # Training pipeline
├── inference/              # Inference pipeline
├── evaluation/             # Evaluation and comparison
├── utils/                  # Utility functions
│   ├── data_utils.py       # Data loading utilities
│   └── field_conversion.py # High-to-low field conversion
├── experiments/            # Experiment scripts
│   └── stage1/            # Stage 1 experiments
├── checkpoints/            # Model checkpoints
├── logs/                   # Training logs
└── requirements.txt        # Python dependencies
```

## Stage 1: High-Field to Low-Field Conversion

The first stage focuses on:
1. Converting high-field MRI images to simulate low-field characteristics
2. Training the hybrid Transformer-GNN model on converted data
3. Comparing performance with baseline methods
4. Establishing baseline metrics for further development

## Installation

1. Clone the repository:
```bash
cd "/home/gk-krishnan/Desktop/VIT Paper"
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.

### Stage 1: High-Field to Low-Field Conversion and Comparison

**With synthetic data (for testing):**
```bash
python main.py --stage stage1 --use-synthetic
```

**With real data:**
```bash
python main.py --stage stage1 --data-dir /path/to/your/data
```

**Direct script:**
```bash
python experiments/stage1/run_stage1.py --use-synthetic
```

### Training

```python
from training.trainer import Trainer
from utils.data_utils import create_data_loaders
import yaml

# Load config
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(config, 'data/processed')

# Train model
trainer = Trainer(config)
trainer.train(train_loader, val_loader)
```

### Inference

```python
from inference.inferencer import CostAwareInferencer, load_model_for_inference
import yaml

# Load config and model
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = load_model_for_inference('checkpoints/best_model.pt', config)
inferencer = CostAwareInferencer(config, model)

# Make predictions with uncertainty
predictions, uncertainty = inferencer.predict_with_uncertainty(input_volume)
```

## Configuration

The system is highly configurable through YAML files:

- **Model Architecture**: Transformer and GNN dimensions, layers, fusion methods
- **Uncertainty Methods**: Ensemble, dropout, evidential, Bayesian
- **Training**: Optimizer, scheduler, early stopping
- **Inference**: Cost-aware strategies, uncertainty thresholds
- **Field Conversion**: Noise levels, resolution factors, contrast reduction

See `configs/config.yaml` for detailed configuration options.

## Key Features

### 1. Hybrid Transformer-GNN Architecture
- Vision Transformer for patch-based spatial feature extraction
- Graph Neural Network for modeling structural relationships
- Attention-based fusion of both modalities

### 2. Uncertainty Quantification
- **Ensemble**: Multiple model predictions
- **Monte Carlo Dropout**: Stochastic forward passes
- **Evidential Deep Learning**: Explicit uncertainty modeling
- **Bayesian**: (Planned) Full Bayesian inference

### 3. Cost-Aware Inference
- **Early Exit**: Stop computation when confidence is high
- **Adaptive Sampling**: Use more samples for uncertain predictions
- **Model Cascade**: (Planned) Progressive model complexity

### 4. High-to-Low Field Conversion
- Noise degradation simulation
- Resolution downsampling
- Contrast reduction
- Combined degradation strategies

## Evaluation Metrics

- **Regression Metrics**: MSE, MAE, RMSE, R², Correlation
- **Uncertainty Metrics**: Calibration, uncertainty correlation with error
- **Efficiency Metrics**: Inference time, compute cost
- **Comparison**: Baseline models (CNN, ViT, GCN, U-Net)

## Development Roadmap

- [x] Project structure and core architecture
- [x] High-field to low-field conversion pipeline
- [x] Hybrid Transformer-GNN model
- [x] Uncertainty quantification methods
- [x] Training and inference pipelines
- [ ] Data loading implementation
- [ ] Baseline model implementations
- [ ] Full evaluation pipeline
- [ ] Stage 2: Real low-field data integration
- [ ] Production deployment utilities

## Notes

- This is an **ML infrastructure project**, not a medical research project
- Focus is on scalable, production-ready ML systems
- Data handling is abstracted for flexibility
- System designed for resource-constrained environments

## License

[Add your license here]

## Citation

[Add citation if applicable]

