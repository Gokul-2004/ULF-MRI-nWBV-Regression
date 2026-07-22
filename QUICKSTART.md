# Quick Start Guide

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify setup:**
```bash
python test_pipeline.py
# or
python3 test_pipeline.py
```

## Running Stage 1

### Option 1: With Synthetic Data (for testing)
```bash
python main.py --stage stage1 --use-synthetic
```

### Option 2: With Real Data
```bash
python main.py --stage stage1 --data-dir /path/to/your/data
```

### Option 3: Direct Stage 1 Script
```bash
python experiments/stage1/run_stage1.py --use-synthetic
```

## What Stage 1 Does

1. **Data Loading**: Creates train/val/test splits
   - Uses synthetic data if `--use-synthetic` flag is set
   - Otherwise looks for data in `data/processed/`

2. **Field Conversion Setup**: Prepares high-field to low-field conversion pipeline

3. **Model Training**:
   - Trains hybrid Transformer-GNN model
   - Trains baseline models (CNN, ViT, U-Net)

4. **Evaluation & Comparison**:
   - Evaluates all models on test set
   - Generates comparison report
   - Saves results to `experiments/stage1/results/`

## Expected Output

After running, you should see:
- Training progress for each model
- Evaluation metrics (MSE, MAE, R², etc.)
- Comparison report saved to `experiments/stage1/results/report.txt`
- CSV file with detailed metrics: `experiments/stage1/results/comparison.csv`

## Data Format

If using real data, place your MRI files in `data/processed/` with:
- Supported formats: `.nii`, `.nii.gz`, `.npy`, `.npz`
- Create `metadata.json` with biomarker labels (optional)
- Or the system will auto-discover files and use placeholder labels

Example `metadata.json`:
```json
[
  {
    "id": "sample_001",
    "path": "data/processed/sample_001.nii.gz",
    "biomarker": 0.75
  }
]
```

## Troubleshooting

- **Import errors**: Make sure you're in the project root directory
- **CUDA errors**: The system will fall back to CPU automatically
- **Memory errors**: Reduce batch size in `configs/config.yaml`
- **No data found**: Use `--use-synthetic` flag for testing

## Next Steps

After Stage 1 completes successfully:
- Review comparison results in `experiments/stage1/results/`
- Check model checkpoints in `checkpoints/`
- Proceed to Stage 2 (when implemented)

