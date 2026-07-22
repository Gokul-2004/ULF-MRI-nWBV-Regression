# Stage 1: What Happens?

## Overview

Stage 1 is the **High-Field to Low-Field Conversion and Comparison** stage. It validates the entire pipeline by:
1. Using high-field MRI data (real or synthetic)
2. Converting it to simulate low-field characteristics
3. Training models to estimate biomarkers
4. Comparing the hybrid Transformer-GNN model with baseline methods

---

## Step-by-Step Process

### **Step 1: Data Loading** 📂
**What happens:**
- Loads high-field MRI volumes from `data/high_field/` or generates synthetic data
- Splits data into:
  - **Training set** (70%) - for learning
  - **Validation set** (15%) - for tuning
  - **Test set** (15%) - for final evaluation
- Creates data loaders with proper batching

**Output:**
- Train/val/test data loaders ready for training

---

### **Step 2: Field Conversion Setup** 🔄
**What happens:**
- Sets up the high-field to low-field conversion pipeline
- Configures conversion parameters:
  - Noise levels
  - Resolution downsampling
  - Contrast reduction
- (Note: We already converted the data, so this is mainly setup)

**Output:**
- Conversion pipeline ready (data already converted)

---

### **Step 3: Train Hybrid Transformer-GNN Model** 🧠
**What happens:**
- Creates the hybrid model:
  - **Transformer branch**: Extracts spatial features from patches
  - **GNN branch**: Models structural relationships
  - **Fusion module**: Combines both modalities
- Trains the model:
  - Forward pass through the network
  - Computes loss (MSE for biomarker prediction)
  - Backpropagation to update weights
  - Validation after each epoch
  - Early stopping if no improvement
  - Saves best model checkpoint

**Output:**
- Trained hybrid model saved in `checkpoints/best_model.pt`
- Training logs in `logs/training.log`

**Time:** ~30-60 minutes (depending on data size and epochs)

---

### **Step 4: Train Baseline Models** 📊
**What happens:**
- Trains multiple baseline models for comparison:
  - **CNN (3D ResNet)**: Traditional convolutional approach
  - **ViT (Vision Transformer)**: Transformer-only approach
  - **U-Net**: Segmentation-style architecture
- Each model is trained separately with:
  - Same training data
  - Same validation data
  - Similar training procedure (but faster, fewer epochs)

**Output:**
- Trained baseline models (in memory, for comparison)

**Time:** ~10-20 minutes per baseline model

---

### **Step 5: Evaluation and Comparison** 📈
**What happens:**
- Evaluates all models on the **test set**:
  - Makes predictions for each test sample
  - Computes metrics:
    - **MSE** (Mean Squared Error)
    - **MAE** (Mean Absolute Error)
    - **R²** (Coefficient of determination)
    - **Correlation** (with ground truth)
    - **Uncertainty** (if available)
- Compares all models:
  - Creates comparison table
  - Identifies best model for each metric
  - Generates detailed report

**Output:**
- Comparison CSV: `experiments/stage1/results/comparison.csv`
- Detailed report: `experiments/stage1/results/report.txt`
- JSON results: `experiments/stage1/results/comparison_detailed.json`

---

## What You Get

### **Models Trained:**
1. ✅ Hybrid Transformer-GNN (your main model)
2. ✅ Baseline CNN
3. ✅ Baseline ViT
4. ✅ Baseline U-Net

### **Results:**
- Performance metrics for each model
- Comparison showing which model is best
- Model checkpoints for future use
- Training logs and history

### **Files Created:**
```
experiments/stage1/results/
├── comparison.csv          # Comparison table
├── comparison_detailed.json # Detailed metrics
└── report.txt              # Human-readable report

checkpoints/
├── best_model.pt           # Best hybrid model
└── checkpoint_epoch_*.pt   # Other checkpoints

logs/
└── training.log            # Training history
```

---

## Example Output

After Stage 1 completes, you'll see something like:

```
================================================================================
COMPARISON RESULTS
================================================================================

Best MSE: Hybrid-Transformer-GNN (0.0234)
Best MAE: Hybrid-Transformer-GNN (0.1456)
Best R2: Hybrid-Transformer-GNN (0.8765)
Best Correlation: Hybrid-Transformer-GNN (0.9342)

Detailed Metrics:
Model                  MSE      MAE      R2       Correlation
Hybrid-Transformer-GNN 0.0234   0.1456   0.8765   0.9342
CNN                    0.0345   0.1892   0.8123   0.9012
ViT                    0.0289   0.1678   0.8456   0.9234
UNet                   0.0312   0.1756   0.8321   0.9156
```

---

## Time Estimates

| Step | Time | Description |
|------|------|-------------|
| Data Loading | 1-2 min | Fast, just organizing data |
| Field Conversion | Already done | We converted data earlier |
| Train Hybrid | 30-60 min | Main model training |
| Train Baselines | 20-40 min | 3 baseline models |
| Evaluation | 2-5 min | Testing and comparison |
| **Total** | **~1-2 hours** | Full Stage 1 run |

---

## What This Validates

✅ **Conversion Pipeline**: High-to-low field conversion works  
✅ **Model Architecture**: Hybrid Transformer-GNN can learn  
✅ **Training Pipeline**: End-to-end training works  
✅ **Baseline Comparison**: Shows if hybrid model is better  
✅ **Evaluation System**: Metrics and comparison work  

---

## Next Steps After Stage 1

Once Stage 1 completes successfully:

1. **Review Results**: Check which model performed best
2. **Analyze Metrics**: Understand model strengths/weaknesses
3. **Stage 2**: Move to real low-field data (if available)
4. **Optimization**: Improve model based on results
5. **Production**: Deploy the best model

---

## Running Stage 1

```bash
# With real data (what we have now)
python3 main.py --stage stage1 --data-dir data/high_field

# With synthetic data (faster for testing)
python3 main.py --stage stage1 --use-synthetic

# Skip training, just evaluate existing models
python3 main.py --stage stage1 --skip-training
```

---

**In Summary:** Stage 1 trains your hybrid model and baselines, then compares them to see which works best for biomarker estimation on low-field MRI data! 🚀




