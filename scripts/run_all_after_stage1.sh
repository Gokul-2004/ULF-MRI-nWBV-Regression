#!/bin/bash
# Run all experiments that depend on Stage 1 checkpoint.
# Execute this AFTER Stage 1 retrain completes.
#
# Usage: bash scripts/run_all_after_stage1.sh

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
LOGS="$PROJECT_ROOT/logs"
mkdir -p "$LOGS"

echo "============================================================"
echo "Post-Stage1 Experiments"
echo "============================================================"

# 1. OASIS fine-tuning with new best_model.pt
echo ""
echo "[1/4] OASIS Fine-tuning..."
python3 -u scripts/finetune_oasis.py \
    2>&1 | tee "$LOGS/oasis_finetune_v2.log"

# 2. Zenodo validation with new base model
echo ""
echo "[2/4] Zenodo Validation (base model)..."
python3 -u scripts/validate_zenodo.py \
    2>&1 | tee "$LOGS/zenodo_v2.log"

# 3. Real 64mT generalization with new fine-tuned model
echo ""
echo "[3/4] Real 64mT Evaluation..."
python3 -u scripts/evaluate_real_64mt.py \
    2>&1 | tee "$LOGS/real64mt_v2.log"

# 4. Ablation (trains from scratch, uses high_field/ data)
echo ""
echo "[4/4] Arnold Ablation..."
python3 -u scripts/ablation_arnold.py \
    2>&1 | tee "$LOGS/ablation_v2.log"

echo ""
echo "All experiments complete. Running results collection..."
python3 -u scripts/collect_paper_results.py 2>&1 | tee "$LOGS/final_results.log"
python3 -u scripts/paper_statistics.py 2>&1 | tee "$LOGS/paper_statistics.log"

echo ""
echo "============================================================"
echo "DONE. Results in experiments/ and logs/"
echo "Key files:"
echo "  experiments/oasis_finetune/finetune_results.json"
echo "  experiments/ablation_arnold/results.json"
echo "  experiments/real64mt_eval/predictions.json"
echo "  logs/final_results.log"
echo "  logs/paper_statistics.log"
echo "============================================================"
