#!/bin/bash
# Run MAE, DINO, contrastive sequentially (CPU-bound; avoid parallel thrash).
set -u
cd "$(dirname "$0")/.."
LOG=experiments/ssl_extra_chain.log
echo "[$(date +%T)] SSL CHAIN START" >> "$LOG"
for method in mae dino contrastive; do
  echo "[$(date +%T)] $method START" >> "$LOG"
  python3 -u scripts/ssl_comparators_extra.py $method >> experiments/ssl_${method}_run.log 2>&1
  echo "[$(date +%T)] $method DONE (exit $?)" >> "$LOG"
done
echo "[$(date +%T)] SSL CHAIN COMPLETE" >> "$LOG"
