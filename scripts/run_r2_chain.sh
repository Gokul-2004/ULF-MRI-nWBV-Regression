#!/bin/bash
# Wait for Swin-UNETR (R2.2), then run SimMIM (R2.3). CPU-bound; avoid parallel thrash.
set -u
cd "$(dirname "$0")/.."
LOG=experiments/r2_chain.log
echo "[$(date +%T)] R2 CHAIN START" >> "$LOG"
if [ "${1:-}" != "" ]; then
  echo "[$(date +%T)] waiting for Swin-UNETR PID $1" >> "$LOG"
  while kill -0 "$1" 2>/dev/null; do sleep 30; done
  echo "[$(date +%T)] R2.2 Swin-UNETR DONE" >> "$LOG"
fi
echo "[$(date +%T)] R2.3 SimMIM START" >> "$LOG"
python3 -u scripts/ssl_comparator_simmim.py >> experiments/ssl_simmim_run.log 2>&1
echo "[$(date +%T)] R2.3 SimMIM DONE (exit $?)" >> "$LOG"
echo "[$(date +%T)] R2 CHAIN COMPLETE" >> "$LOG"
