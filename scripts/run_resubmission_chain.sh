#!/bin/bash
# Chain the 3 resubmission experiments sequentially (CPU-bound, avoid thrash).
# #1 adapter ablation is already running as PID passed in $1; wait for it,
# then run #2 multiseed and #3 sensitivity.
set -u
cd "$(dirname "$0")/.."
LOG=experiments/resubmission_chain.log
PYFLAGS="-u"   # unbuffered so progress is visible

echo "[$(date +%T)] CHAIN START" >> "$LOG"

# Wait for the already-running adapter ablation (PID in $1) if provided
if [ "${1:-}" != "" ]; then
  echo "[$(date +%T)] waiting for adapter ablation PID $1" >> "$LOG"
  while kill -0 "$1" 2>/dev/null; do sleep 15; done
  echo "[$(date +%T)] EXP1 adapter ablation DONE" >> "$LOG"
fi

echo "[$(date +%T)] EXP2 multiseed START" >> "$LOG"
python3 $PYFLAGS scripts/multiseed_loocv.py >> experiments/multiseed_loocv_run.log 2>&1
echo "[$(date +%T)] EXP2 multiseed DONE (exit $?)" >> "$LOG"

echo "[$(date +%T)] EXP3 sensitivity START" >> "$LOG"
python3 $PYFLAGS scripts/simulation_sensitivity.py >> experiments/simulation_sensitivity_run.log 2>&1
echo "[$(date +%T)] EXP3 sensitivity DONE (exit $?)" >> "$LOG"

echo "[$(date +%T)] CHAIN COMPLETE" >> "$LOG"
