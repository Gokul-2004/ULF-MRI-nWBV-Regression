#!/bin/bash
# Wait for Swin retrain (saves swin_oasis.pt), then run the 64mT transfer probe.
set -u
cd "$(dirname "$0")/.."
LOG=experiments/transfer_chain.log
echo "[$(date +%T)] TRANSFER CHAIN START" >> "$LOG"
if [ "${1:-}" != "" ]; then
  echo "[$(date +%T)] waiting for Swin retrain PID $1" >> "$LOG"
  while kill -0 "$1" 2>/dev/null; do sleep 30; done
  echo "[$(date +%T)] Swin retrain DONE" >> "$LOG"
fi
# confirm checkpoint exists before probing
if [ -f checkpoints/swin_oasis.pt ]; then
  echo "[$(date +%T)] PROBE START" >> "$LOG"
  python3 -u scripts/transfer_probe_64mt.py >> experiments/transfer_probe_run.log 2>&1
  echo "[$(date +%T)] PROBE DONE (exit $?)" >> "$LOG"
else
  echo "[$(date +%T)] ERROR: swin_oasis.pt missing, probe skipped" >> "$LOG"
fi
echo "[$(date +%T)] TRANSFER CHAIN COMPLETE" >> "$LOG"
