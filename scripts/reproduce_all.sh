#!/usr/bin/env bash
# Reproduce the results of "Direct nWBV Regression from 64 mT Ultra-Low-Field MRI"
# (IEEE Access resubmission of Access-2026-45125). See REPRODUCE.md.
#
#   scripts/reproduce_all.sh build        # tables, figures, numbers from committed results (minutes; no data)
#   scripts/reproduce_all.sh experiments  # re-run round-3 evaluations (needs data/ + checkpoints; ~2 h CPU)
#   scripts/reproduce_all.sh retrain      # + matched physics-vs-blur retraining E4 and CNN3D (~9 h CPU)
#   scripts/reproduce_all.sh all          # experiments + retrain + build
#
# Every step overwrites its experiments/<name>/results.json; `build` then reads them.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
R3="$ROOT/scripts/round3"
LEVEL="${1:-build}"

say() { echo; echo "=== [$(date '+%F %T')] $* ==="; }

need() {
  for p in "$@"; do
    [ -e "$ROOT/$p" ] || { echo "Missing: $p  (see REPRODUCE.md, 'Inputs')"; exit 1; }
  done
}

build() {
  say "paper_build: tables, figures and numbers from experiments/*.json"
  python3 -m paper_build --strict
  echo "Report: paper_build/out/BUILD_REPORT.md"
}

experiments() {
  need data/ds006557_data checkpoints/oasis_finetuned.pt checkpoints/swin_oasis.pt \
       checkpoints/real64mt_finetuned.pt "Website_dat/extracted"
  cd "$R3"
  say "E1 fold-independent reproducibility + nuisance";  python3 -u e1_fold_independent_reproducibility.py
  say "E2 cross-session linear/non-neural readouts";     rm -f "$ROOT/experiments/r3_e2_nonneural_baselines/features.npz"
                                                         python3 -u e2_nonneural_baselines.py
  say "E2b Swin readout stress tests (~12 min)";         python3 -u e2b_swin_readout_verification.py
  say "Existing-data analyses (R2.2, R2.11, R4.3, E5)";  python3 -u a_existing_data_analyses.py
  say "E3 external cohort (trains 5 all-23 adapters if absent)"; python3 -u e3_external_headline_adapter.py
  say "E7 conformal per cohort";                         python3 -u e7_conformal_per_cohort.py
  say "E6 time budget";                                  python3 -u e6_time_budget.py
  if [ -e "$ROOT/checkpoints/r3_cnn3d_oasis.pt" ]; then
    say "E2c CNN3D target-domain readouts (saved checkpoint)"; python3 -u e2c_cnn3d_target_domain.py
    say "E8 OASIS-1 test on fixed seeded inputs";        python3 -u e8_oasis_fixed_inputs.py
  else
    echo "checkpoints/r3_cnn3d_oasis.pt absent: run 'retrain' for E2c and E8"
  fi
  cd "$ROOT"
}

retrain() {
  need data/high_field data/low_field data/oasis_processed data/oasis_raw data/ds006557_data
  cd "$R3"
  for seed in 42 1 7; do
    for arm in physics blur; do
      say "E4 $arm seed $seed (~75 min)"
      python3 -u e4_matched_physics_vs_blur.py --arm "$arm" --seed "$seed"
    done
  done
  python3 -u e4_matched_physics_vs_blur.py --aggregate
  if [ ! -e "$ROOT/checkpoints/r3_cnn3d_oasis.pt" ]; then
    say "E2c CNN3D retrain (Table-II recipe) + target-domain readouts"; python3 -u e2c_cnn3d_target_domain.py
    say "E8 OASIS-1 test on fixed seeded inputs";                       python3 -u e8_oasis_fixed_inputs.py
  fi
  cd "$ROOT"
}

case "$LEVEL" in
  build)       build ;;
  experiments) experiments; build ;;
  retrain)     retrain; build ;;
  all)         experiments; retrain; build ;;
  *) echo "usage: $0 {build|experiments|retrain|all}"; exit 2 ;;
esac
