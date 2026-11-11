#!/bin/bash
# Round-2 Phase 1 (no new feature extraction). Execute from the repository root.
# 0 tests; 1-3 stimulus statistics + analytic baselines; 4 empirical-alpha target;
# 5 leave-one-level-out; 6 simultaneous bounds. Probe protocol = final_probe.py, unchanged.
set -u
cd "$(dirname "$0")"
P=${PYTHON:-python3}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D=data/stimuli_v2; V3=results/features/v3; SD=results/features/v3sd; R=results/round2
mkdir -p logs $R
log() { echo "[$(date +%T)] $*"; }

log "commit $(git rev-parse --short HEAD); tests"
$P -m pytest -q tests/ > logs/round2_tests.log 2>&1 || { log "TESTS FAILED - stopping"; exit 1; }

log "items 1-3: stimulus statistics and baseline features"
$P scripts/r2_stimulus_stats.py --stimuli $D --feat-dir $R/features/base --out $R/phase1_stimulus.json \
   > logs/round2_stimulus.log 2>&1 || { log "STIMULUS STATS FAILED"; exit 1; }

# The 18 conditions for items 4 and 5: primary k=4, headroom k=8, ImageGPT, SD t=500 g128 seeds 0-4.
KEY=()
for m in clip mae dino supvit_in1k supvit_21k resnet50; do KEY+=("$V3/${m}_nearest_g56_main.npz" "$V3/${m}_nearest_g28_main.npz"); done
KEY+=("$V3/imagegpt_nearest_g32_main.npz")
for s in 0 1 2 3 4; do KEY+=("$SD/sd15_mid_t500_s${s}_nearest_g128_main.npz"); done

log "item 4: empirical-alpha target files"
$P scripts/r2_empirical_target.py "${KEY[@]}" --stimuli $D --out-dir $R/features/emp \
   > logs/round2_emp.log 2>&1 || { log "EMPIRICAL TARGET FAILED"; exit 1; }

log "items 3-4: probes (final_probe.py, 10 x 5 nested CV)"
probe() { $P scripts/final_probe.py "$@" --n-jobs 3 >> logs/round2_probe.log 2>&1 || log "PROBE FAILED $1"; }
export -f probe log; export P
{
  for f in $R/features/base/*.npz; do echo "$f --out-dir $R/oof_base"; done
  for f in $R/features/emp/*.npz; do echo "$f --target-key alpha_check --out-dir $R/oof_emp"; done
} | xargs -P 4 -I{} bash -c 'probe {}'

log "statistics for items 3-4 (final_stats.py)"
$P scripts/final_stats.py --oof-dir $R/oof_base --out $R/base_stats.json > logs/round2_stats_base.log 2>&1 || log "STATS FAILED base"
$P scripts/final_stats.py --oof-dir $R/oof_emp --out $R/emp_stats.json > logs/round2_stats_emp.log 2>&1 || log "STATS FAILED emp"

log "item 5: leave-one-level-out"
$P scripts/r2_lolo.py "${KEY[@]}" --out $R/lolo.json --n-jobs 12 > logs/round2_lolo.log 2>&1 || log "LOLO FAILED"

log "item 6: simultaneous bounds (replays final_stats.py and verifies against final_stats.json)"
$P scripts/r2_simultaneous.py --oof-dir results/final/oof --stats results/final/final_stats.json \
   --out $R/simultaneous.json > logs/round2_simultaneous.log 2>&1 || log "SIMULTANEOUS FAILED"

log "PHASE 1 DONE; probe failures: $(grep -c 'PROBE FAILED' logs/round2_run.log 2>/dev/null || echo 0); tracebacks: $(cat logs/round2_*.log | grep -c Traceback)"
