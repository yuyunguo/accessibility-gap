#!/bin/bash
# Round-2 Phase 2: random-initialization baselines. Execute from the repository root on a CUDA machine.
# 1 random-init features (224-px models, 3 init seeds x k = 8/4/2); 2 random-UNet SD (pretrained VAE,
# init seeds 0-2 x noise seeds 0-4 x t = 100/500, g128); 3 probes (final_probe.py, unchanged); 4 statistics.
# Pretrained counterparts are already in results/final/final_stats.json (same stimuli, same protocol).
set -u
cd "$(dirname "$0")"
P=${PYTHON:-python3}
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D=data/stimuli_v2; F=results/round2/features/randinit; OOF=results/round2/oof_randinit
mkdir -p logs $F $OOF
log() { echo "[$(date +%T)] $*"; }

log "commit $(git rev-parse --short HEAD); tests"
$P -m pytest -q tests/ > logs/round2p2_tests.log 2>&1 || { log "TESTS FAILED - stopping"; exit 1; }

log "step 1: random-init 224-px models"
$P scripts/extract_random_init.py --stimuli $D --out-dir $F > logs/round2p2_extract.log 2>&1 || { log "EXTRACT FAILED"; exit 1; }

log "step 2: random-UNet SD (pretrained VAE)"
$P scripts/extract_random_init.py --stimuli $D --out-dir $F --sd --batch 16 >> logs/round2p2_extract.log 2>&1 || { log "SD EXTRACT FAILED"; exit 1; }

log "step 3: probes"
probe() { $P scripts/final_probe.py "$@" --out-dir $OOF --n-jobs 3 >> logs/round2p2_probe.log 2>&1 || log "PROBE FAILED $1"; }
export -f probe log; export P OOF
ls $F/*.npz | xargs -P 4 -I{} bash -c 'probe {}'

log "step 4: statistics"
$P scripts/final_stats.py --oof-dir $OOF --out results/round2/randinit_stats.json > logs/round2p2_stats.log 2>&1 || log "STATS FAILED"
log "PHASE 2 DONE; probe failures: $(grep -c 'PROBE FAILED' logs/round2p2_run.log 2>/dev/null || echo 0); tracebacks: $(cat logs/round2p2_*.log | grep -c Traceback)"
