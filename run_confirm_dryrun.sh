#!/bin/bash
# Confirmatory dry run (confirmatory/PROTOCOL.md §9): one exploratory set, seeds 49900+, set name "dry".
# Times every stage and checks the pipeline end to end. Its data are NOT part of the confirmatory test.
set -u
cd "$(dirname "$0")"
P=${PYTHON:-python3}
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
BTC=${BTC_CODE_DIR:-../efficient-coding/jv}
S=data/stimuli_conf/dry; F=results/confirm_dry/features; OOF=results/confirm_dry/oof
mkdir -p logs $S $F $OOF
log() { echo "[$(date +%T)] $*"; }
t0=$(date +%s); lap() { echo "$(( $(date +%s) - t0 ))s"; }

log "commit $(git rev-parse --short HEAD); tests"
$P -m pytest -q tests/ > logs/dry_tests.log 2>&1 || { log "TESTS FAILED - stopping"; exit 1; }

log "stimuli: 6 levels x grids 28, 56, 32, 128 (seeds 49900+)"
$P stimuli/generate_btc_octave.py --btc-dir $BTC --out-dir $S --levels 0.05 0.1 0.2 0.3 0.4 0.5 \
   --sizes 28 56 32 128 --set-name dry --base-seed 49900 > logs/dry_stimuli.log 2>&1 || { log "STIMULI FAILED"; exit 1; }
$P -c "import json,sys; m=json.load(open('$S/manifest.json')); bad=[e['file'] for e in m if not e['valid']]; print(len(m),'files; invalid:',bad); sys.exit(1 if bad or len(m)!=24 else 0)" \
   || { log "VALIDATION FAILED"; exit 1; }
log "stimuli done $(lap)"

for g in 28 56; do
  $P scripts/extract_features_v2.py --stim-dir $S --out-dir $F --resize nearest --grid $g --set-name dry \
     --models clip mae dino supvit_in1k supvit_21k resnet50 >> logs/dry_extract.log 2>&1 || { log "EXTRACT FAILED g$g"; exit 1; }
done
$P scripts/extract_features_v2.py --stim-dir $S --out-dir $F --resize nearest --grid 32 --set-name dry \
   --models imagegpt >> logs/dry_extract.log 2>&1 || { log "EXTRACT FAILED imagegpt"; exit 1; }
log "224-px + ImageGPT extraction done $(lap)"
$P scripts/extract_sd_seeded.py --stim-dir $S --grid 128 --set-name dry --resize nearest --seeds 0 1 2 3 4 \
   --timesteps 500 --out-dir $F >> logs/dry_extract.log 2>&1 || { log "EXTRACT FAILED sd"; exit 1; }
log "SD extraction done $(lap)"

probe() { $P scripts/final_probe.py "$@" --out-dir $OOF --n-jobs 3 >> logs/dry_probe.log 2>&1 || log "PROBE FAILED $1"; }
export -f probe log; export P OOF
ls $F/*_dry.npz | grep -v vae_latents | xargs -P 4 -I{} bash -c 'probe {}'
log "probes done $(lap)"
$P scripts/final_stats.py --oof-dir $OOF --out results/confirm_dry/dry_stats.json > logs/dry_stats.log 2>&1 || log "STATS FAILED"
log "DRY RUN DONE $(lap); tracebacks: $(cat logs/dry_*.log | grep -c Traceback)"
