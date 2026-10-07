#!/bin/bash
# Final analysis run (frozen). Execute from the repository root on a CUDA machine.
# Step 0 tests; 1 extract remaining SD seeds; 2 natural-texture features; 3 OOF probes; 4 statistics.
set -u
cd "$(dirname "$0")"
P=${PYTHON:-python3}
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D=data/stimuli_v2; V3=results/features/v3; SD=results/features/v3sd; NAT=results/features/natural; OOF=results/final/oof
log() { echo "[$(date +%T)] $*"; }

log "commit $(git rev-parse --short HEAD); tests"
$P -m pytest -q tests/ > logs/final_tests.log 2>&1 || { log "TESTS FAILED - stopping"; exit 1; }

log "step 1: SD seeds 5-9 (main, k=8/4/2, t=100/300/500) and replication seeds 0-9 (k=4, t=500)"
for g in 64 128 256; do
  $P scripts/extract_sd_seeded.py --stim-dir $D/main_sd --grid $g --set-name main --resize nearest \
     --seeds 5 6 7 8 9 --timesteps 100 300 500 --out-dir $SD >> logs/final_extract.log 2>&1 || { log "EXTRACT FAILED g$g"; exit 1; }
done
$P scripts/extract_sd_seeded.py --stim-dir $D/replication --grid 128 --set-name rep --resize nearest \
   --seeds 0 1 2 3 4 5 6 7 8 9 --timesteps 500 --out-dir $SD >> logs/final_extract.log 2>&1 || { log "EXTRACT FAILED rep"; exit 1; }

log "step 2: natural-texture features"
$P scripts/final_natural.py --patches ${NATURAL_PATCHES:-data/natural/dtd_patches_gamma.npz} \
   --out-dir $NAT > logs/final_natural.log 2>&1 || { log "NATURAL FAILED"; exit 1; }

log "step 3: OOF probes"
mkdir -p $OOF
probe() { $P scripts/final_probe.py "$@" --out-dir $OOF --n-jobs 3 >> logs/final_probe.log 2>&1 || log "PROBE FAILED $1"; }
export -f probe log; export P OOF
{
  for m in clip mae dino supvit_in1k supvit_21k resnet50; do
    for g in 28 112; do echo "$V3/${m}_nearest_g${g}_main.npz"; done
    echo "$V3/${m}_nearest_g56_rep.npz"
    echo "$V3/${m}_nearest_g56_main.npz --mlp"
  done
  echo "$V3/imagegpt_nearest_g32_main.npz --mlp"
  for g in 64 128 256; do for t in 100 300 500; do for s in 0 1 2 3 4 5 6 7 8 9; do
    if [ "$g" = 128 ] && [ "$t" = 500 ]; then echo "$SD/sd15_mid_t${t}_s${s}_nearest_g${g}_main.npz --mlp"
    else echo "$SD/sd15_mid_t${t}_s${s}_nearest_g${g}_main.npz"; fi
  done; done; done
  for s in 0 1 2 3 4 5 6 7 8 9; do echo "$SD/sd15_mid_t500_s${s}_nearest_g128_rep.npz"; done
  for f in $NAT/nat_*.npz; do echo "$f --target-key gammas --repeats 2"; done
} | xargs -P 4 -I{} bash -c 'probe {}'

log "step 4: statistics"
$P scripts/final_stats.py --oof-dir $OOF --out results/final/final_stats.json > logs/final_stats.log 2>&1 || { log "STATS FAILED"; exit 1; }
sha256sum $D/*/*.mat $V3/*.npz $SD/*.npz $NAT/*.npz > results/final/input_hashes.txt
log "FINAL ALL DONE; probe failures: $(grep -c 'PROBE FAILED' logs/run_final.log 2>/dev/null || echo 0); tracebacks: $(cat logs/final_*.log | grep -c Traceback)"
