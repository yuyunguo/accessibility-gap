#!/bin/bash
# Confirmatory run (confirmatory/PROTOCOL.md). Execute from the repository root at the tagged commit.
# Sets m = 0, 1, ... are generated (base seed 40000 + 100*m) until 10 valid sets exist; the first 5 valid
# sets also get the 128 grid for SD. A set with any invalid file is discarded whole (PROTOCOL §5).
# RUN_SD=0 skips SD extraction (for running it on another host; PROTOCOL §6), and the SD probes and
# confirmatory statistics are then run after the SD features are copied into $F.
set -u
cd "$(dirname "$0")"
P=${PYTHON:-python3}
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
BTC=${BTC_CODE_DIR:-../efficient-coding/jv}; RUN_SD=${RUN_SD:-1}
S=data/stimuli_conf; F=results/confirm/features; OOF=results/confirm/oof
mkdir -p logs $S $F $OOF
log() { echo "[$(date +%T)] $*"; }

log "commit $(git rev-parse --short HEAD) ($(git describe --tags --always)); host $(hostname); tests"
$P -m pytest -q tests/ > logs/confirm_tests.log 2>&1 || { log "TESTS FAILED - stopping"; exit 1; }

valid=(); n_sd=0; m=0
while [ ${#valid[@]} -lt 10 ]; do
  [ $m -gt 19 ] && { log "more than 10 discarded sets - stopping"; exit 1; }
  sizes="28 56 32"; [ $n_sd -lt 5 ] && sizes="$sizes 128"
  $P stimuli/generate_btc_octave.py --btc-dir $BTC --out-dir $S/set$m --levels 0.05 0.1 0.2 0.3 0.4 0.5 \
     --sizes $sizes --set-name c$m --base-seed $((40000 + 100 * m)) >> logs/confirm_stimuli.log 2>&1 \
     || { log "GENERATOR FAILED set $m"; exit 1; }
  if $P -c "import json,sys; m=json.load(open('$S/set$m/manifest.json')); sys.exit(0 if all(e['valid'] for e in m) else 1)"; then
    valid+=($m); [[ " $sizes " == *" 128 "* ]] && n_sd=$((n_sd + 1)); log "set $m valid (sizes $sizes)"
  else
    log "set $m DISCARDED (invalid file; see $S/set$m/manifest.json)"
  fi
  m=$((m + 1))
done
log "valid sets: ${valid[*]}"

for m in "${valid[@]}"; do
  for g in 28 56; do
    $P scripts/extract_features_v2.py --stim-dir $S/set$m --out-dir $F --resize nearest --grid $g --set-name c$m \
       --models clip mae dino supvit_in1k supvit_21k resnet50 >> logs/confirm_extract.log 2>&1 || { log "EXTRACT FAILED c$m g$g"; exit 1; }
  done
  $P scripts/extract_features_v2.py --stim-dir $S/set$m --out-dir $F --resize nearest --grid 32 --set-name c$m \
     --models imagegpt >> logs/confirm_extract.log 2>&1 || { log "EXTRACT FAILED c$m imagegpt"; exit 1; }
  if [ "$RUN_SD" = 1 ] && [ -f $S/set$m/alpha_0p05_128x128_c$m.mat ]; then
    $P scripts/extract_sd_seeded.py --stim-dir $S/set$m --grid 128 --set-name c$m --resize nearest \
       --seeds 0 1 2 3 4 --timesteps 500 --out-dir $F >> logs/confirm_extract.log 2>&1 || { log "EXTRACT FAILED c$m sd"; exit 1; }
  fi
  log "extraction done c$m"
done

probe() { $P scripts/final_probe.py "$@" --out-dir $OOF --n-jobs 3 >> logs/confirm_probe.log 2>&1 || log "PROBE FAILED $1"; }
export -f probe log; export P OOF
first=${valid[0]}
{
  for f in $(ls $F/*_c*.npz | grep -v vae_latents); do
    case "$f" in   # secondary: tuned MLP on Family P, first valid set only (PROTOCOL §8)
      *_g56_c${first}.npz|*imagegpt_nearest_g32_c${first}.npz|*sd15_mid_t500_s*_nearest_g128_c${first}.npz) echo "$f --mlp" ;;
      *) echo "$f" ;;
    esac
  done
} | xargs -P 4 -I{} bash -c 'probe {}'
log "probes done"

$P scripts/r2_confirm_stats.py --oof-dir $OOF --out results/confirm/confirm_stats.json > logs/confirm_stats.log 2>&1 \
   || { log "CONFIRMATORY STATISTICS FAILED (see logs/confirm_stats.log)"; exit 1; }
cat logs/confirm_stats.log
log "CONFIRMATORY STATISTICS DONE"

# Secondary analyses on the first valid set (PROTOCOL §8; specified after exploratory results).
SEC=results/confirm/secondary; KEY=()
for mod in clip mae dino supvit_in1k supvit_21k resnet50; do KEY+=($F/${mod}_nearest_g56_c$first.npz $F/${mod}_nearest_g28_c$first.npz); done
KEY+=($F/imagegpt_nearest_g32_c$first.npz)
for s in 0 1 2 3 4; do [ -f $F/sd15_mid_t500_s${s}_nearest_g128_c$first.npz ] && KEY+=($F/sd15_mid_t500_s${s}_nearest_g128_c$first.npz); done
$P scripts/r2_empirical_target.py "${KEY[@]}" --stimuli $S --stim-subdir set$first --out-dir $SEC/emp_features \
   > logs/confirm_secondary.log 2>&1 || log "EMPIRICAL TARGET FAILED"
ls $SEC/emp_features/*.npz | sed "s|\$| --target-key alpha_check|" | xargs -P 4 -I{} bash -c 'OOF='$SEC'/oof_emp probe {}'
$P scripts/final_stats.py --oof-dir $SEC/oof_emp --out $SEC/emp_stats.json >> logs/confirm_secondary.log 2>&1 || log "STATS FAILED emp"
$P scripts/r2_lolo.py "${KEY[@]}" --out $SEC/lolo.json --n-jobs 12 >> logs/confirm_secondary.log 2>&1 || log "LOLO FAILED"
log "CONFIRMATORY RUN DONE; probe failures: $(grep -c 'PROBE FAILED' logs/confirm_run.log 2>/dev/null); tracebacks: $(cat logs/confirm_*.log | grep -c Traceback)"
