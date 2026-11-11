#!/bin/bash
# Confirmatory run, probe stage under Amendment 1 (confirmatory/AMENDMENT_1.md).
# Stimuli and features were produced by the frozen run_confirm.sh (tag confirmatory-protocol), which was
# stopped at the extraction/probe boundary. This script runs the remaining steps of run_confirm.sh, with the
# single change that every probe gets --drop-rare 0.01. It refuses to run if any confirmatory OOF exists.
set -u
cd "$(dirname "$0")"
P=${PYTHON:-python3}
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S=data/stimuli_conf; F=results/confirm/features; OOF=results/confirm/oof; DR="--drop-rare 0.01"
mkdir -p logs $OOF
log() { echo "[$(date +%T)] $*"; }

log "commit $(git rev-parse --short HEAD) ($(git describe --tags --always)); host $(hostname); tests"
$P -m pytest -q tests/ > logs/confirm_probes_tests.log 2>&1 || { log "TESTS FAILED - stopping"; exit 1; }

[ -z "$(ls $OOF/*.npz 2>/dev/null)" ] || { log "confirmatory OOF files already exist - precondition violated"; exit 1; }
read -r -a valid <<< "$(grep 'valid sets:' logs/confirm_run.log | tail -1 | sed 's/.*valid sets: //')"
[ ${#valid[@]} -eq 10 ] || { log "expected 10 valid sets, found ${#valid[@]}"; exit 1; }
missing=0; n_sd=0
for m in "${valid[@]}"; do
  for mod in clip mae dino supvit_in1k supvit_21k resnet50; do for g in 28 56; do
    [ -f $F/${mod}_nearest_g${g}_c$m.npz ] || { log "missing $mod g$g c$m"; missing=1; }
  done; done
  [ -f $F/imagegpt_nearest_g32_c$m.npz ] || { log "missing imagegpt c$m"; missing=1; }
  if [ -f $S/set$m/alpha_0p05_128x128_c$m.mat ]; then
    n_sd=$((n_sd + 1))
    for s in 0 1 2 3 4; do [ -f $F/sd15_mid_t500_s${s}_nearest_g128_c$m.npz ] || { log "missing sd s$s c$m"; missing=1; }; done
  fi
done
[ $missing = 0 ] && [ $n_sd = 5 ] || { log "feature set incomplete (SD sets: $n_sd) - stopping"; exit 1; }
log "valid sets: ${valid[*]}; all features present"

probe() { $P scripts/final_probe.py "$@" --out-dir $OOF --n-jobs 3 --drop-rare 0.01 >> logs/confirm_probe.log 2>&1 || log "PROBE FAILED $1"; }
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

SEC=results/confirm/secondary; KEY=()
for mod in clip mae dino supvit_in1k supvit_21k resnet50; do KEY+=($F/${mod}_nearest_g56_c$first.npz $F/${mod}_nearest_g28_c$first.npz); done
KEY+=($F/imagegpt_nearest_g32_c$first.npz)
for s in 0 1 2 3 4; do KEY+=($F/sd15_mid_t500_s${s}_nearest_g128_c$first.npz); done
$P scripts/r2_empirical_target.py "${KEY[@]}" --stimuli $S --stim-subdir set$first --out-dir $SEC/emp_features \
   > logs/confirm_secondary.log 2>&1 || log "EMPIRICAL TARGET FAILED"
ls $SEC/emp_features/*.npz | sed "s|\$| --target-key alpha_check|" | xargs -P 4 -I{} bash -c 'OOF='$SEC'/oof_emp probe {}'
$P scripts/final_stats.py --oof-dir $SEC/oof_emp --out $SEC/emp_stats.json >> logs/confirm_secondary.log 2>&1 || log "STATS FAILED emp"
$P scripts/r2_lolo.py "${KEY[@]}" --out $SEC/lolo.json --n-jobs 12 $DR >> logs/confirm_secondary.log 2>&1 || log "LOLO FAILED"
log "CONFIRMATORY RUN DONE; probe failures: $(grep -c 'PROBE FAILED' logs/confirm_probes_run.log 2>/dev/null); tracebacks: $(cat logs/confirm_probe*.log logs/confirm_stats.log logs/confirm_secondary.log | grep -c Traceback)"
