#!/bin/bash
# One probe pipeline throughout: apply the confirmatory Amendment 1 rare-feature filter (--drop-rare 0.01)
# to the main and round-2 analyses. Execute from the repository root after run_final.sh and run_round2_phase*.sh.
# Writes only results/merged_a1/; results/final and results/round2 are left unchanged.
#  1 scope: find the feature files the filter can change (scripts/rare_filter_scope.py, a worst-case bound);
#    every other file is provably unchanged, so its original OOF predictions are reused as they are.
#  2 refit only those files, with the same probe arguments as the original run plus --drop-rare 0.01.
#  3 merged OOF directories (originals with the refit files overlaid), statistics, LOLO, simultaneous bounds.
# On 2026-11-11 this selected 24 files (ResNet50: 4 main + 2 empirical-alpha; random-init ResNet50: 9;
# random-ReLU pixel baseline: 9). The released merged results were produced by identical code (tag
# confirmatory-amendment-1) run from a separate directory; the probes are seeded, so this script reproduces them.
set -u
cd "$(dirname "$0")"
P=${PYTHON:-python3}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
M=results/merged_a1; R2=results/round2
V3=results/features/v3; SD=results/features/v3sd; NAT=results/features/natural
RI=$R2/features/randinit; EMP=$R2/features/emp
test -e $M && { echo "$M exists - remove it first"; exit 1; }
mkdir -p logs $M/refit_final $M/refit_randinit $M/refit_emp
log() { echo "[$(date +%T)] $*"; }

log "commit $(git rev-parse --short HEAD); step 1: filter scope"
$P scripts/rare_filter_scope.py $V3/*.npz $SD/*.npz $NAT/*.npz $RI/*.npz $EMP/*.npz --list $M/acts.txt \
   > logs/merge_a1_scope.log 2>&1 || { log "SCOPE FAILED"; exit 1; }

log "step 2: refit $(wc -l < $M/acts.txt) files"
probe() { $P scripts/final_probe.py "$@" --n-jobs 3 --drop-rare 0.01 >> logs/merge_a1_probe.log 2>&1 || log "PROBE FAILED $1"; }
export -f probe log; export P
while read -r f; do
  case "$f" in
    # primary conditions carry the MLP probe, as in run_final.sh
    $V3/*_g56_main.npz|$V3/imagegpt_nearest_g32_main.npz|$SD/sd15_mid_t500_s*_nearest_g128_main.npz)
      echo "$f --mlp --out-dir $M/refit_final" ;;
    $V3/*.npz|$SD/*.npz) echo "$f --out-dir $M/refit_final" ;;
    $NAT/*.npz) echo "$f --target-key gammas --repeats 2 --out-dir $M/refit_final" ;;
    $RI/*.npz) echo "$f --out-dir $M/refit_randinit" ;;
    $EMP/*.npz) echo "$f --target-key alpha_check --out-dir $M/refit_emp" ;;
    *) log "UNMAPPED $f"; exit 1 ;;
  esac
done < $M/acts.txt | xargs -P 4 -I{} bash -c 'probe {}'

log "step 3: merged OOF directories and statistics"
for s in final randinit emp; do
  case $s in final) src=results/final/oof ;; *) src=$R2/oof_$s ;; esac
  mkdir -p $M/oof_$s && cp -p $src/* $M/oof_$s/ && cp -p $M/refit_$s/* $M/oof_$s/ 2>/dev/null
  $P scripts/final_stats.py --oof-dir $M/oof_$s --out $M/${s}_stats.json > logs/merge_a1_stats_$s.log 2>&1 || log "STATS FAILED $s"
done
$P scripts/r2_simultaneous.py --oof-dir $M/oof_final --stats $M/final_stats.json --out $M/simultaneous.json \
   > logs/merge_a1_simultaneous.log 2>&1 || log "SIMULTANEOUS FAILED"

log "step 4: leave-one-level-out for the changed files of the LOLO set (run_round2_phase1.sh; other entries copied)"
lolo=$(grep -E "^$V3/.*_g(56|28)_main\.npz$|^$V3/imagegpt_nearest_g32_main\.npz$|^$SD/sd15_mid_t500_s[0-4]_nearest_g128_main\.npz$" \
       $M/acts.txt | tr '\n' ' ')
$P - "$R2/lolo.json" "$M/lolo.json" $lolo <<'EOF'
import json, sys
from pathlib import Path
src, dst, changed = sys.argv[1], sys.argv[2], {Path(f).stem for f in sys.argv[3:]}
d = json.loads(Path(src).read_text())
Path(dst).write_text(json.dumps({k: v for k, v in d.items() if k not in changed}, indent=1))
EOF
[ -n "$lolo" ] && { $P scripts/r2_lolo.py $lolo --out $M/lolo.json --n-jobs 12 --drop-rare 0.01 \
   > logs/merge_a1_lolo.log 2>&1 || log "LOLO FAILED"; }
log "MERGE DONE; probe failures: $(grep -c 'PROBE FAILED' logs/merge_a1_run.log 2>/dev/null || echo 0); tracebacks: $(cat logs/merge_a1_*.log | grep -c Traceback)"
