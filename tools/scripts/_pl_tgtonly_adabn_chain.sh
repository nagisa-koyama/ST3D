#!/usr/bin/env bash
# Check PL-TGTONLY (experiments_md 20261022_08 §3 item 3, §8): the chained test-time AdaBN evaluation after a row.
# A bash script rather than `sbatch --wrap`, because --wrap runs under /bin/sh (dash), which rejects the ${var/#..}
# substitution that maps the host checkpoint path to /storage (the other agent's 30220 died on it, 2026-10-23).
#
#   sbatch --dependency=afterany:<row> --kill-on-invalid-dep=yes ... scripts/_pl_tgtonly_adabn_chain.sh \
#       <row jobid> <last epoch> <row cfg> <tag> [teacher ckpt (container /storage path)]
# Student: arms none then target with the row's own config (adabn_eval.sh, 1,000 strided target TRAIN frames, MIX 1.0).
# Teacher (optional, T3 only): arms none then target with the same config, so its statistics come from the same frames.
# Runs from the live checkout (as adabn_eval.sh always does). Do not edit while a job runs it.
set -euo pipefail
ROW=${1:?row jobid}; EPOCH=${2:?epoch}; CFG=${3:?cfg}; TAG=${4:?tag}; TEACHER=${5:-}
cd /home/koyama/code/ST3D/tools
echo "=== live HEAD $(git -C /home/koyama/code/ST3D rev-parse --short HEAD) on $(hostname) at $(date '+%F %T') ==="
C=$(bash analysis/ckpt_of_job.sh "$ROW" "$EPOCH")
CS=${C/#\/home\/koyama\/data//storage}
echo "student: $CS"
for ARM in none target; do
    bash analysis/adabn_eval.sh "$CFG" "$CS" "$ARM" 1.0 "${TAG}_${ARM}_${ROW}"
    echo "student $ARM exit $?"
done
if [ -n "$TEACHER" ]; then
    echo "teacher: $TEACHER"
    for ARM in none target; do
        bash analysis/adabn_eval.sh "$CFG" "$TEACHER" "$ARM" 1.0 "${TAG}_teacher_${ARM}"
        echo "teacher $ARM exit $?"
    done
fi
echo "=== done $(date '+%F %T') ==="
