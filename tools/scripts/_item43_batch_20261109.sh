#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 43 (design fixed 2026-11-09 01:18 JST; report experiments_md 20261109_01): the 43b CPU
# checks (scripts/_item43b_checks_20261109.sh, outputs in /home/koyama/data/item43/) in the background and the 43a evaluations
# (scripts/_item43a_evals_20261109.sh, one GPU, rows logged to this job's stderr) in the foreground, in ONE job because the
# account's MaxSubmit (20, pending included) was full when item 43 started. ANALYSIS.
#   bash scripts/_item43_batch_20261109.sh [43a row ...]
set -u
cd /home/koyama/code/ST3D/tools
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash scripts/_item43b_checks_20261109.sh > /home/koyama/data/item43_checks_$SLURM_JOB_ID.log 2>&1 &
CHECKS=$!
bash scripts/_item43a_evals_20261109.sh "$@"
wait $CHECKS
echo "=== item 43 batch done $(date '+%F %T'): checks summary /home/koyama/data/item43_checks_$SLURM_JOB_ID.log ===" >&2
cat /home/koyama/data/item43_checks_$SLURM_JOB_ID.log >&2
