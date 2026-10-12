#!/usr/bin/env bash
# CPU-only Slurm job for experiments_md 20261013_02: rescore the paper's rows at IoU 0.7 / 0.5 (Car) and 0.5 / 0.25
# (Pedestrian), one oracle_box_error_multiiou.py --arms base call per (row, class), $SLURM_CPUS_PER_TASK at a time.
# Usage (from ST3D/tools, through scripts/submit.sh, no --gres): analysis/_multiiou_rows_job.sh <cmds.txt>
# Do not edit while a job runs it.
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "=== node $(hostname) cpus ${SLURM_CPUS_PER_TASK:-?} $(date)"
xargs -d '\n' -P "${SLURM_CPUS_PER_TASK:-8}" -I{} bash -c '{}; echo "ROW_DONE rc=$? $(date +%H:%M:%S) $(echo "{}" | grep -o "label [^ ]*")"' < "$1"
echo "=== ALL ROWS DONE $(date)"
