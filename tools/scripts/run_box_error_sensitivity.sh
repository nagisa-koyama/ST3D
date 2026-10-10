#!/bin/bash
# CPU-only Slurm job (experiments_md 20261010_11, ANALYSIS): re-score an oracle's predictions after one controlled box error.
# Usage (through scripts/submit.sh, no --gres):  run_box_error_sensitivity.sh <label> <cfg> <result.pkl> [arms]
# Runs the sentinel tests first, so a broken perturbation fails the job before any scoring.
set -u
LABEL=$1; CFG=$2; RESULT=$3; ARMS=${4:-all}
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
cd /home/koyama/code/ST3D
mkdir -p tools/logs/box_error
echo "=== node $(hostname) cpus ${SLURM_CPUS_PER_TASK:-?} $(date)"
singularity exec --bind /home/koyama/data/:/storage $SIF python3 -m pytest -q tests/test_box_error_perturbation.py || exit 3
cd tools
singularity exec --bind /home/koyama/data/:/storage $SIF python3 analysis/oracle_box_error_sensitivity.py \
  --cfg "$CFG" --result "$RESULT" --label "$LABEL" --out logs/box_error/${LABEL}.jsonl \
  --arms "$ARMS" --workers "${SLURM_CPUS_PER_TASK:-8}"
echo "=== done $(date)"
