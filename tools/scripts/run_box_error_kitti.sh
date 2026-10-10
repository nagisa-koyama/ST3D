#!/bin/bash
# CPU-only Slurm job (experiments_md 20261011_01, ANALYSIS): KITTI oracle box-error sensitivity.
# Usage (through scripts/submit.sh, no --gres):  run_box_error_kitti.sh <label> <result.pkl> [arms]
set -u
LABEL=$1; RESULT=$2; ARMS=${3:-all}
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
cd /home/koyama/code/ST3D
mkdir -p tools/logs/box_error
echo "=== node $(hostname) cpus ${SLURM_CPUS_PER_TASK:-?} $(date)"
singularity exec --bind /home/koyama/data/:/storage $SIF python3 -m pytest -q tests/test_box_error_perturbation.py || exit 3
cd tools
singularity exec --bind /home/koyama/data/:/storage $SIF python3 analysis/oracle_box_error_kitti.py \
  --result "$RESULT" --label "$LABEL" --out logs/box_error/${LABEL}.jsonl --arms "$ARMS" --workers "${SLURM_CPUS_PER_TASK:-8}"
echo "=== done $(date)"
