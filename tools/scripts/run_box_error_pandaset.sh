#!/bin/bash
# CPU-only Slurm job (experiments_md 20261011_01, ANALYSIS): PandaSet oracle box-error sensitivity under rule A.
# Usage (through scripts/submit.sh, no --gres):  run_box_error_pandaset.sh <label> <result.pkl> <device 0|1> <cone 0|1> [arms]
set -u
LABEL=$1; RESULT=$2; DEVICE=$3; CONE=$4; ARMS=${5:-all}
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
cd /home/koyama/code/ST3D
mkdir -p tools/logs/box_error
echo "=== node $(hostname) cpus ${SLURM_CPUS_PER_TASK:-?} $(date)"
singularity exec --bind /home/koyama/data/:/storage $SIF python3 -m pytest -q tests/test_box_error_perturbation.py || exit 3
cd tools
singularity exec --bind /home/koyama/data/:/storage $SIF python3 analysis/oracle_box_error_pandaset.py \
  --result "$RESULT" --device "$DEVICE" --cone "$CONE" --label "$LABEL" --out logs/box_error/${LABEL}.jsonl \
  --arms "$ARMS" --workers "${SLURM_CPUS_PER_TASK:-8}"
echo "=== done $(date)"
