#!/bin/bash
# CPU-only Slurm job (experiments_md 20261011_06, ANALYSIS): oracle box-error sensitivity, any class, several IoU bars.
# Usage (through scripts/submit.sh, no --gres):  run_box_error_multiiou.sh <dataset> <label> <result.pkl> <cfg|-> [device] [cone]
set -u
DS=$1; LABEL=$2; RESULT=$3; CFG=${4:--}; DEVICE=${5:-0}; CONE=${6:-0}
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
cd /home/koyama/code/ST3D
mkdir -p tools/logs/box_error
echo "=== node $(hostname) cpus ${SLURM_CPUS_PER_TASK:-?} $(date)"
singularity exec --bind /home/koyama/data/:/storage $SIF python3 -m pytest -q tests/test_box_error_perturbation.py || exit 3
cd tools
ARGS=(--dataset "$DS" --cls Pedestrian --thresholds 0.5,0.25,0.1 --result "$RESULT" --label "$LABEL"
      --out logs/box_error/${LABEL}.jsonl --workers "${SLURM_CPUS_PER_TASK:-8}" --device "$DEVICE" --cone "$CONE")
[ "$CFG" != "-" ] && ARGS+=(--cfg "$CFG")
singularity exec --bind /home/koyama/data/:/storage $SIF python3 analysis/oracle_box_error_multiiou.py "${ARGS[@]}"
echo "=== done $(date)"
