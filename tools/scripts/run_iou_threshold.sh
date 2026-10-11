#!/bin/bash
# CPU-only Slurm job (experiments_md 20261011_04, ANALYSIS): IoU-threshold sensitivity of box-size / localisation effects.
set -u
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
cd /home/koyama/code/ST3D
mkdir -p tools/logs/box_error
echo "=== node $(hostname) cpus ${SLURM_CPUS_PER_TASK:-?} $(date)"
singularity exec --bind /home/koyama/data/:/storage $SIF python3 -m pytest -q tests/test_box_error_perturbation.py || exit 3
cd tools
singularity exec --bind /home/koyama/data/:/storage $SIF python3 analysis/iou_threshold_sensitivity.py \
  --out logs/box_error/iou_threshold.jsonl --workers "${SLURM_CPUS_PER_TASK:-8}"
echo "=== done $(date)"
