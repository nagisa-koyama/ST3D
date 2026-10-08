#!/usr/bin/env bash
# CPU-only scoring for experiments_md/20261008_04 (analysis). Do not edit while a job runs it.
cd /home/koyama/code/ST3D/tools
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
singularity exec --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif \
  python3 analysis/protocol_matched_residuals.py run --group "$1" --workers "$2" --out /home/koyama/code/ST3D/output/analysis/protocol_residuals_20261008
