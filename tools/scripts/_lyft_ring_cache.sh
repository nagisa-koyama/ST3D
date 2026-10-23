#!/usr/bin/env bash
# Lyft 40-beam ring cache + kept-frame list (experiments_md 20261024_01 §3). CPU only, no W&B.
#   WANDB_NOTES="CPU only, no W&B: ..." scripts/submit.sh --partition=a100,a6000_ada --cpus-per-task=8 --mem=32G \
#     --time=03:00:00 --job-name=lyftcache --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
#     scripts/_lyft_ring_cache.sh [out_dir]
set -euo pipefail
OUT=${1:-/home/koyama/data/lyft_rings/v1_40beam}
cd /home/koyama/code/ST3D/tools
echo "=== $(date '+%Y-%m-%d %H:%M %Z') on $(hostname), ST3D $(git -C /home/koyama/code/ST3D rev-parse --short HEAD), out $OUT ==="
singularity exec --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif \
    python3 analysis/lyft_ring_cache.py --out_dir "$OUT" --workers "${SLURM_CPUS_PER_TASK:-8}"
echo "=== done $(date '+%Y-%m-%d %H:%M %Z') ==="
