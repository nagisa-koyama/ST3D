#!/usr/bin/env bash
# COND-TAB (experiments_md 20261028_01; design fixed in T05 §5 "User 10-28 (received between 07:50 and 11:39)", 11:45 JST).
# ANALYSIS: Waymo / nuScenes weather and illumination tags per split and cell; GT counted for diagnosis only. CPU only.
# From ST3D/tools (a bash script, not a dash --wrap):
#
#   WANDB_NOTES="CPU only, no W&B: ..." scripts/submit.sh --partition=a100,a6000_ada --cpus-per-task=8 --mem=32G \
#     --time=02:00:00 --job-name=condTab --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
#     analysis/condition_tags_job.sh
set -euo pipefail
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
OUT=/home/koyama/data/cond_tab_20261028
mkdir -p "$OUT"
echo "=== COND-TAB at ST3D $(git -C /home/koyama/code/ST3D log -1 --format=%h) on $(hostname), $(date '+%Y-%m-%d %H:%M %Z') ==="
singularity exec --bind /home/koyama/data/:/storage "$SIF" python3 analysis/condition_tags.py \
  --out "$OUT" --workers "${SLURM_CPUS_PER_TASK:-8}"
echo "=== COND-TAB exit 0 at $(date '+%Y-%m-%d %H:%M %Z') ==="
