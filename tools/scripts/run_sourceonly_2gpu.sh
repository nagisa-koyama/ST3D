#!/usr/bin/env bash
#SBATCH --job-name=sourceonly
#SBATCH --partition=a6000_ada,a6000,rtx8000
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/output_%j_%x.txt
#SBATCH --error=logs/error_%j_%x.txt
#
# Launch one da-ieee-access source-only baseline on 2 GPUs at a GLOBAL batch of 6.
#
#   sbatch scripts/run_sourceonly_2gpu.sh <source> [extra_tag]
#   <source> = kitti | lyft | nuscenes | pandaset | waymo
#
# Kept separate from run_experiment.sh, which holds one active line at a time and is shared with
# other work; this one is parameterized instead, because the five runs differ only by source.
#
# WHY --batch_size 6 IS NOT OPTIONAL. It is the TOTAL across ranks and train.py:104-107 divides it
# by the GPU count, so 6 gives 3 per rank and a global batch of 6 - identical to the single-GPU
# recipe: same 93,766 optimizer steps, same LR 0.003. OMITTING it does not fall back to that; it
# makes args.batch_size = BATCH_SIZE_PER_GPU = 6 PER RANK, a global batch of 12 and HALF the
# steps, silently. train.py logs "global batch size:" at startup - it must read 6.
#
# Why 2 GPUs at 3/rank rather than 1 GPU or 2 GPUs at 6/rank: measured same-node, jobs 25734/25735.
# Scaling is ~linear (2.06x PandaSet, 2.03x KITTI), bs=3 is still saturated (0.944-0.965x), and
# 3/rank beats 6/rank on wall-clock while keeping twice the optimizer steps. Details in
# experiments_md/20260922_06 section 2e.
#
# --workers is PER RANK and comes from each config's OPTIMIZATION.NUM_WORKERS (4, or 8 for Waymo
# and PandaSet whose loaders stall), so it is deliberately NOT passed here.
set -euo pipefail

SOURCE=${1:?usage: sbatch scripts/run_sourceonly_2gpu.sh <kitti|lyft|nuscenes|pandaset|waymo> [extra_tag]}
TAG=${2:-$(date +%Y%m%d)_sourceonly}
CFG=cfgs/da-ieee-access/centerpoint-sourceonly-${SOURCE}.yaml
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
cd /home/koyama/code/ST3D/tools
[ -f "$CFG" ] || { echo "no such config: $CFG" >&2; exit 2; }

# Random port so two of these can run concurrently without a rendezvous collision.
PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))

# The /root/ST3D bind is only strictly needed by PandaSet, whose infos bake in absolute
# /root/ST3D paths, but it is harmless elsewhere and keeps one invocation for all five.
singularity exec --nv \
    --bind /home/koyama/data/:/storage \
    --bind /home/koyama/code/ST3D:/root/ST3D \
    "$SIF" \
    python -m torch.distributed.launch --use-env --nproc_per_node=2 \
        --rdzv_endpoint=localhost:"$PORT" \
    train.py --launcher pytorch --tcp_port "$PORT" \
        --cfg_file "$CFG" \
        --batch_size 6 \
        --fix_random_seed \
        --run_name "sourceonly_${SOURCE}2nuscenes" \
        --extra_tag "$TAG"
