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

# $1 is either one of the five source names, or a path to any config in the family - the latter
# so one-off rows (e.g. the Lyft->Lyft oracle, which is not an `X -> nuScenes` source) get the same
# 2-GPU recipe and the same frozen-code guarantee instead of a hand-rolled invocation.
SOURCE=${1:?usage: sbatch scripts/run_sourceonly_2gpu.sh <kitti|lyft|nuscenes|pandaset|waymo|path/to.yaml> [extra_tag]}
TAG=${2:-$(date +%Y%m%d)_sourceonly}
case "$SOURCE" in
  *.yaml) CFG=$SOURCE; SOURCE=$(basename "$CFG" .yaml) ;;
  *)      CFG=cfgs/da-ieee-access/centerpoint-sourceonly-${SOURCE}.yaml ;;
esac
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
REPO=/home/koyama/code/ST3D
cd "$REPO/tools"
[ -f "$CFG" ] || { echo "no such config: $CFG" >&2; exit 2; }

# --- freeze the code for the life of this job ------------------------------------------------
#
# Job 25743 trained all 20 epochs cleanly and then died in EVALUATION with
# `AttributeError: 'DataProcessor' object has no attribute 'ptsn_scale'`, because a commit landed
# 37 minutes earlier and the eval DataLoader workers picked it up mid-run.
#
# The mechanism is specific to DDP. `init_dist_pytorch` forces `mp.set_start_method('spawn')`, and
# a spawned worker is a FRESH interpreter that re-imports every module from the live bind-mounted
# checkout - while the dataset object it receives was pickled from the parent, whose class was
# imported hours earlier. Unpickling restores `__dict__` without calling `__init__`, so you get the
# NEW methods over the OLD instance state. A single-GPU run forks instead and is immune; going to
# 2 GPUs is what created this exposure, and with more than one person committing to this repo it is
# not a rare event over an 8-15 h run.
#
# So: rsync the code to the node's per-job scratch SSD (auto-created, auto-deleted at job end) and
# bind it over the repo path, so every process in this job - parent and every spawned worker -
# reads one frozen copy. ~119 MB / 2,800 files, a few seconds.
#
# The excludes matter: `output/`, `wandb/`, `tools/wandb/` and `tools/logs/` are 50+ GB of run
# artifacts, and `.nfs*` catches NFS silly-renames of deleted `.sif` images, which `*.sif` does not
# match and which are 11 GB each. `data/` IS copied - it is 248 KB of symlinks into ~/data, and
# rsync -a preserves them as links. `.git` is copied too (10 MB) so wandb can still record
# git.commit, which is how a run's exact code version stays recoverable.
SNAP=/local_cache/${SLURM_JOB_ID}/ST3D
mkdir -p "$SNAP"
rsync -a \
  --exclude='/output/' --exclude='/wandb/' --exclude='/build/' \
  --exclude='tools/wandb/' --exclude='tools/logs/' \
  --exclude='*.sif' --exclude='.nfs*' --exclude='pcdet.egg-info/' \
  "$REPO/" "$SNAP/"
echo "=== code frozen at $(git -C "$REPO" rev-parse --short HEAD) -> $SNAP ==="

# Random port so two of these can run concurrently without a rendezvous collision.
PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))

# The /root/ST3D bind is only strictly needed by PandaSet, whose infos bake in absolute
# /root/ST3D paths, but it is harmless elsewhere and keeps one invocation for all five.
# Both binds point at the SNAPSHOT: the first shadows the live repo at its own path (so relative
# config paths and `import pcdet` resolve to frozen code), the second satisfies PandaSet's infos,
# which bake absolute /root/ST3D/... paths. --pwd is explicit so cwd cannot resolve elsewhere.
singularity exec --nv \
    --bind /home/koyama/data/:/storage \
    --bind "$SNAP":/home/koyama/code/ST3D \
    --bind "$SNAP":/root/ST3D \
    --pwd /home/koyama/code/ST3D/tools \
    "$SIF" \
    python -m torch.distributed.launch --use-env --nproc_per_node=2 \
        --rdzv_endpoint=localhost:"$PORT" \
    train.py --launcher pytorch --tcp_port "$PORT" \
        --cfg_file "$CFG" \
        --batch_size 6 \
        --fix_random_seed \
        --run_name "sourceonly_${SOURCE}2nuscenes" \
        --extra_tag "$TAG"
