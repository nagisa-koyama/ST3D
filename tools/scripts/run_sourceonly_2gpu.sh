#!/usr/bin/env bash
#SBATCH --job-name=sourceonly
#SBATCH --partition=a6000_ada,a6000,rtx8000
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=96G               # enough for the five N=1 source-only rows (~10 GB resident each).
                                # An accumulating row needs more, so override it on the command
                                # line: `sbatch --mem=150G ...` for a 2-GPU accum row. 96G still
                                # covers the same row on ONE GPU. Keep the value under the
                                # SMALLEST node of whatever --partition list is in force - 239G
                                # rtx8000, 299G a6000, 478G a6000_ada; 150G is chosen to stay
                                # rtx8000-eligible, and 220-250G was not - it silently gave up a
                                # whole partition for headroom nothing used.
                                #
                                # MEASURED, 2026-09-23, not estimated - `sacct`/`sstat` cannot
                                # tell you, because this cluster runs JobAcctGatherType=(null)
                                # and never records MaxRSS for any job. Building the real dataset
                                # objects from centerpoint-accum-global-nuscenes2kitti.yaml:
                                # NUSCENES_N008 +7.28 GB, NUSCENES_N015 +3.03 GB, main process
                                # 10.90 GB. Two source loaders at NUM_WORKERS 4 means 8 workers,
                                # and DDP forces SPAWN, so every worker carries a full pickled
                                # copy with no copy-on-write sharing: ~52 GB on 1 GPU, ~104 GB on
                                # 2, plus CUDA contexts. 150G is ~36% headroom over that, which
                                # /dev/shm IPC wants (see singularity_usage_and_tips.md Gotcha #2
                                # - under-requesting surfaces as an untraceable segfault at an
                                # epoch boundary, never as an OOM message).
                                #
                                # The earlier note here said the 200-sweep infos are "2.75 GB per
                                # process". That was the ON-DISK pickle size and understates the
                                # real cost by 4.3x: nuscenes_infos_200sweeps_train.pkl is 2.56
                                # GB on disk and 11.0 GB resident once loaded (12.0 GB peak).
#SBATCH --time=48:00:00
#SBATCH --output=logs/output_%j_%x.txt
#SBATCH --error=logs/error_%j_%x.txt
#
# Launch one da-ieee-access row at a GLOBAL batch of 6, on however many GPUs Slurm allocates.
# 2 by default; `sbatch --gres=gpu:1 ...` runs the same recipe on one, which is the same
# experiment at roughly twice the wall-clock (see the NGPU branch near the bottom).
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
SOURCE=${1:?usage: sbatch scripts/run_sourceonly_2gpu.sh <kitti|lyft|nuscenes|pandaset|waymo|path/to.yaml> [extra_tag] [run_name]}
TAG=${2:-$(date +%Y%m%d)_sourceonly}
case "$SOURCE" in
  # A config PATH names itself. The five-source shorthand keeps the "X -> nuScenes" run name it
  # has always had, but a path can be any row in the family - an ablation, an oracle - and
  # calling those "sourceonly_<x>2nuscenes" would put a wrong claim in the W&B run name, which is
  # what a reader sorts by. $3 overrides either.
  *.yaml) CFG=$SOURCE; SOURCE=$(basename "$CFG" .yaml)
          RUN_NAME=${3:-$(basename "$CFG" .yaml | sed 's/^centerpoint-//')} ;;
  *)      CFG=cfgs/da-ieee-access/centerpoint-sourceonly-${SOURCE}.yaml
          RUN_NAME=${3:-sourceonly_${SOURCE}2nuscenes} ;;
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

# --- optional: stage the dataset on the node's local SSD ----------------------------------------
#
# STAGE_PANDASET=1 copies PandaSet's lidar/annotations/meta into this job's scratch and binds it
# over the NFS copy. Off by default, so every existing invocation is unchanged.
#
# WHY. Measured 2026-09-23: a COLD read of a PandaSet lidar file over NFS costs 47 ms against 2 ms
# warm - 28x - and MAX_SWEEPS 5 reads FIVE files per sample. Training walks 4,880 distinct frames
# an epoch, so it is cold essentially throughout, and job 25814 measured the accumulating row at
# 86.8% data-wait with 8 workers and still 62.6% with 16. Workers cannot fix a per-read latency;
# they only buy more readers waiting in parallel. Moving the reads off NFS attacks the 28x itself.
#
# SIZE. 18 GiB for the 61 train sequences, 32 GiB for all 103, against 3.4 TB free on node13's
# /local_cache (probe job 25818, a 3.6 TB ext4 SSD at 1% used). `camera/` (10 GiB) and
# `gt_database/` are excluded - this pipeline reads neither.
#
# REMOVAL needs no handling. Slurm creates /local_cache/$SLURM_JOB_ID per job and deletes it when
# the job ends; the probe saw directories for RUNNING jobs only, none of this session's finished
# ones. Nothing is written back, so an interrupted job loses nothing but the copy.
#
# The bind targets the SYMLINK TARGET, /home/koyama/data/pandaset, not the repo's data/pandaset -
# both the relative config path and PandaSet's baked-in absolute /root/ST3D/... paths resolve
# through that symlink, so one bind covers both.
DATA_BIND=()
if [ "${STAGE_PANDASET:-0}" = "1" ]; then
    SRC_DATA=/home/koyama/data/pandaset
    STAGE=/local_cache/${SLURM_JOB_ID}/pandaset
    NEED_KB=$(du -sk --exclude=camera --exclude=gt_database "$SRC_DATA" | awk '{print $1}')
    FREE_KB=$(df -Pk /local_cache | awk 'NR==2 {print $4}')
    # 10 GiB of headroom over the copy, so staging never fills the disk under another job.
    if [ "$FREE_KB" -gt $((NEED_KB + 10485760)) ]; then
        mkdir -p "$STAGE"
        echo "=== staging $((NEED_KB/1048576)) GiB -> $STAGE (free $((FREE_KB/1048576)) GiB) ==="
        T0=$SECONDS
        rsync -a --exclude='camera/' --exclude='gt_database/' "$SRC_DATA/" "$STAGE/"
        echo "=== staged in $((SECONDS-T0)) s ==="
        DATA_BIND=(--bind "$STAGE":"$SRC_DATA")
    else
        # Fall back rather than fail: a slow run beats no run, and the reason is printed.
        echo "=== NOT staging: need $((NEED_KB/1048576)) GiB, only $((FREE_KB/1048576)) GiB free - using NFS ===" >&2
    fi
fi

# Random port so two of these can run concurrently without a rendezvous collision.
PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))

# The /root/ST3D bind is only strictly needed by PandaSet, whose infos bake in absolute
# /root/ST3D paths, but it is harmless elsewhere and keeps one invocation for all five.
# Both binds point at the SNAPSHOT: the first shadows the live repo at its own path (so relative
# config paths and `import pcdet` resolve to frozen code), the second satisfies PandaSet's infos,
# which bake absolute /root/ST3D/... paths. --pwd is explicit so cwd cannot resolve elsewhere.
# How many GPUs Slurm actually gave us, so `sbatch --gres=gpu:1 ...` does the right thing instead
# of starting a 2-rank rendezvous against one device and hanging. The queue is often full enough
# that a 1-GPU slot opens long before a 2-GPU one, and the two are INTERCHANGEABLE for results:
# --batch_size is the TOTAL across ranks, so 6 is a global batch of 6 either way - 6 per rank on
# one GPU, 3 per rank on two - with the same optimizer step count and the same LR. Only wall-clock
# differs, at close to 2x (measured scaling is 2.03-2.06x, experiments_md/20260922_06).
NGPU=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 1)}
echo "=== allocated GPUs: $NGPU ==="

if [ "$NGPU" -le 1 ]; then
    # Single GPU: no launcher, no DDP. Also immune to the mid-run code-edit failure that the
    # snapshot above guards against, since a forked worker inherits the parent's already-imported
    # modules rather than re-importing from disk - the snapshot is kept anyway, for one behaviour
    # across both paths.
    LAUNCH=(python train.py)
else
    LAUNCH=(python -m torch.distributed.launch --use-env --nproc_per_node="$NGPU"
            --rdzv_endpoint=localhost:"$PORT"
            train.py --launcher pytorch --tcp_port "$PORT")
fi

singularity exec --nv \
    --bind /home/koyama/data/:/storage \
    ${DATA_BIND[@]+"${DATA_BIND[@]}"} \
    --bind "$SNAP":/home/koyama/code/ST3D \
    --bind "$SNAP":/root/ST3D \
    --pwd /home/koyama/code/ST3D/tools \
    "$SIF" \
    "${LAUNCH[@]}" \
        --cfg_file "$CFG" \
        --batch_size 6 \
        --fix_random_seed \
        --run_name "$RUN_NAME" \
        --extra_tag "$TAG"
