#!/usr/bin/env bash
#SBATCH --job-name=pcseg
#SBATCH --partition=a100,a6000_ada,a6000,rtx8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/output_%j_%x.txt
#SBATCH --error=logs/error_%j_%x.txt
#
# pcseg launcher for the segmentation branch's WORKTREE (experiments_md episodes/20261116_01 §8.5).
#
#   cd <worktree>/tools
#   BATCH=<total batch> WANDB_NOTES="..." scripts/submit.sh --nice=100 [sbatch options] \
#       scripts/run_seg.sh cfgs/pcseg/<x>.yaml [extra_tag] [run_name] [extra train_seg.py args...]
#
# Why a new script and not run_sourceonly_2gpu.sh: that launcher hardcodes REPO=/home/koyama/code/ST3D, the
# MAIN checkout on the detection branch, which carries no pcseg. Submitted from a worktree it would freeze the
# wrong tree, silently (memory/repo/parallel_sessions_code_management.md, Direction 1, prerequisite 1).
# What is the same: the frozen-code snapshot, the binds (the snapshot is bound over /home/koyama/code/ST3D, so
# config paths, `import pcdet` and `_init_path` resolve as in the main checkout), NGPU from Slurm, --batch_size
# as the TOTAL across ranks, STAGE_PANDASET. What differs:
#   - REPO is the git work tree that contains $SLURM_SUBMIT_DIR, and the log names its path and branch;
#   - the worktree's `.git` is a FILE pointing into the main repo's .git, which the snapshot bind hides inside
#     the container. The snapshot therefore gets its own copy of the git metadata (about 21 MB), so W&B still
#     records git.commit and diff.patch;
#   - no post-training auto-recovery: train_seg.py never prints the detection `End training` marker, and that
#     path runs eval_checkpoint.sh from the main checkout.
set -euo pipefail

CFG=${1:?usage: scripts/run_seg.sh cfgs/pcseg/<x>.yaml [extra_tag] [run_name] [extra args...]}
TAG=${2:-$(date +%Y%m%d)_pcseg}
RUN_NAME=${3:-$(basename "$CFG" .yaml)}
EXTRA_ARGS=("${@:4}")
ENTRY=${ENTRY:-train_seg.py}
BATCH=${BATCH:?set BATCH, the TOTAL batch across ranks (12 for the SpUNet recipe)}
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif

REPO=$(git -C "${SLURM_SUBMIT_DIR:?run under sbatch}" rev-parse --show-toplevel)
cd "$REPO/tools"
[ -f "$CFG" ] || { echo "no such config: $CFG" >&2; exit 2; }
[ -f "$ENTRY" ] || { echo "no $ENTRY in $REPO/tools: this tree carries no pcseg (submitted from the main checkout?)" >&2; exit 2; }
BRANCH=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
GIT_DIR_ABS=$(git -C "$REPO" rev-parse --absolute-git-dir)
GIT_COMMON=$(cd "$REPO" && cd "$(git rev-parse --git-common-dir)" && pwd)

# --- freeze the code for the life of this job (as run_sourceonly_2gpu.sh) ---------------------
if [ -d "/local_cache/${SLURM_JOB_ID}" ] && [ -w "/local_cache/${SLURM_JOB_ID}" ]; then
    SCRATCH=/local_cache/${SLURM_JOB_ID}
else
    SCRATCH=/tmp/st3d_${SLURM_JOB_ID}
    echo "=== /local_cache/${SLURM_JOB_ID} unavailable on $(hostname) - falling back to $SCRATCH ==="
    trap 'rm -rf "/tmp/st3d_${SLURM_JOB_ID}"' EXIT
fi
mkdir -p "$SCRATCH" || { echo "cannot create scratch dir $SCRATCH" >&2; exit 1; }
SNAP=$SCRATCH/ST3D
mkdir -p "$SNAP"
rsync -a \
  --exclude='/output/' --exclude='/wandb/' --exclude='/build/' \
  --exclude='tools/wandb/' --exclude='tools/logs/' \
  --exclude='*.sif' --exclude='.nfs*' --exclude='pcdet.egg-info/' \
  "$REPO/" "$SNAP/"
if [ -f "$SNAP/.git" ]; then
    # A linked worktree: copy the common git dir into the snapshot and point the snapshot's .git file at the
    # copy, by the path the snapshot has INSIDE the container (/home/koyama/code/ST3D).
    rsync -a --exclude='*.lock' "$GIT_COMMON/" "$SNAP/.git_common/"
    REL=${GIT_DIR_ABS#"$GIT_COMMON"/}
    [ "$REL" != "$GIT_DIR_ABS" ] || { echo "git dir $GIT_DIR_ABS is not under $GIT_COMMON" >&2; exit 1; }
    echo "gitdir: /home/koyama/code/ST3D/.git_common/$REL" > "$SNAP/.git"
    mkdir -p "$SNAP/.git_common/info" && echo ".git_common/" >> "$SNAP/.git_common/info/exclude"
fi
echo "=== code frozen at $(git -C "$REPO" rev-parse --short HEAD) (branch $BRANCH, tree $REPO) -> $SNAP ==="

# --- optional: stage PandaSet on the node's local SSD (as run_sourceonly_2gpu.sh) ---------------
DATA_BIND=()
if [ "${STAGE_PANDASET:-0}" = "1" ]; then
    SRC_DATA=/home/koyama/data/pandaset
    STAGE=$SCRATCH/pandaset
    NEED_KB=$(du -sk --exclude=camera --exclude=gt_database "$SRC_DATA" | awk '{print $1}')
    FREE_KB=$(df -Pk "$SCRATCH" | awk 'NR==2 {print $4}')
    if [ "$FREE_KB" -gt $((NEED_KB + 10485760)) ]; then
        mkdir -p "$STAGE"
        echo "=== staging $((NEED_KB/1048576)) GiB -> $STAGE (free $((FREE_KB/1048576)) GiB) ==="
        T0=$SECONDS
        rsync -a --exclude='camera/' --exclude='gt_database/' "$SRC_DATA/" "$STAGE/"
        echo "=== staged in $((SECONDS-T0)) s ==="
        DATA_BIND=(--bind "$STAGE":"$SRC_DATA")
    else
        echo "=== NOT staging: need $((NEED_KB/1048576)) GiB, only $((FREE_KB/1048576)) GiB free - using NFS ===" >&2
    fi
fi

PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
NGPU=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | /usr/bin/grep -c '^GPU' || echo 1)}
echo "=== allocated GPUs: $NGPU ==="
if [ "$NGPU" -le 1 ]; then
    LAUNCH=(python "$ENTRY")
else
    LAUNCH=(python -m torch.distributed.launch --use-env --nproc_per_node="$NGPU"
            --rdzv_endpoint=localhost:"$PORT"
            "$ENTRY" --launcher pytorch --tcp_port "$PORT")
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
        --batch_size "$BATCH" \
        --fix_random_seed \
        --run_name "$RUN_NAME" \
        --extra_tag "$TAG" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
