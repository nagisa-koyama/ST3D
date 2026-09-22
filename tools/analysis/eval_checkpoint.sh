#!/usr/bin/env bash
# Evaluate one already-trained checkpoint, without retraining.
#
#   sbatch --partition=a6000_ada,a6000,rtx8000 --gres=gpu:1 --cpus-per-task=8 --mem=64G \
#          --time=00:45:00 --job-name=eval_ckpt --output=logs/output_%j_%x.txt \
#          --error=logs/error_%j_%x.txt \
#          --wrap="bash analysis/eval_checkpoint.sh <cfg> <ckpt.pth> [eval_tag]"
#
# NOTE the script lives in the repo, not in a scratch dir: /tmp on the login node is NOT shared
# with the compute nodes, so an sbatch --wrap pointing at /tmp/... dies instantly with exit 127.
#
# SINGLE GPU on purpose, even for a family that trains on two. `--launcher none` means DataLoader
# workers are FORKED, inheriting the parent's already-imported modules, so nothing re-reads the
# repo mid-run. Under DDP, init_dist_pytorch forces mp.set_start_method('spawn'), and a spawned
# worker starts a fresh interpreter that re-imports from the live bind-mounted checkout - which is
# exactly how job 25743 died after 8 h of clean training (a commit landed 37 min before its eval,
# so the workers got the new class while unpickling an instance built by the old one). Evaluation
# is also dominated by single-threaded AP computation that DDP cannot speed up anyway.
set -u
CFG=${1:?usage: eval_checkpoint.sh <cfg_file> <ckpt.pth> [eval_tag]}
CKPT=${2:?usage: eval_checkpoint.sh <cfg_file> <ckpt.pth> [eval_tag]}
TAG=${3:-recover_$(date +%Y%m%d_%H%M%S)}
cd /home/koyama/code/ST3D/tools
[ -f "$CFG" ]  || { echo "no such config: $CFG" >&2; exit 2; }
[ -f "$CKPT" ] || { echo "no such checkpoint: $CKPT" >&2; exit 2; }

singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D \
  /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif \
  python3 test.py --cfg_file "$CFG" --ckpt "$CKPT" --batch_size 6 \
    --extra_tag 20260922_sourceonly --eval_tag "$TAG"
