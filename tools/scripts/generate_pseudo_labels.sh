#!/usr/bin/env bash
#SBATCH --job-name=gen_ps
#SBATCH --partition=a6000_ada,a6000,rtx8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/output_%j_%x.txt
#SBATCH --error=logs/error_%j_%x.txt
#
# One teacher pass over a self-training config's target, storing EVERY emitted box (score floor
# 0.0001) so the pseudo-label score distribution is not truncated at NEG_THRESH. No W&B run.
# ~20 min on one GPU for nuScenes train (28,130 frames). Submit from tools/:
#
#   sbatch --comment="<purpose>" scripts/generate_pseudo_labels.sh [cfg] [teacher_ckpt] [out_dir] [thresh]
set -euo pipefail
CFG=${1:-cfgs/da-ieee-access/centerpoint-foreground-lyft2nuscenes.yaml}
CKPT=${2:-/storage/wandb/run-20260923_093504-iwg6l5v1/files/ckpt/checkpoint_epoch_30.pth}
OUT=${3:-/storage/pseudo_labels/iwg6l5v1_ep30_nuscenes_train_thr0.0001}
THR=${4:-0.0001}
MAXOBJ=${5:-}
cd /home/koyama/code/ST3D/tools
singularity exec --nv --bind /home/koyama/data/:/storage \
  /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif \
  python analysis/pseudo_label_threshold/generate_pseudo_labels.py \
    --cfg_file "$CFG" --teacher_ckpt "$CKPT" --out_dir "$OUT" --thresh "$THR" ${MAXOBJ:+--max_obj $MAXOBJ}
