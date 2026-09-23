#!/usr/bin/env bash
#SBATCH --job-name=ptsn
#SBATCH --partition=a6000_ada,a6000,rtx8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=logs/output_%j_%x.txt
#SBATCH --error=logs/error_%j_%x.txt
#
# DALI Tier D1: the PTSN scale search. INFERENCE ONLY - one forward pass over --frames target
# frames per candidate scale, no training - which is what makes it the cheapest new number in the
# DALI plan and why it runs before any of Tier D2 is written. If the sweep moves the predicted
# mean size onto the estimate but target AP does not follow, the distribution-level story does not
# hold here and D2 is not justified (experiments_md/20260922_04 section 2.4's stopping rule).
#
# --ros_factor, NOT --target_size: ROS only ever perturbs the SOURCE, so the row stays target-free
# and comparable with our own rows. A --target_size taken from SN would be the target statistic and
# the row would not be UDA-legal; the script refuses to guess between them and prints the
# provenance into its own log.
#
# Inputs were fixed by the CPU dry run (20260922_04 section 2.7):
#   checkpoint  run mxdvmi16 epoch 30, nuScenes -> KITTI
#   source mean Car [4.426, 1.844, 1.630], x 0.86 -> estimate [3.806, 1.586, 1.402]
# against KITTI's true [3.854, 1.638, 1.523]. Read the result knowing the gap is ANISOTROPIC
# (1.148 / 1.126 / 1.070 per dimension), so an isotropic scale cannot close it and the estimate
# runs ~12 cm low on height.
set -euo pipefail

SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
cd /home/koyama/code/ST3D/tools

singularity exec --nv --bind /home/koyama/data/:/storage "$SIF" \
    python3 analysis/ptsn_search.py \
        --cfg_file cfgs/da-MIRU2025/second_old_anchor_basebev_multi_nuscenes2kitti_car_ped_default.yaml \
        --ckpt /storage/wandb/run-20260921_031539-mxdvmi16/files/ckpt/checkpoint_epoch_30.pth \
        --ros_factor 0.86 \
        --frames 500 \
        --yaml
