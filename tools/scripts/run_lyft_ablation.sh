#!/bin/bash
#SBATCH --job-name=lyftabl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G               # Far below the nuScenes ablation's 220G: this pair runs at
                                 # MAX_SWEEPS 1, so a sample is ~61k points on 40-beam and ~108k
                                 # on 64-beam rather than ~374k, and the Lyft infos are small
                                 # where the 200-sweep nuScenes pickle is 2.75 GB per worker.
                                 # 120G against the 96G run_experiment.sh proves for Lyft at N=1,
                                 # and it clears rtx8000's 239G floor comfortably.
#SBATCH --gres=gpu:1             # untyped: each partition holds one GPU type, so the typed form
                                 # would pin this back to a single partition
#SBATCH --partition=a6000_ada,a6000,rtx8000   # NOT a100 - broken at the node level, cuInit()
                                              # returns CUDA_ERROR_NO_DEVICE on node21
#SBATCH --output=logs/output_%j_lyft.txt
#SBATCH --error=logs/error_%j_lyft.txt
#SBATCH --time=99:00:00

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error
set -e

# Lyft -> nuScenes correction ablation, both platforms as sources. One active line at a time;
# comment out finished ones with the job id and outcome rather than deleting.
#
#   1  no correction        centerpoint-lyft2nuscenes.yaml
#   2  + global             centerpoint-global-lyft2nuscenes.yaml            <- active
#   3  + foreground-aware   centerpoint-foreground-lyft2nuscenes.yaml (adaptive_train.py; needs
#                           --pretrained_model / --pretrained_model_teacher from a run above)

singularity exec --nv --bind /home/koyama/data/:/storage \
  /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif \
  python3 train.py --cfg_file cfgs/da-ieee-access/centerpoint-global-lyft2nuscenes.yaml \
  --fix_random_seed --run_name "lyft_global_2nuscenes_v2" --extra_tag 20260923_lyft_global_v2
