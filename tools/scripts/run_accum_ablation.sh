#!/bin/bash
#SBATCH --job-name=accumabl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G               # NOT the 96G the N=1 sourceonly configs use. MAX_SWEEPS 9 makes
                                 # each sample ~224k points against ~22k at N=1, a 10x rise in what
                                 # every DataLoader worker holds and ships through /dev/shm - and
                                 # 64G already segfaulted centerpoint-sourceonly at N=1 (see
                                 # run_experiment.sh's header). Raise further if epoch 2 dies at a
                                 # worker respawn; that is the signature.
#SBATCH --gres=gpu:1             # untyped on purpose: each partition holds one GPU type, so the
                                 # typed form would pin this back to a single partition
#SBATCH --partition=a6000_ada,a6000   # NOT a100 - job 25536 died instantly there with an untyped
                                      # gres (RuntimeError: No CUDA GPUs are available)
#SBATCH --output=logs/output_%j_accum.txt
#SBATCH --error=logs/error_%j_accum.txt
#SBATCH --time=99:00:00

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error
set -e

# nuScenes -> KITTI accumulation / density-correction ablation. One active line at a time, as in
# run_experiment.sh; comment out finished ones with the job id and outcome rather than deleting.
#
#   1  accumulation only        centerpoint-accum-nuscenes2kitti.yaml
#   2  + global correction      centerpoint-accum-global-nuscenes2kitti.yaml     <- active
#   3  + foreground-aware       centerpoint-accum-foreground-nuscenes2kitti.yaml (needs
#                               adaptive_train.py and a teacher checkpoint from variant 1)

singularity exec --nv --bind /home/koyama/data/:/storage \
  /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif \
  python3 train.py --cfg_file cfgs/da-ieee-access/centerpoint-accum-global-nuscenes2kitti.yaml \
  --fix_random_seed --run_name "accum_global_nuscenes2kitti" --extra_tag 20260922_accum_global
