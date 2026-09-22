#!/bin/bash
#SBATCH --job-name=accumabl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G               # Sized to the SMALLEST node in the partition list, not to the
                                 # largest: rtx8000 node61/62 have 239G and a6000 node13 299G, so
                                 # a 320G request - which an earlier version of this script made -
                                 # is simply unschedulable on three of the eight candidate nodes
                                 # and would have sat pending behind them forever.
                                 #
                                 # 220G against the 96G the N=1 source-only configs use. The rise
                                 # is NOT proportional to the 17x more points per sample: voxel
                                 # memory is capped by MAX_NUMBER_OF_VOXELS (150000) regardless of
                                 # input size, and the raw point buffers are megabytes. The real
                                 # new cost is the infos - the 200-sweep pickle is 2.75 GB against
                                 # the 10-sweep one's 454 MB, and every DataLoader worker holds a
                                 # copy.
                                 #
                                 # If epoch 2 dies at a worker respawn - SIGSEGV rather than a
                                 # clean OOM, because /dev/shm pages count against the cgroup -
                                 # raise to 290G and drop rtx8000 from the partition list.
#SBATCH --gres=gpu:1             # untyped on purpose: each partition holds one GPU type, so the
                                 # typed form would pin this back to a single partition
#SBATCH --partition=a6000_ada,a6000,rtx8000   # whichever frees first. The container's extensions
                                              # were built with TORCH_CUDA_ARCH_LIST covering 7.5
                                              # (Turing/rtx8000) through 8.9 (Ada), so there is no
                                              # arch mismatch on any of them.
                                              # NOT a100 - job 25536 died instantly there with an
                                              # untyped gres (RuntimeError: No CUDA GPUs available)
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
  --fix_random_seed --run_name "accum_global_nuscenes2kitti_mc" --extra_tag 20260923_accum_global_mc
