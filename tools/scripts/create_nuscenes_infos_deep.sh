#!/bin/bash
#SBATCH --job-name=nsinfo200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G               # the whole infos list is built in RAM before pickling; at 199
                                 # sweeps x ~34k samples that is millions of dicts each holding
                                 # three 4x4 matrices
#SBATCH --partition=a6000_ada,a6000
#SBATCH --output=logs/output_%j_nsinfo.txt
#SBATCH --error=logs/error_%j_nsinfo.txt
#SBATCH --time=99:00:00
# No --gres: this is metadata only. It walks the nuScenes DB's prev-pointers and composes
# transforms; it never loads a point cloud and never touches a GPU.
#
# Runs under sbatch precisely so it survives an ssh disconnect - the scheduler owns it, not the
# login shell.
set -e

singularity exec --bind /home/koyama/data/:/storage \
  /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif \
  python3 scripts/create_nuscenes_infos_deep.py --max_sweeps 200
