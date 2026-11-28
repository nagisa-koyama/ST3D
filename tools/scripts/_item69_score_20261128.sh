#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 69 (design fixed 2026-11-28 17:51 JST): manifest, ring scoring (CPU IoU op) and readings
# to /home/koyama/data/item69/item69_read.txt. CPU only; run from ST3D/tools as a Slurm CPU job.
#   bash scripts/_item69_score_20261128.sh <eval job 666> <eval job 667> <eval job 668>
set -euo pipefail
cd /home/koyama/code/ST3D/tools
OUT=/home/koyama/data/item69
run() { singularity exec --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D \
  /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif "$@" < /dev/null; }
run python3 analysis/item69_real_bn.py manifest $OUT/item69_manifest.json --jobs 666=$1 667=$2 668=$3
run python3 analysis/item38_kitti_rings.py ap $OUT/item69_manifest.json $OUT/item69_ap.json --workers 4
run python3 analysis/item69_real_bn.py read $OUT/item69_ap.json | tee $OUT/item69_read.txt
