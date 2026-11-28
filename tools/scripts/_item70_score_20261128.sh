#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 70 (design fixed 2026-11-28 20:10 JST): manifest, ring scoring (CUDA kernel) and readings
# to /home/koyama/data/item70/item70_read.txt. Run from ST3D/tools as a short 1-GPU job (the CUDA IoU kernel; the CPU op missed
# by 0.046 on 41a seed 666 in 68-pre 37454).
#   bash scripts/_item70_score_20261128.sh
set -euo pipefail
cd /home/koyama/code/ST3D/tools
OUT=/home/koyama/data/item70
mkdir -p $OUT
run() { singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D \
  /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif "$@" < /dev/null; }
run python3 analysis/item70_rs_adabn_real.py manifest $OUT/item70_manifest.json || echo "manifest: some rows missing (read anyway)"
run python3 analysis/item38_kitti_rings.py ap $OUT/item70_manifest.json $OUT/item70_ap.json --real-kernel --workers 4
run python3 analysis/item70_rs_adabn_real.py read $OUT/item70_ap.json | tee $OUT/item70_read.txt
