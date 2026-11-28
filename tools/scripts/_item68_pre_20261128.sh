#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 68, 68-pre (reading fixed 2026-11-28 17:44 JST): item 41a's BN rows (with and without
# AdaBN) by ring on KITTI val, against the S2 oracle O under AdaBN; readings to /home/koyama/data/item68/item68_pre_read.txt.
# CPU only; run from ST3D/tools as a Slurm CPU job.
set -euo pipefail
cd /home/koyama/code/ST3D/tools
OUT=/home/koyama/data/item68
mkdir -p $OUT
run() { singularity exec --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D \
  /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif "$@" < /dev/null; }
run python3 analysis/item68_norm_swap.py bn $OUT/item68_pre_manifest.json
run python3 analysis/item38_kitti_rings.py ap $OUT/item68_pre_manifest.json $OUT/item68_pre_ap.json --workers 4
run python3 analysis/item68_norm_swap.py bnread $OUT/item68_pre_ap.json | tee $OUT/item68_pre_read.txt
