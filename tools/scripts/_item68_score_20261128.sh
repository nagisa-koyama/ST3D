#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 68 (design fixed 2026-11-28 16:42 JST): after the six evaluations, the manifest (LN / GN
# rows + item 41a's BN rows, resolved from the job logs), ring scoring on CPU (item 38's rings; reproduction within 0.02) and
# the readings to /home/koyama/data/item68/item68_read.txt. CPU only; run from ST3D/tools as a Slurm CPU job.
set -euo pipefail
cd /home/koyama/code/ST3D/tools
OUT=/home/koyama/data/item68
mkdir -p $OUT
run() { singularity exec --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D \
  /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif "$@" < /dev/null; }
echo "=== item68 scoring ($(date '+%F %T')) ==="
run python3 analysis/item68_norm_swap.py manifest $OUT/item68_kitti_manifest.json $OUT/item68_source.json
run python3 analysis/item38_kitti_rings.py ap $OUT/item68_kitti_manifest.json $OUT/item68_ap.json --workers 4
run python3 analysis/item68_norm_swap.py read $OUT/item68_ap.json $OUT/item68_source.json | tee $OUT/item68_read.txt
echo "=== item68 scoring done ($(date '+%F %T')) ==="
