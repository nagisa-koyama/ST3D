#!/usr/bin/env bash
# experiments_md 20261003_05 §5 items 61c and 65′ (designs fixed 2026-11-27 15:47 JST): after their rows, inside a 1-GPU
# allocation (the ring scoring's real kernel), run from ST3D/tools:
#   bash scripts/_item61c_65p_score_20261127.sh MOVED666_JOB MOVED667_JOB
# 61c: bitwise check, manifest, ring scoring, readings -> /home/koyama/data/item61/item61c_read.txt
# 65′: bitwise checks ran in the rows; manifest, ring scoring, readings -> /home/koyama/data/item65p/item65p_read.txt
# The two parts run independently (a 65′ row held by its guard leaves 61c's read intact). Do not edit while a job runs it.
set -u
J6=$1; J7=$2
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
RUN="singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D $SIF"
F=0
$RUN bash -c "set -eo pipefail; { python3 analysis/item61c_moved.py bitwise /home/koyama/data/item61/item61c_bitwise.json --jobs moved666=$J6 moved667=$J7 && python3 analysis/item61c_moved.py manifest /home/koyama/data/item61/item61c_manifest.json --jobs moved666=$J6 moved667=$J7 && python3 analysis/item38_kitti_rings.py ap /home/koyama/data/item61/item61c_manifest.json /home/koyama/data/item61/item61c_ap.json --real-kernel --workers 4 && python3 analysis/item61c_moved.py read /home/koyama/data/item61/item61c_ap.json; } 2>&1 | tee /home/koyama/data/item61/item61c_read.txt" || { echo "item61c scoring FAILED"; F=$((F+1)); }
mkdir -p /home/koyama/data/item65p
$RUN bash -c "set -eo pipefail; { python3 analysis/item65_tent.py manifest /home/koyama/data/item65p/item65p_manifest.json --variant prior && python3 analysis/item38_kitti_rings.py ap /home/koyama/data/item65p/item65p_manifest.json /home/koyama/data/item65p/item65p_ap.json --real-kernel --workers 4 && python3 analysis/item65_tent.py read /home/koyama/data/item65p/item65p_ap.json --variant prior; } 2>&1 | tee /home/koyama/data/item65p/item65p_read.txt" || { echo "item65p scoring FAILED"; F=$((F+1)); }
echo "=== item61c / 65p scoring: failures $F ($(date '+%F %T')) ==="
[ $F -eq 0 ]
