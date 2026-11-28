#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 72 (design fixed 2026-11-29 00:15 JST, before any number): the CPU checks before
# any GPU row. Builds Pandar64's table (TRAIN clouds of the oracle's training block, label-free, against the nuScenes
# histogram stored in nuscenes2waymo_rings.npz), checks that the three tables share one nuScenes side exactly, then runs
# item72_intensity_checks.py per target (identity, raw, direction, evaluation path; 40 frames). Output:
# /home/koyama/data/item72/checks.txt. Run as a Slurm CPU job from ST3D/tools, on the host (it calls singularity).
# Do not edit while a job runs it.
set -u
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
OUT=/home/koyama/data/item72
mkdir -p $OUT
run() { singularity exec --bind /home/koyama/data/:/storage "$SIF" "$@" < /dev/null; }
T=/storage/intensity_tables
F=0
echo "=== item72 table for Pandar64 ($(date '+%F %T')) ==="
if [ -e /home/koyama/data/intensity_tables/nuscenes2pandaspin_rings.npz ]; then
  echo "nuscenes2pandaspin_rings.npz exists: not rebuilt"
else
  run python3 analysis/intensity_testtime_tables.py --source_hist_from $T/nuscenes2waymo_rings.npz \
    --target_cfg cfgs/da-ieee-access/centerpoint-pandaset-spin2spin-xyzi.yaml --target_block DATA_CONFIG \
    --out_prefix $T/nuscenes2pandaspin --frames 300 || F=$((F+1))
fi
run python3 analysis/item72_intensity_checks.py --same_nuscenes_side $T/nuscenes2kitti_rings.npz \
  $T/nuscenes2waymo_rings.npz $T/nuscenes2pandaspin_rings.npz || F=$((F+1))
for X in kitti waymo pandaspin; do
  echo "=== item72 checks $X ($(date '+%F %T')) ==="
  run python3 analysis/item72_intensity_checks.py --target $X --frames 40 || F=$((F+1))
done
echo "=== item72 checks done: failures $F ($(date '+%F %T')) ==="
[ $F -eq 0 ]
