#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 59b job P0 (design fixed 2026-11-27 08:15 JST, before any number): for each seed, the
# frozen teacher P_K + global AdaBN (SELF_TRAIN.TEACHER_ADABN of the base config, statistics from KITTI testing clouds) labels
# every KITTI testing frame at the score floor 0.0001; then rule A (background plateau, label-free) per class gives the cuts
# (/home/koyama/data/item59/cuts.json). No label exists for these frames; none is read. Usage (from ST3D/tools, 1 GPU):
#   bash scripts/_item59b_p0_20261127.sh
# Do not edit while a job runs it.
set -u
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
BASE=cfgs/da-ieee-access-analysis/item59/centerpoint-item59-b-base.yaml
F=0
for pair in 666:32283 667:32284; do
  seed=${pair%%:*}; job=${pair##*:}
  echo "=== item59b P0 seed $seed (P_K $job) ($(date '+%F %T')) ==="
  C=$(bash analysis/ckpt_of_job.sh "$job" 152) || { echo "no checkpoint"; F=$((F+1)); continue; }
  singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF" \
    python3 analysis/pseudo_label_threshold/generate_pseudo_labels.py --cfg_file "$BASE" --teacher_ckpt "$C" \
      --out_dir /home/koyama/data/item59/p0_s$seed --thresh 0.0001 < /dev/null
  s=$?; echo "exit $s"; [ $s -eq 0 ] || F=$((F+1))
done
[ $F -eq 0 ] || { echo "item59b P0: $F failures, no cuts"; exit 1; }
singularity exec --bind /home/koyama/data/:/storage "$SIF" python3 analysis/item59_bn_split.py cuts
