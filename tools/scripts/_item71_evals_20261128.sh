#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 71 (design fixed 2026-11-28 21:31 JST, before any number): the 30 KITTI val evaluations of
# item 71's six RS-BN rows at epoch 20 (arms none / ada / none_s / ada_s / glob through item60_rsbn_eval.py), then the
# manifest, ring scoring (CUDA kernel) and readings to /home/koyama/data/item71/item71_read.txt.
# Usage (from ST3D/tools, inside a 1-GPU allocation, on the host; it calls singularity itself):
#   bash scripts/_item71_evals_20261128.sh <st666> <st667> <st668> <stg666> <stg667> <stg668>            # the job
#   bash scripts/_item71_evals_20261128.sh smoke <ckpt>    # stg-config arms ada / ada_s on 16 frames (40 statistics frames)
# Each evaluation logs its exit status and the next runs regardless. Do not edit while a job runs it.
set -u
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
OUT=/home/koyama/data/item71
mkdir -p $OUT
run() { singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF" "$@" < /dev/null; }
H="Item 71 (T05 §5 item 71, design fixed 2026-11-28 21:31 JST): items 60 / 67 on nuScenes->KITTI"
if [ "${1:-}" = smoke ]; then
  C=${2:?usage: smoke <ckpt>}
  for A in ada:1.0 ada:0.301; do
    AD=${A%%:*}; SC=${A##*:}
    echo "=== item71 eval smoke $AD scale $SC on $C ($(date '+%F %T')) ==="
    WANDB_NOTES="$H. SMOKE of the evaluation path on an epoch-1 checkpoint: adapt $AD, range scale $SC, 16 frames." \
      run python3 analysis/item60_rsbn_eval.py --cfg_file cfgs/da-ieee-access-analysis/item71/centerpoint-item71-nuscenes2kitti-stg.yaml \
        --ckpt "$C" --adapt "$AD" --range_scale "$SC" --out_dir $OUT/smoke/${AD}_x$SC --run_name i71_smoke_${AD}_x$SC \
        --smoke --stats_frames 40
    echo "exit $?"
  done
  run python3 analysis/item71_rsbn_real.py ckptcheck "$C"
  exit $?
fi
[ $# -eq 6 ] || { echo "usage: $0 <st666> <st667> <st668> <stg666> <stg667> <stg668>"; exit 2; }
declare -A JOB=([st666]=$1 [st667]=$2 [st668]=$3 [stg666]=$4 [stg667]=$5 [stg668]=$6)
F=0
SPEC=$(run python3 analysis/item71_rsbn_real.py spec) || { echo "item71: no spec"; exit 2; }
while read -r V ARM SEED ADAPT SCALE DIR NAME CFG; do
  [ -n "$CFG" ] || continue
  J=${JOB[$V$SEED]}
  C=$(bash analysis/ckpt_of_job.sh "$J" 20) || { echo "item71 $NAME: no epoch-20 checkpoint of $J"; F=$((F+1)); continue; }
  echo "=== item71 $NAME (row $J, variant $V, seed $SEED, adapt $ADAPT, scale $SCALE) ($(date '+%F %T')) ==="
  WANDB_NOTES="$H. Row $J ($V, seed $SEED, epoch 20) on KITTI val, arm $ARM: adapt $ADAPT, knots looked up at $SCALE x range ($([ "$ADAPT" = none ] && echo 'the source per-knot statistics, no target data' || echo 're-estimated on 1000 strided KITTI train clouds, no labels')). Readings fixed in T05 item 71." \
    run python3 analysis/item60_rsbn_eval.py --cfg_file "$CFG" --ckpt "$C" --adapt "$ADAPT" --range_scale "$SCALE" \
      --out_dir "$DIR" --run_name "$NAME"
  s=$?; echo "exit $s"; [ $s -eq 0 ] || F=$((F+1))
done <<< "$SPEC"
echo "=== item71 evaluations done: failures $F ($(date '+%F %T')) ==="
run python3 analysis/item71_rsbn_real.py manifest $OUT/item71_manifest.json --jobs st666=$1 st667=$2 st668=$3 stg666=$4 stg667=$5 stg668=$6
run python3 analysis/item38_kitti_rings.py ap $OUT/item71_manifest.json $OUT/item71_ap.json --real-kernel --workers 4
run python3 analysis/item71_rsbn_real.py read $OUT/item71_ap.json | tee $OUT/item71_read.txt
echo "=== item71 done ($(date '+%F %T')) ==="
