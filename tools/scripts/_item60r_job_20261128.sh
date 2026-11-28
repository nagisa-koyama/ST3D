#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 60r (design fixed 2026-11-28 15:32 JST, before any number): item 60's RS-BN checkpoints
# (37227 / 37228, epoch 152) with their knots looked up at s * range on the target (item60_rsbn_eval.py --range_scale), s =
# 0.4 / 1.33 = 0.301 (plus 0.251, descriptive, and the identity s = 1 on seed 666), then the manifest (these rows + item 60g's),
# ring scoring (CUDA kernel) and the readings to /home/koyama/data/item60/item60r_read.txt.
# Usage (from ST3D/tools, inside a 1-GPU allocation, on the host: it calls singularity itself):
#   bash scripts/_item60r_job_20261128.sh            # the job
#   bash scripts/_item60r_job_20261128.sh smoke      # seed 666, none_s and ada_s, 16 evaluation frames, 40 statistics frames
# Do not edit while a job runs it.
set -euo pipefail
MODE=${1:-full}
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
CFG=cfgs/da-ieee-access-analysis/item60/centerpoint-item60-pk-rsbn.yaml
OUT=/home/koyama/data/item60
run() { singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF" "$@" < /dev/null; }
declare -A JOBS=([666]=37227 [667]=37228)
SPEC=$(python3 analysis/item60r_rowstep.py spec)
while read -r ARM SEED SCALE ADAPT DIR NAME; do
  EXTRA=()
  if [ "$MODE" = smoke ]; then
    [ "$SEED" = 666 ] && { [ "$ARM" = none_s ] || [ "$ARM" = ada_s ]; } || continue
    DIR=$OUT/smoke/${ARM}_s$SEED; NAME=${NAME}_smoke; EXTRA=(--smoke --stats_frames 40)
  fi
  JOB=${JOBS[$SEED]}
  C=$(bash analysis/ckpt_of_job.sh "$JOB" 152)
  echo "=== item60r $ARM seed $SEED, scale $SCALE, adapt $ADAPT, job $JOB, checkpoint $C -> $DIR ($(date '+%F %T')) ==="
  WANDB_NOTES="Item 60r (T05 §5, design 2026-11-28 15:32 JST), labelled ANALYSIS: item 60's RS-BN row $JOB seed $SEED (epoch 152) with its knots looked up at $SCALE x range on KITTI val (row step of the HDL-64E mapped to the S' lattice; 0.301 = 0.4/1.33 from the specs), adapt $ADAPT ($([ "$ADAPT" = ada ] && echo 'per-knot re-estimation on 1000 strided KITTI train clouds, no labels' || echo 'the source per-knot statistics, no target data')). Readings fixed in T05 item 60r.$([ "$MODE" = smoke ] && echo ' SMOKE: 16 frames.')" \
    run python3 analysis/item60_rsbn_eval.py --cfg_file "$CFG" --ckpt "$C" --adapt "$ADAPT" --range_scale "$SCALE" \
      --out_dir "$DIR" --run_name "$NAME" "${EXTRA[@]}"
done <<< "$SPEC"
[ "$MODE" = smoke ] && { echo "=== item60r smoke done ($(date '+%F %T')) ==="; exit 0; }
run python3 analysis/item60r_rowstep.py manifest $OUT/item60r_manifest.json --jobs s666=37227 s667=37228
run python3 analysis/item38_kitti_rings.py ap $OUT/item60r_manifest.json $OUT/item60r_ap.json --real-kernel --workers 4
run python3 analysis/item60r_rowstep.py read $OUT/item60r_ap.json | tee $OUT/item60r_read.txt
echo "=== item60r done ($(date '+%F %T')) ==="
