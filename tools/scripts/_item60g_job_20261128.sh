#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 60g (design fixed 2026-11-28 10:38 JST, before any number): item 60's RS-BN checkpoints
# (37227 / 37228, epoch 152) under one pooled set of BN statistics (item60_rsbn_eval.py --adapt glob, 1,000 strided KITTI
# train clouds, no labels), then the manifest (glob rows + item 60's), ring scoring (CUDA kernel) and the readings.
# Usage (from ST3D/tools, inside a 1-GPU allocation, on the host: it calls singularity itself):
#   bash scripts/_item60g_job_20261128.sh            # the job
#   bash scripts/_item60g_job_20261128.sh smoke      # seed 666 only, 16 evaluation frames, 40 statistics frames, no scoring
# Do not edit while a job runs it.
set -euo pipefail
MODE=${1:-full}
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
CFG=cfgs/da-ieee-access-analysis/item60/centerpoint-item60-pk-rsbn.yaml
OUT=/home/koyama/data/item60
run() { singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF" "$@" < /dev/null; }
declare -A JOBS=([666]=37227 [667]=37228)
SEEDS="666 667"; EXTRA=(); [ "$MODE" = smoke ] && { SEEDS=666; EXTRA=(--smoke --stats_frames 40); }
for SEED in $SEEDS; do
  JOB=${JOBS[$SEED]}
  C=$(bash analysis/ckpt_of_job.sh "$JOB" 152)
  if [ "$MODE" = smoke ]; then out=$OUT/smoke/glob_s$SEED; else out=$(python3 analysis/item60g_pooled.py dir "$SEED"); fi
  echo "=== item60g glob seed $SEED, job $JOB, checkpoint $C -> $out ($(date '+%F %T')) ==="
  WANDB_NOTES="Item 60g (T05 §5, design 2026-11-28 10:38 JST), labelled ANALYSIS: item 60's RS-BN row $JOB seed $SEED (epoch 152) under ONE pooled set of BN statistics (global AdaBN on the RS-BN model; 1000 strided KITTI train clouds, no labels), KITTI val. Readings fixed in T05 item 60g.$([ "$MODE" = smoke ] && echo ' SMOKE: 16 frames.')" \
    run python3 analysis/item60_rsbn_eval.py --cfg_file "$CFG" --ckpt "$C" --adapt glob --out_dir "$out" \
      --run_name "i60g_glob_s${SEED}$([ "$MODE" = smoke ] && echo _smoke)" "${EXTRA[@]}"
done
[ "$MODE" = smoke ] && { echo "=== item60g smoke done ($(date '+%F %T')) ==="; exit 0; }
run python3 analysis/item60g_pooled.py manifest $OUT/item60g_manifest.json --jobs s666=37227 s667=37228
run python3 analysis/item38_kitti_rings.py ap $OUT/item60g_manifest.json $OUT/item60g_ap.json --real-kernel --workers 4
run python3 analysis/item60g_pooled.py read $OUT/item60g_ap.json
echo "=== item60g done ($(date '+%F %T')) ==="
