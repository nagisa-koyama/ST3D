#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 56 (design fixed 2026-11-27 00:02 JST, before any number): item 47's KITTI O and P checkpoints
# scored on the two halves of S' - S'_rows (pattern_rows: S''s rings, all their points) and S'_az (pattern_az: every ring at the
# 0.332 deg azimuth step) - with their own BN statistics (stats none). Rows: analysis/item56_halves.py EVAL_ROWS. Usage (from
# ST3D/tools, inside a 1-GPU allocation): bash scripts/_item56_evals_20261127.sh. Each row resolves its checkpoint from the
# training job's log (analysis/ckpt_of_job.sh) and logs its exit status; the next row runs regardless; exit 1 if any row failed.
# Do not edit while a job runs it. ANALYSIS: target labels score only.
set -u
cd /home/koyama/code/ST3D/tools
H="Item 56 (T05 §5 item 56, design fixed 2026-11-27 00:02 JST), labelled ANALYSIS, eval-only"
declare -A CLOUD=([rows]="S'_rows = KITTI val thinned to nuScenes' row lattice only (the S' rings, all their points)" \
                  [az]="S'_az = KITTI val thinned in azimuth only (every ring, one point per 0.332 deg bin, S' phase)")
F=0
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif   # the host python3 has no numpy
rows=$(singularity exec --bind /home/koyama/data/:/storage "$SIF" python3 analysis/item56_halves.py eval_rows) || { echo "item56: no rows"; exit 2; }
while read -r tag cfg job epoch arm; do
  [ -n "$tag" ] || continue
  echo "=== item56 $tag (train job $job, epoch $epoch, $cfg) ($(date '+%F %T')) ==="
  C=$(bash analysis/ckpt_of_job.sh "$job" "$epoch") || { echo "exit 2 (no checkpoint)"; F=$((F+1)); continue; }
  echo "checkpoint $C"
  WANDB_NOTES="$H: $tag (train job $job, epoch $epoch) on ${CLOUD[$arm]}, with its own BN statistics (no re-estimation). O's far loss on S' split into its vertical and azimuth steps; readings fixed in T05 item 56." \
    bash analysis/adabn_eval.sh "$cfg" "$C" none 1.0 "$tag" --extra_tag "$tag" --eval_tag "$tag" < /dev/null
  s=$?; echo "exit $s"; [ $s -eq 0 ] || F=$((F+1))
done <<< "$rows"
echo "=== item56 done: failures $F ($(date '+%F %T')) ==="
[ $F -eq 0 ]
