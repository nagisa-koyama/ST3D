#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 47 (design fixed 2026-11-21 19:30 JST, before any number): the structure term in the other
# direction. Item 42a's checkpoints scored on S', the dense sensor's own clouds thinned to nuScenes' lattice by P's own training
# cut, with stats `none` (their own BN statistics) and `target` (full AdaBN on 1,000 strided frames of S' TRAIN, thinned the
# same way, eval-mode view, no labels); part B: the Waymo->nuScenes and PandaSet->nuScenes P / L / C1 rows on nuScenes val, both
# ways. Rows, configs and output tags come from analysis/make_item47_configs.py (ROWS); tests/test_item47_configs.py pins the
# configs. Usage (from ST3D/tools, inside a 1-GPU allocation):
#   bash scripts/_item47_evals_20261121.sh K P      # parts in order: K KITTI (first row = the smoke, P_K 32283 `none`),
#                                                   # P Pandar64 (rule A by a CPU successor), W Waymo, BW / BP nuScenes val
# Each row resolves its checkpoint from the training job's log (analysis/ckpt_of_job.sh) and logs its own exit status; the next
# row runs regardless; the script exits 1 if any row failed. Do not edit while a job runs it. ANALYSIS: target labels score only.
set -u
cd /home/koyama/code/ST3D/tools
[ $# -gt 0 ] || { echo "usage: _item47_evals_20261121.sh <K|P|W|BW|BP> ..." >&2; exit 2; }
H="Item 47 (T05 §5 item 47, design fixed 2026-11-21 19:30 JST), labelled ANALYSIS, eval-only"
declare -A TGT=([K]="S' = KITTI val thinned to nuScenes' lattice (P_K's RING_THIN pattern; moderate, camera-FOV filter)" \
                [W]="S' = Waymo val (every 5th frame) thinned to nuScenes' lattice (P's RING_PATTERN, TOP only)" \
                [P]="S' = Pandar64 spin val thinned to nuScenes' lattice (P's RING_PATTERN; rule A by the CPU successor)" \
                [BW]="nuScenes val (Waymo->nuScenes row, part B, beside)" [BP]="nuScenes val (PandaSet->nuScenes row, part B, beside)")
declare -A ST=([none]="with its own BN statistics (no re-estimation)" \
               [target]="after full AdaBN, statistics from 1000 strided frames of the same target's TRAIN split in the eval-mode view (thinned like its val), no labels")
F=0
for part in "$@"; do
  rows=$(python3 analysis/make_item47_configs.py --rows "$part") || { echo "item47: no rows for part $part"; F=$((F+1)); continue; }
  while read -r tag cfg job epoch stats; do
    [ -n "$tag" ] || continue
    echo "=== item47 $part $tag (train job $job, epoch $epoch, stats $stats, $cfg) ($(date '+%F %T')) ==="
    C=$(bash analysis/ckpt_of_job.sh "$job" "$epoch") || { echo "exit 2 (no checkpoint)"; F=$((F+1)); continue; }
    echo "checkpoint $C"
    WANDB_NOTES="$H: $tag (train job $job, epoch $epoch) on ${TGT[$part]}, ${ST[$stats]}. structure_rev = P - L paired by seed, with and without AdaBN; readings fixed in T05 item 47." \
      bash analysis/adabn_eval.sh "$cfg" "$C" "$stats" 1.0 "$tag" --extra_tag "$tag" --eval_tag "$tag" < /dev/null
    s=$?; echo "exit $s"; [ $s -eq 0 ] || F=$((F+1))
  done <<< "$rows"
done
echo "=== item47 done: parts $* failures $F ($(date '+%F %T')) ==="
[ $F -eq 0 ]
