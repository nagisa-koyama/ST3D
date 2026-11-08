#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 42a / 42a' (design fixed 2026-11-08 22:40 JST, before any number; report
# experiments_md 20261108_05): the factor-swap checkpoints rescored on their target's val after FULL AdaBN with statistics from
# the target's UNPERTURBED training clouds (analysis/adabn_eval.sh <cfg> <ckpt> target 1.0, as item 39 B1), and 42a': L_K on
# KITTI val thinned by its own RANGE_KEEP, without AdaBN (analysis/eval_checkpoint.sh, as item 33e). Rows, configs and output
# tags come from analysis/make_item42_configs.py (ROWS); tests/test_item42_configs.py checks that every AdaBN statistics block
# is the target oracle's DATA_CONFIG_TAR with no swap / thinning key. Usage (from ST3D/tools, inside a 1-GPU allocation):
#   bash scripts/_item42a_evals_20261108.sh K A P L     # parts in order: K KITTI (first row = the smoke, O 26843), A 42a',
#                                                       # P Pandar64 (rule A by a CPU successor), L Lyft-40, W Waymo
# Each row resolves its checkpoint from the training job's log (analysis/ckpt_of_job.sh) and logs its own exit status; the next
# row runs regardless; the script exits 1 if any row failed. Do not edit while a job runs it. ANALYSIS: target labels score only.
set -u
cd /home/koyama/code/ST3D/tools
[ $# -gt 0 ] || { echo "usage: _item42a_evals_20261108.sh <K|A|P|L|W> ..." >&2; exit 2; }
H="Item 42a (T05 §5 item 42, design fixed 2026-11-08 22:40 JST; report 20261108_05), labelled ANALYSIS, eval-only"
declare -A TGT=([K]="KITTI val (moderate, camera-FOV filter)" [W]="Waymo val (every 5th frame)" \
                [P]="Pandar64 spin val (rule A by the CPU successor)" [L]="Lyft-40 val (all objects)")
F=0
for part in "$@"; do
  rows=$(python3 analysis/make_item42_configs.py --rows "$part") || { echo "item42a: no rows for part $part"; F=$((F+1)); continue; }
  while read -r tag cfg job epoch; do
    [ -n "$tag" ] || continue
    echo "=== item42a $part $tag (train job $job, epoch $epoch, $cfg) ($(date '+%F %T')) ==="
    C=$(bash analysis/ckpt_of_job.sh "$job" "$epoch") || { echo "exit 2 (no checkpoint)"; F=$((F+1)); continue; }
    echo "checkpoint $C"
    if [ "$part" = A ]; then
      WANDB_NOTES="$H, 42a': L_K (train job $job, epoch $epoch) on KITTI val thinned by L_K's own TRAIN_FACTOR_SWAP RANGE_KEEP (EVAL_FACTOR_SWAP, same frame-seeded draw), NO AdaBN, as item 33e scored R3 at its own density. Reading: seed spread <= 3 -> L_K extrapolates as R1 / R3." \
        bash analysis/eval_checkpoint.sh "$cfg" "$C" "$tag" 20261108_i42ap < /dev/null
    else
      WANDB_NOTES="$H: $tag (train job $job, epoch $epoch) on ${TGT[$part]} after full AdaBN, statistics from 1000 strided frames of the target's unperturbed TRAIN split (DATA_CONFIG_TAR, eval-mode view, no labels). structure = L - P, pattern = C1 - P, density = O - C1, paired by seed." \
        bash analysis/adabn_eval.sh "$cfg" "$C" target 1.0 "$tag" --extra_tag "$tag" --eval_tag "$tag" < /dev/null
    fi
    s=$?; echo "exit $s"; [ $s -eq 0 ] || F=$((F+1))
  done <<< "$rows"
done
echo "=== item42a done: parts $* failures $F ($(date '+%F %T')) ==="
[ $F -eq 0 ]
