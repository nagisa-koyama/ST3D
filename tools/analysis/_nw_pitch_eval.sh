#!/usr/bin/env bash
# GPU job for NW-STRAT S1 (experiments_md 20261027_03; design 20261003_05 §5 item 23, FIXED 2026-10-27 21:15 JST; ANALYSIS,
# Waymo val labels at evaluation only): 27704's epoch-20 checkpoint scored on Waymo val as is ('asscored') and under its
# eval-only pitch twins (cfgs/da-ieee-access-analysis/nw_strat/), one evaluation at a time through
# analysis/eval_checkpoint.sh, in the order given. Each evaluation creates its own W&B run (eval tag nwpitch_<arm>, extra
# tag 20261027_nwpitch). Runs on the host from ST3D/tools. Do not edit while a job runs it.
#   usage: bash analysis/_nw_pitch_eval.sh <arm> [<arm> ...]        arms: asscored, p218, p124, p176
set -u
BASE=centerpoint-gblobs-accum-legaldepth-st3d-nuscenes2waymo
CKPT=/home/koyama/data/wandb/run-20261008_030315-5hzcs4e4/files/ckpt/checkpoint_epoch_20.pth     # 27704 (5hzcs4e4)
for a in "$@"; do
  case "$a" in
    asscored) cfg=cfgs/da-ieee-access/$BASE.yaml ;;
    p218|p124|p176) cfg=cfgs/da-ieee-access-analysis/nw_strat/$BASE-nwpitch-$a.yaml ;;
    *) echo "unknown arm $a" >&2; exit 2 ;;
  esac
  echo "=== ARM $a START $(date '+%Y-%m-%d %H:%M:%S') $cfg"
  bash analysis/eval_checkpoint.sh "$cfg" "$CKPT" "nwpitch_$a" 20261027_nwpitch
  echo "=== ARM $a END rc=$? $(date '+%Y-%m-%d %H:%M:%S')"
done
echo "=== ALL ARMS DONE"
