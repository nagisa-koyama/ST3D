#!/usr/bin/env bash
# GPU job for OSENS-2 (experiments_md 20261025_03; ANALYSIS, target labels at evaluation only): one target oracle scored
# as is and under its eval-only twins (analysis/make_osens2_configs.py), one evaluation at a time through
# analysis/eval_checkpoint.sh, in the order given. Each evaluation creates its own W&B run (eval tag osens2_<oracle>_<arm>,
# extra tag 20261025_osens2). 'asscored_a' / 'asscored_b' score the oracle's own config (arm 0 twice).
# Runs on the host from ST3D/tools. Do not edit while a job runs it.
#   usage: bash analysis/_osens2_eval.sh <waymo|nuscenes|kitti> <arm> [<arm> ...]
set -u
ORACLE=${1:?oracle}; shift
case "$ORACLE" in
  waymo)    BASE=centerpoint-sourceonly-waymo2waymo
            CKPT=/home/koyama/data/wandb/run-20260928_082639-y6p61n0x/files/ckpt/checkpoint_epoch_7.pth ;;     # 26398
  nuscenes) BASE=centerpoint-sourceonly-nuscenes
            CKPT=/home/koyama/data/wandb/run-20260922_155434-wfdcq75s/files/ckpt/checkpoint_epoch_20.pth ;;    # 25743
  kitti)    BASE=centerpoint-sourceonly-kitti2kitti-noros-zshift
            CKPT=/home/koyama/data/wandb/run-20261001_005323-06iwywhs/files/ckpt/checkpoint_epoch_152.pth ;;   # 26843
  *) echo "unknown oracle $ORACLE" >&2; exit 2 ;;
esac
for a in "$@"; do
  case "$a" in
    asscored_*) cfg=cfgs/da-ieee-access/$BASE.yaml ;;
    *)          cfg=cfgs/da-ieee-access-analysis/osens2/$BASE-osens2-$a.yaml ;;
  esac
  echo "=== ARM $ORACLE $a START $(date '+%Y-%m-%d %H:%M:%S') $cfg"
  bash analysis/eval_checkpoint.sh "$cfg" "$CKPT" "osens2_${ORACLE}_$a" 20261025_osens2
  echo "=== ARM $ORACLE $a END rc=$? $(date '+%Y-%m-%d %H:%M:%S')"
done
echo "=== ALL ARMS DONE"
