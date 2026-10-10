#!/usr/bin/env bash
# GPU job for experiments_md 20261011_02 (ANALYSIS): the KITTI oracle 26843 scored as is, then under each of its 21
# eval-only twins, in the pre-declared order, one evaluation at a time (analysis/eval_checkpoint.sh). Each evaluation
# creates its own W&B run, tagged with the arm. Runs on the host from ST3D/tools. Do not edit while a job runs it.
set -u
CKPT=/home/koyama/data/wandb/run-20261001_005323-06iwywhs/files/ckpt/checkpoint_epoch_152.pth
BASE=centerpoint-sourceonly-kitti2kitti-noros-zshift
D=cfgs/da-ieee-access-analysis/sensitivity
ARMS="asscored ratio075 ratio050 ratio025 ringrows2 ringrandom2 ringrows3 ringrandom3 ringrows4 ringrandom4 \
ringcols2 ringrandomcols2 ringcols4 ringrandomcols4 ringazbin0p332 ringrandomazbin0p332 ringpatternhdl32e \
elmax0 elmaxm3 elminm14 elminm10 elminm17p6"
for a in $ARMS; do
  if [ "$a" = asscored ]; then cfg=cfgs/da-ieee-access/$BASE.yaml; else cfg=$D/$BASE-$a.yaml; fi
  echo "=== ARM $a START $(date '+%H:%M:%S') $cfg"
  bash analysis/eval_checkpoint.sh "$cfg" "$CKPT" "kittisens_$a" 20261011_kitti_sens
  echo "=== ARM $a END rc=$? $(date '+%H:%M:%S')"
done
echo "=== ALL ARMS DONE"
