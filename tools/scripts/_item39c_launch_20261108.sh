#!/bin/bash
# experiments_md 20261108_03 §2 C (T05 §5 item 39; design fixed 2026-11-08 11:12 JST, before any number). One row per call:
#     bash scripts/_item39c_launch_20261108.sh <teacher seed 666|667|668> [student seed, default 666]
# C1 = teacher 666 / 667 / 668 with student seed 666 (only the teacher and its cuts differ); C2 (proposed only) = teacher 666
# with student seed 667 / 668. Pseudo-labelling from the S2 source-only control seed (27566's recipe at the control's single
# sweep); teacher = init = that seed's epoch-20 checkpoint; layout as 27566 (2 GPUs x 10 CPUs, 150G). ANALYSIS: no target
# label enters any value; KITTI val scores the row.
set -euo pipefail
cd /home/koyama/code/ST3D/tools
T=${1:?teacher seed}; S=${2:-666}
W=/home/koyama/data/wandb
case $T in
  666) CK=$W/run-20260922_155434-wfdcq75s/files/ckpt/checkpoint_epoch_20.pth; TJ=25743 ;;
  667) CK=$W/run-20261105_122438-ftp96tq4/files/ckpt/checkpoint_epoch_20.pth; TJ=31818 ;;
  668) CK=$W/run-20261107_205850-kovce887/files/ckpt/checkpoint_epoch_20.pth; TJ=32207 ;;
  *) echo "unknown teacher seed $T" >&2; exit 2 ;;
esac
C=cfgs/da-ieee-access-analysis/item39/centerpoint-control-st3d-t$T-nuscenes2kitti.yaml
[ -f $C ] && [ -f $CK ] || { echo "missing $C or $CK" >&2; exit 2; }
if /usr/bin/grep -q TBD $C; then echo "$C still has TBD cuts" >&2; exit 2; fi
EXTRA=""; [ "$S" = 666 ] || EXTRA="--seed $S"
EST_H=24 WANDB_NOTES="Item 39 C (experiments_md 20261108_03 §2, design fixed 2026-11-08 11:12 JST), ANALYSIS: pseudo-labelling from the UNSTABLE S2 source-only control seed $T ($TJ epoch 20; frozen teacher = init), 27566's recipe at the control's single sweep, cuts re-derived for this teacher by rules A/B; student seed $S. Reading: spread of the three C1 students <= 3 vs the teachers' 9.81 -> PL removes the extrapolation spread; >= 6 -> inherits it." \
  scripts/submit.sh --partition=a100,a6000_ada,a6000,rtx8000 --gres=gpu:2 --cpus-per-task=20 --mem=150G \
  --time=1-06:00:00 --exclude=node03 --job-name=i39cT${T: -1}S${S: -1} \
  scripts/run_sourceonly_2gpu.sh $C 20261108_i39c_t${T}_s$S control-st3d-t$T-s$S \
  --pretrained_model ${CK/#\/home\/koyama\/data\//\/storage\/} --pretrained_model_teacher ${CK/#\/home\/koyama\/data\//\/storage\/} $EXTRA < /dev/null
