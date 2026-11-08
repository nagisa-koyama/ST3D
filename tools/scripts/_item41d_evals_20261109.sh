#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 41d (design fixed 2026-11-09 02:05 JST): the exempt 1-GPU evaluations of one seed of the
# S2 source-only control trained at 2 GPUs with SyncBN and a per-seed sampler, run as the afterany successor of its training job. Usage (from ST3D/tools, inside
# an sbatch allocation with 1 GPU):
#   bash scripts/_item41d_evals_20261109.sh <train jobid> <seed>
# Mirrors item 39 g1 (B1 AdaBN, B2 own-input twins) without the C cuts, plus full KITTI val at epochs 20 / 19 / 18.
# Each step logs its own exit status and the next one runs regardless. Do not edit while a job runs it. Evaluation of a
# UDA-legal row (nuScenes labels only); target labels are used for scoring only.
set -u
cd /home/koyama/code/ST3D/tools
J=${1:?usage: _item41_evals_20261108.sh <train jobid> <seed>}
S=${2:?usage: _item41_evals_20261108.sh <train jobid> <seed>}
CTRL=cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes2kitti.yaml
I39=cfgs/da-ieee-access-analysis/item39
step() { echo "=== item41d $* ($(date '+%F %T')) ==="; }
for e in 20 19 18; do
  step "full KITTI val, seed $S, epoch $e"
  C=$(bash analysis/ckpt_of_job.sh $J $e) && bash analysis/eval_checkpoint.sh $CTRL $C i41_kitti_2gpusync_s${S}_e$e 20261109_i41d
  echo "exit $?"
done
C=$(bash analysis/ckpt_of_job.sh $J 20) || { echo "item41d: no epoch-20 checkpoint of $J"; exit 2; }
step "B1 AdaBN target, seed $S"
bash analysis/adabn_eval.sh $CTRL $C target 1.0 i41_adabn_2gpusync_s$S --extra_tag i41_adabn_2gpusync_s$S --eval_tag i41_adabn_2gpusync_s$S
echo "exit $?"
for tw in r3-density r3-halfway p-pattern; do
  step "B2 $tw, seed $S"
  bash analysis/eval_checkpoint.sh $I39/centerpoint-control-eval-$tw.yaml $C i41_${tw}_2gpusync_s$S 20261109_i41d
  echo "exit $?"
done
step "done"
