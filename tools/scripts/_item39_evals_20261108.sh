#!/usr/bin/env bash
# experiments_md 20261108_03 §2 (T05 §5 item 39; design fixed 2026-11-08 11:13 JST): the exempt 1-GPU parts, run in order inside
# one allocation. Usage (from ST3D/tools, inside an sbatch allocation with 1 GPU):
#   bash scripts/_item39_evals_20261108.sh g1   # C cuts for the three control teachers, then B1 AdaBN, then B2 own-density
#   bash scripts/_item39_evals_20261108.sh g2   # A census (log parsing, CPU), then B3 N->W seeds on Waymo val
# Each step logs its own exit status and the next one runs regardless. Do not edit while a job runs it. ANALYSIS.
set -u
cd /home/koyama/code/ST3D/tools
PART=${1:?usage: _item39_evals_20261108.sh g1|g2}
W=/home/koyama/data/wandb
declare -A CK=([666]=$W/run-20260922_155434-wfdcq75s/files/ckpt/checkpoint_epoch_20.pth
               [667]=$W/run-20261105_122438-ftp96tq4/files/ckpt/checkpoint_epoch_20.pth
               [668]=$W/run-20261107_205850-kovce887/files/ckpt/checkpoint_epoch_20.pth)
CTRL=cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes2kitti.yaml
I39=cfgs/da-ieee-access-analysis/item39
step() { echo "=== item39 $* ($(date '+%F %T')) ==="; }
if [ "$PART" = g1 ]; then
  for s in 666 667 668; do step "C cuts teacher $s"; bash scripts/_item39_cuts.sh ${CK[$s]} control_s$s; echo "exit $?"; done
  for s in 666 667 668; do
    step "B1 AdaBN target seed $s"
    bash analysis/adabn_eval.sh $CTRL ${CK[$s]} target 1.0 i39_adabn_s$s --extra_tag i39_adabn_s$s --eval_tag i39_adabn_s$s
    echo "exit $?"
  done
  for tw in r3-density r3-halfway p-pattern; do
    for s in 666 667 668; do
      step "B2 $tw seed $s"
      bash analysis/eval_checkpoint.sh $I39/centerpoint-control-eval-$tw.yaml ${CK[$s]} i39_${tw}_s$s 20261108_i39
      echo "exit $?"
    done
  done
elif [ "$PART" = g2 ]; then
  step "A census"
  mkdir -p /home/koyama/data/seed_census/v1
  singularity exec --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif \
    python3 analysis/seed_census_from_logs.py --out /home/koyama/data/seed_census/v1/census.json \
    25743 31818 32207 32206 32208 32210 32211 26843 31817 27263 27656 27468 27804 27471 27619 27566 28283 27612 28284 \
    27570 27602 27174 28650 27704 28563 27590 27596 27178 25835 27172 25723 29707 26695 26838 27173 26696 26716 26809 \
    27594 28304 27592 28394 27807 27840 27850 28278 28034 28279 28041 28280 29327 29602 29708 29755 29328 29603 29330 \
    29701 29702 29756 32253
  echo "exit $?"
  step "B3 Waymo: teacher 27468 seed 666 (drift check against 27590's 48.30)"
  bash analysis/eval_checkpoint.sh cfgs/da-ieee-access/centerpoint-accum-legaldepth-nuscenes2waymo.yaml \
    $W/run-20261006_004144-o3l96a3p/files/ckpt/checkpoint_epoch_20.pth i39_waymo_t27468 20261108_i39; echo "exit $?"
  step "B3 Waymo: teacher 27804 (seed 667 of 27468)"
  bash analysis/eval_checkpoint.sh cfgs/da-ieee-access/centerpoint-accum-legaldepth-nuscenes2waymo.yaml \
    $W/run-20261008_202000-udtv9tvw/files/ckpt/checkpoint_epoch_20.pth i39_waymo_t27804 20261108_i39; echo "exit $?"
  step "B3 Waymo: teacher 27619 (seed 667 of 27471)"
  bash analysis/eval_checkpoint.sh cfgs/da-ieee-access/centerpoint-gblobs-sourceonly-nuscenes2waymo.yaml \
    $W/run-20261007_164839-6b66y4bl/files/ckpt/checkpoint_epoch_20.pth i39_waymo_t27619 20261108_i39; echo "exit $?"
  for s in 667 668; do
    step "B3 Waymo: control seed $s"
    bash analysis/eval_checkpoint.sh cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes2waymo.yaml ${CK[$s]} \
      i39_waymo_ctrl_s$s 20261108_i39; echo "exit $?"
  done
fi
step "done"
