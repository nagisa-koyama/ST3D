#!/usr/bin/env bash
# P-PL test-time AdaBN evaluation (experiments_md 20261022_07 §3.3; 20261022_01's procedure), chained afterany on the P-PL
# row. Four arms on nuScenes val, each `analysis/adabn_eval.sh` (1 GPU, no training, no weight changed):
#   teacher 28034 ep115 (its own config) none / target;  student <row> ep115 (the P-PL config) none / target.
# `target` re-estimates every BN layer on 1,000 strided nuScenes TRAIN frames, MIX 1.0. The student's checkpoint is resolved
# from the row's log at run time (ckpt_of_job.sh); a missing checkpoint fails the job loudly after the teacher arms.
# Usage (inside an sbatch allocation, 1 GPU):  bash scripts/_ppl_adabn_eval.sh <P-PL row job id>
# Do not edit while a job runs it.
set -u -o pipefail
ROW=${1:?usage: _ppl_adabn_eval.sh <P-PL row job id>}
cd /home/koyama/code/ST3D/tools
echo "=== live HEAD $(git rev-parse --short HEAD) ==="
T_CFG=cfgs/da-ieee-access/centerpoint-ringpattern-pandaset2nuscenes.yaml
S_CFG=cfgs/da-ieee-access/centerpoint-ringpattern-st3d-pandaset2nuscenes.yaml
T_CKPT=/storage/wandb/run-20261010_193142-lf9c36yn/files/ckpt/checkpoint_epoch_115.pth
rc=0
for arm in none target; do
  bash analysis/adabn_eval.sh $T_CFG $T_CKPT $arm 1.0 ppl_teacher28034_$arm; r=$?; echo "teacher $arm exit $r"; [ $r -eq 0 ] || rc=$r
done
S_HOST=$(bash analysis/ckpt_of_job.sh "$ROW" 115) || { echo "no epoch-115 checkpoint of $ROW"; exit 2; }
S_CKPT=${S_HOST/\/home\/koyama\/data/\/storage}
echo "student checkpoint: $S_CKPT"
for arm in none target; do
  bash analysis/adabn_eval.sh $S_CFG $S_CKPT $arm 1.0 ppl_student${ROW}_$arm; r=$?; echo "student $arm exit $r"; [ $r -eq 0 ] || rc=$r
done
echo "=== done (exit $rc) ==="
exit $rc
