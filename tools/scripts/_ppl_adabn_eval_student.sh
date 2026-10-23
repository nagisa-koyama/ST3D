#!/usr/bin/env bash
# P-PL test-time AdaBN, STUDENT arms only (experiments_md 20261022_07 §10, the seed-667 row; T05 §5 "User 10-24 ~02:40" item 1).
# Same procedure as scripts/_ppl_adabn_eval.sh (job 30231) for the student: `analysis/adabn_eval.sh` on the P-PL config,
# arms `none` and `target` (every BN layer re-estimated on 1,000 strided nuScenes TRAIN frames, MIX 1.0), 1 GPU, no training.
# The teacher arms (28034 ep115: none 33.93, target 33.28) are NOT re-run: the teacher is the same checkpoint and the fixed
# design reuses 30231's `target` value. The only difference from _ppl_adabn_eval.sh's student arms is `--extra_tag
# adabn_job<row>`, which only moves the output directory so the seed-666 student's eval files (extra_tag `adabn`) are not
# overwritten. The student's checkpoint is resolved from the row's log at run time (ckpt_of_job.sh).
# Usage (inside an sbatch allocation, 1 GPU):  bash scripts/_ppl_adabn_eval_student.sh <P-PL row job id>
# Do not edit while a job runs it.
set -u -o pipefail
ROW=${1:?usage: _ppl_adabn_eval_student.sh <P-PL row job id>}
cd /home/koyama/code/ST3D/tools
echo "=== live HEAD $(git rev-parse --short HEAD) ==="
S_CFG=cfgs/da-ieee-access/centerpoint-ringpattern-st3d-pandaset2nuscenes.yaml
S_HOST=$(bash analysis/ckpt_of_job.sh "$ROW" 115) || { echo "no epoch-115 checkpoint of $ROW"; exit 2; }
S_CKPT=${S_HOST/\/home\/koyama\/data/\/storage}
echo "student checkpoint: $S_CKPT"
rc=0
for arm in none target; do
  bash analysis/adabn_eval.sh $S_CFG $S_CKPT $arm 1.0 ppl_student${ROW}_$arm --extra_tag adabn_job${ROW}; r=$?
  echo "student $arm exit $r"; [ $r -eq 0 ] || rc=$r
done
echo "=== done (exit $rc) ==="
exit $rc
