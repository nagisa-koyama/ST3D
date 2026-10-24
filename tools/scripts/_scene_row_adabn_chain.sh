#!/usr/bin/env bash
# SCENE, chained evaluation of one S->B adaptation row (experiments_md 20261022_06 §10): the row's epoch-20 checkpoint on Boston
# val with the S->B source-only config, arm `none` (saved BN statistics) then arm `target` (test-time AdaBN, 1,000 strided
# Boston TRAIN frames, all BN layers, MIX 1.0), as 30244 / 30307 - but with a PER-ROW --extra_tag, so the prediction files
# land in .../<cfg>/adabn_<row>/eval/... and never overwrite another row's (the adabn_eval overwrite trap, T13).
#
#   sbatch --dependency=afterany:<row> --kill-on-invalid-dep=yes ... scripts/_scene_row_adabn_chain.sh <row jobid> <short tag>
#
# Both arms are attempted even if the first fails; the exit code is non-zero if either failed.
# A bash script, never a --wrap (dash). Runs from the live checkout. Do not edit while a job runs it.
set -u
ROW=${1:?usage: _scene_row_adabn_chain.sh <row jobid> <short tag>}
TAG=${2:?usage: _scene_row_adabn_chain.sh <row jobid> <short tag>}
CFG=cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes-singapore2boston.yaml
cd /home/koyama/code/ST3D/tools
echo "=== live HEAD $(git -C /home/koyama/code/ST3D rev-parse --short HEAD) on $(hostname) at $(date '+%F %T') ==="
C=$(bash analysis/ckpt_of_job.sh "$ROW" 20) || { echo "no epoch-20 checkpoint for job $ROW" >&2; exit 2; }
CS=${C/#\/home\/koyama\/data//storage}
echo "checkpoint: $CS; extra_tag adabn_${ROW}"
RC=0
bash analysis/adabn_eval.sh "$CFG" "$CS" none 1.0 "scene_${TAG}_none_${ROW}" --extra_tag "adabn_${ROW}"
R=$?; echo "eval ${TAG} none exit $R"; [ "$R" -eq 0 ] || RC=$R
bash analysis/adabn_eval.sh "$CFG" "$CS" target 1.0 "scene_${TAG}_adabn_${ROW}" --extra_tag "adabn_${ROW}"
R=$?; echo "eval ${TAG} target exit $R"; [ "$R" -eq 0 ] || RC=$R
echo "=== done $(date '+%F %T') rc=$RC ==="
exit "$RC"
