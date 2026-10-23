#!/usr/bin/env bash
# SCENE, the chained evaluation of the PL B->S row (experiments_md/20261022_06 §4.4, §6.4): the row's epoch-20 student on
# Singapore val with the B->S source-only config, arm `none` (saved BN statistics) then arm `target` (test-time AdaBN, 1,000
# strided Singapore TRAIN frames, all BN layers, MIX 1.0; 29306's procedure). Same arms and run names as the --wrap job 30216
# it replaces: that one ran under /bin/sh (dash), which rejects the bash-only ${var/#..} substitution below.
#
#   sbatch --dependency=afterany:<PL row> --kill-on-invalid-dep=yes ... scripts/_scene_pl_adabn_chain.sh <PL row jobid>
#
# Both arms are attempted even if the first fails; the exit code is non-zero if either failed.
# Runs from the live checkout (as adabn_eval.sh always does). Do not edit while a job runs it.
set -u
ROW=${1:?usage: _scene_pl_adabn_chain.sh <PL row jobid>}
CFG=cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes-boston2singapore.yaml
cd /home/koyama/code/ST3D/tools
echo "=== live HEAD $(git -C /home/koyama/code/ST3D rev-parse --short HEAD) on $(hostname) at $(date '+%F %T') ==="
C=$(bash analysis/ckpt_of_job.sh "$ROW" 20) || { echo "no epoch-20 checkpoint for job $ROW" >&2; exit 2; }
CS=${C/#\/home\/koyama\/data//storage}
echo "student: $CS"
RC=0
bash analysis/adabn_eval.sh "$CFG" "$CS" none 1.0 "scene_pl_none_${ROW}"
R=$?; echo "eval PL B->S none exit $R"; [ "$R" -eq 0 ] || RC=$R
bash analysis/adabn_eval.sh "$CFG" "$CS" target 1.0 "scene_pl_adabn_${ROW}"
R=$?; echo "eval PL B->S target exit $R"; [ "$R" -eq 0 ] || RC=$R
echo "=== done $(date '+%F %T') rc=$RC ==="
exit "$RC"
