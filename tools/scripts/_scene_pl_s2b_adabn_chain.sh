#!/usr/bin/env bash
# SCENE, the chained evaluation of the PL S->B row (experiments_md/20261022_06 §8): the row's epoch-20 student on Boston val
# with the S->B source-only config, arm `none` (saved BN statistics) then arm `target` (test-time AdaBN, 1,000 strided Boston
# TRAIN frames, all BN layers, MIX 1.0; 29306's procedure). The PL B->S twin is _scene_pl_adabn_chain.sh.
#
#   sbatch --dependency=afterany:<PL S->B row> --kill-on-invalid-dep=yes ... scripts/_scene_pl_s2b_adabn_chain.sh <row jobid>
#
# Both arms are attempted even if the first fails; the exit code is non-zero if either failed.
# A bash script, never a --wrap (dash rejects ${var/#..}). Runs from the live checkout. Do not edit while a job runs it.
set -u
ROW=${1:?usage: _scene_pl_s2b_adabn_chain.sh <PL S->B row jobid>}
CFG=cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes-singapore2boston.yaml
cd /home/koyama/code/ST3D/tools
echo "=== live HEAD $(git -C /home/koyama/code/ST3D rev-parse --short HEAD) on $(hostname) at $(date '+%F %T') ==="
C=$(bash analysis/ckpt_of_job.sh "$ROW" 20) || { echo "no epoch-20 checkpoint for job $ROW" >&2; exit 2; }
CS=${C/#\/home\/koyama\/data//storage}
echo "student: $CS"
RC=0
bash analysis/adabn_eval.sh "$CFG" "$CS" none 1.0 "scene_pls2b_none_${ROW}"
R=$?; echo "eval PL S->B none exit $R"; [ "$R" -eq 0 ] || RC=$R
bash analysis/adabn_eval.sh "$CFG" "$CS" target 1.0 "scene_pls2b_adabn_${ROW}"
R=$?; echo "eval PL S->B target exit $R"; [ "$R" -eq 0 ] || RC=$R
echo "=== done $(date '+%F %T') rc=$RC ==="
exit "$RC"
